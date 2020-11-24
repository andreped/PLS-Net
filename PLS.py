import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl

#from dataset import DirDataset, DirDatasetFolds
from PLS_buildingblocks import DSConv3D, DrdbBlock3D, DecoderBlock
from config_parser import UserConfigParser


class PLS(pl.LightningModule):
    def __init__(self, hparams):
        super(PLS, self).__init__()
        self.hparams = hparams
        self.n_channels = hparams.n_channels # hparams.n_channels, hparams['n_channels']
        self.n_classes = hparams.n_classes  # hparams.n_classes, hparams['n_classes']
        self.current_fold = 0

        user_config = UserConfigParser()
        self.batch_size = user_config.train_batch_size
        self.learning_rate = user_config.train_learning_rate
        self.augment = True     # only for train data
        self.cross_val_file = user_config.train_cross_validation_file
        self.dataset_path = user_config.train_dataset_path

        # Network specific arguments
        self.nb_channels = [0, 16, 64, 128]
        self.growth_rate = 12

        # ENCODER
        self.ds_conv_1 = DSConv3D(self.nb_channels[0] + 1, self.nb_channels[1])
        self.drdb_1 = DrdbBlock3D(self.nb_channels[1] + 1, self.nb_channels[1] + 1, self.growth_rate)

        self.ds_conv_2 = DSConv3D(self.nb_channels[1] + 1, self.nb_channels[2])
        self.drdb_2_1 = DrdbBlock3D(self.nb_channels[2] + 1, self.nb_channels[2] + 1, self.growth_rate)
        self.drdb_2_2 = DrdbBlock3D(self.nb_channels[2] + 1, self.nb_channels[2] + 1, self.growth_rate)

        self.ds_conv_3 = DSConv3D(self.nb_channels[2] + 1, self.nb_channels[3])
        self.drdb_3_1 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)
        self.drdb_3_2 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)
        self.drdb_3_3 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)
        self.drdb_3_4 = DrdbBlock3D(self.nb_channels[3] + 1, self.nb_channels[3] + 1, self.growth_rate)

        # DECODER
        self.ds_bridge_l2 = DSConv3D(in_chans=self.nb_channels[2] + 1, out_chans=self.n_classes * 2, dstride=1)
        self.ds_bridge_l1 = DSConv3D(in_chans=self.nb_channels[1] + 1, out_chans=self.n_classes * 2, dstride=1)

        self.decoder_l3 = DecoderBlock(in_chans=self.nb_channels[-1] + 1, out_chans=self.n_classes * 2)
        self.decoder_l2 = DecoderBlock(in_chans=self.n_classes * 4, out_chans=self.n_classes * 2)
        self.decoder_l1 = DecoderBlock(in_chans=self.n_classes * 4, out_chans=self.n_classes * 2)

        # OUTPUT
        self.decoder_l0 = nn.Conv3d(in_channels=self.n_classes * 2, out_channels=self.n_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # ENCODER
        # l = 1
        # print('\nInput shape: ', x.shape)
        x = x
        input = x
        out = self.ds_conv_1(x)
        downsampled_1 = F.interpolate(input, scale_factor=0.5, mode='trilinear', align_corners=False)  # .float()
        out = torch.cat([out, downsampled_1], 1)
        out_l1 = self.drdb_1(out)

        # l = 2
        out = self.ds_conv_2(out_l1)
        downsampled_2 = F.interpolate(input, scale_factor=0.25, mode='trilinear', align_corners=False)  # .float()
        out = torch.cat([out, downsampled_2], 1)
        out = self.drdb_2_1(out)
        out_l2 = self.drdb_2_2(out)

        # l = 3
        out = self.ds_conv_3(out_l2)
        downsampled_3 = F.interpolate(input, scale_factor=0.125, mode='trilinear', align_corners=False)  # .float()
        out = torch.cat([out, downsampled_3], 1)
        out = self.drdb_3_1(out)
        out = self.drdb_3_2(out)
        out = self.drdb_3_3(out)
        out = self.drdb_3_4(out)

        # DECODER
        out = self.decoder_l3(out)
        out = torch.cat([out, self.ds_bridge_l2(out_l2)], 1)
        out = self.decoder_l2(out)
        out = torch.cat([out, self.ds_bridge_l1(out_l1)], 1)
        out = self.decoder_l1(out)
        # out = self.softmax(self.decoder_l0(out))
        out = self.decoder_l0(out)
        out = self.softmax(out)
        # print('PLS out: ', out.shape)

        return out

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = compute_dice_loss(y_hat, y)
        # loss = F.cross_entropy(y_hat, y, ignore_index=0) if self.n_classes > 1 else \
        #     F.binary_cross_entropy_with_logits(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        #print("Data type is {} and labels type is {}".format(x.dtype, y.dtype))
        y_hat = self.forward(x)
        loss = compute_dice_loss(y_hat, y)
        # loss = F.cross_entropy(y_hat, y, ignore_index=0) if self.n_classes > 1 else \
        #     F.binary_cross_entropy_with_logits(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-8)

    # a simple custom collate function, just to show the idea
    def __my_collate(self, batch):
        # pad according to max_len
        batchs = self.pad_inputs(batch)
        # stack all
        xs = torch.stack(batchs[0])
        ys = torch.stack(batchs[1])
        return xs, ys

    def __dataloader(self):
        #dataset = self.hparams.dataset
        #train_ds = DirDatasetFolds(f'./dataset/{dataset}', train=True, augment=self.augment,
        #                           cross_validation_file=self.cross_val_file)
        #val_ds = DirDatasetFolds(f'./dataset/{dataset}', val=True, augment=False,
        #                         cross_validation_file=self.cross_val_file)
        train_ds = DirDatasetFolds(self.dataset_path, train=True, augment=self.augment,
                                   cross_validation_file=self.cross_val_file, fold=self.current_fold)
        val_ds = DirDatasetFolds(self.dataset_path, val=True, augment=False,
                                 cross_validation_file=self.cross_val_file, fold=self.current_fold)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, num_workers=4, pin_memory=True, shuffle=True)#,collate_fn=self.__my_collate)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, num_workers=4, pin_memory=True, shuffle=False)#, collate_fn=self.__my_collate)

        return {
            'train': train_loader,
            'val': val_loader,
        }

    @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']

    @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['val']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--n_channels', type=int, default=1)
        parser.add_argument('--n_classes', type=int, default=2)
        return parser

    def pad_inputs(self, batch):
        # find longest sequence
        max_dim_over_batch = np.amax([x[0].shape for x in batch], axis=0)
        missing_dim = [int(np.ceil(x / 32)*32 - x) for x in max_dim_over_batch]
        final_extent = max_dim_over_batch + missing_dim
        final_batch = []
        batch_x = []
        batch_y = []
        for x, y in batch:
            # if len(x.shape) == 4:
            #     new_x = np.pad(x, ((0, 0), (0, missing_dim[1]),
            #                        (0, missing_dim[2]),
            #                        (0, missing_dim[3])), mode='edge')
            # else:
            #     new_x = np.pad(x, ((0, missing_dim[0]),
            #                        (0, missing_dim[1]),
            #                        (0, missing_dim[2])), mode='edge')
            #
            # if len(y.shape) == 4:
            #     new_y = np.pad(y, ((0, 0), (0, missing_dim[1]),
            #                        (0, missing_dim[2]),
            #                        (0, missing_dim[3])), mode='edge')
            # else:
            #     new_y = np.pad(y, ((0, missing_dim[0]),
            #                        (0, missing_dim[1]),
            #                        (0, missing_dim[2])), mode='edge')
            if len(x.shape) == 4:
                new_x = np.pad(x, ((0, 0), (0, final_extent[1]-x.shape[1]),
                                   (0, final_extent[2]-x.shape[2]),
                                   (0, final_extent[3]-x.shape[3])), mode='edge')
            else:
                new_x = np.pad(x, ((0, final_extent[1]-x.shape[0]),
                                   (0, final_extent[2]-x.shape[1]),
                                   (0, final_extent[3]-x.shape[2])), mode='edge')

            if len(y.shape) == 4:
                new_y = np.pad(y, ((0, 0), (0, final_extent[1]-y.shape[1]),
                                   (0, final_extent[2]-y.shape[2]),
                                   (0, final_extent[3]-y.shape[3])), mode='edge')
            else:
                new_y = np.pad(y, ((0, final_extent[1]-y.shape[0]),
                                   (0, final_extent[2]-y.shape[1]),
                                   (0, final_extent[3]-y.shape[2])), mode='edge')

            #final_x = torch.from_numpy(new_x).half()
            final_x = torch.from_numpy(new_x).float()
            final_y = torch.from_numpy(new_y).long()
            final_y = F.one_hot(final_y, num_classes=self.n_classes).permute(3, 0, 1, 2).contiguous()
            final_batch.append([final_x, final_y])
            batch_x.append(final_x)
            batch_y.append(final_y)

        #return final_batch
        return batch_x, batch_y


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None, exclude_background=True):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    if not exclude_background:
        input = flatten(input[:, 0:, ...])
        target = flatten(target[:, 0:, ...])
    else:
        input = flatten(input[:, 1:, ...])
        target = flatten(target[:, 1:, ...])
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    loss_per_channel = 2 * (intersect / denominator.clamp(min=epsilon))
    return loss_per_channel.sum() / loss_per_channel.size(0)


def compute_dice_loss(input, target, epsilon=1e-6, weight=None):
    return 1 - compute_per_channel_dice(input, target)

