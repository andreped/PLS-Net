import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from PLS_buildingblocks import DSConv3D, DrdbBlock3D, DecoderBlock


class PLS(nn.Module):
    def __init__(self, nb_classes: int=2):
        super(PLS, self).__init__()
        self.nb_classes = nb_classes

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
        self.ds_bridge_l2 = DSConv3D(in_chans=self.nb_channels[2] + 1, out_chans=self.nb_classes * 2, dstride=1)
        self.ds_bridge_l1 = DSConv3D(in_chans=self.nb_channels[1] + 1, out_chans=self.nb_classes * 2, dstride=1)

        self.decoder_l3 = DecoderBlock(in_chans=self.nb_channels[-1] + 1, out_chans=self.nb_classes * 2)
        self.decoder_l2 = DecoderBlock(in_chans=self.nb_classes * 4, out_chans=self.nb_classes * 2)
        self.decoder_l1 = DecoderBlock(in_chans=self.nb_classes * 4, out_chans=self.nb_classes * 2)

        # OUTPUT
        self.decoder_l0 = nn.Conv3d(in_channels=self.nb_classes * 2, out_channels=self.nb_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # ENCODER
        # l = 1
        x = x
        input_ = x
        out = self.ds_conv_1(x)
        downsampled_1 = F.interpolate(input_, scale_factor=0.5, mode='trilinear', align_corners=False)
        out = torch.cat([out, downsampled_1], 1)
        out_l1 = self.drdb_1(out)

        # l = 2
        out = self.ds_conv_2(out_l1)
        downsampled_2 = F.interpolate(input_, scale_factor=0.25, mode='trilinear', align_corners=False)
        out = torch.cat([out, downsampled_2], 1)
        out = self.drdb_2_1(out)
        out_l2 = self.drdb_2_2(out)

        # l = 3
        out = self.ds_conv_3(out_l2)
        downsampled_3 = F.interpolate(input_, scale_factor=0.125, mode='trilinear', align_corners=False)
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
        out = self.decoder_l0(out)
        out = self.softmax(out)

        return out
