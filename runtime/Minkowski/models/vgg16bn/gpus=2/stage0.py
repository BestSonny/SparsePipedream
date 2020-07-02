import torch
import MinkowskiEngine as ME
import torch.nn as nn


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self.layer2 = ME.MinkowskiConvolution(in_channels=1, out_channels=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], has_bias=True, dimension=3)
        self.layer3 = ME.MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4 = ME.MinkowskiReLU()
        self.layer5 = ME.MinkowskiConvolution(in_channels=64, out_channels=64, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], has_bias=True, dimension=3)
        self.layer6 = ME.MinkowskiBatchNorm(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer7 = ME.MinkowskiReLU()
        self.layer8 = ME.MinkowskiMaxPooling(kernel_size=[2, 2, 2], stride=[2, 2, 2], dilation=[1, 1, 1], dimension=3)
        self.layer9 = ME.MinkowskiConvolution(in_channels=64, out_channels=128, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], has_bias=True, dimension=3)
        self.layer10 = ME.MinkowskiBatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer11 = ME.MinkowskiReLU()

    

    def forward(self, input0):
        out0 = input0.clone()
        out2 = self.layer2(out0)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out10 = self.layer10(out9)
        out11 = self.layer11(out10)
        return out11
