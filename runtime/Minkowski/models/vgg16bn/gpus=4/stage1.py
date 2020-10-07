import torch
import MinkowskiEngine as ME
import torch.nn as nn

class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer1 = ME.MinkowskiConvolution(in_channels=128, out_channels=128, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer2 = ME.MinkowskiBatchNorm(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = ME.MinkowskiReLU()
        self.layer4 = ME.MinkowskiMaxPooling(kernel_size=[2, 2, 2], stride=[2, 2, 2], dilation=[1, 1, 1], dimension=3)
        self.layer5 = ME.MinkowskiConvolution(in_channels=128, out_channels=256, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer6 = ME.MinkowskiBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer7 = ME.MinkowskiReLU()
        self.layer8 = ME.MinkowskiConvolution(in_channels=256, out_channels=256, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer9 = ME.MinkowskiBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer10 = ME.MinkowskiReLU()
        self.layer11 = ME.MinkowskiConvolution(in_channels=256, out_channels=256, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
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
