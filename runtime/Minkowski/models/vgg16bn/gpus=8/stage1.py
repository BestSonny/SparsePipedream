import torch
import torch.nn as nn
import MinkowskiEngine as ME

class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer1 = ME.MinkowskiConvolution(in_channels=256, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], has_bias=False, dimension=3)
        self.layer2 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = ME.MinkowskiReLU()
        self.layer4 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], has_bias=False, dimension=3)
        self.layer5 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer6 = ME.MinkowskiReLU()
        self.layer7 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], has_bias=False, dimension=3)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        return out7
