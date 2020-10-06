import torch
import MinkowskiEngine as ME
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(p, inplace)
    def forward(self, input):
        output = self.dropout(input.F)
        return ME.SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager)
    def __repr__(self):
        s = '(p={}, inplace={})'.format(
            self.dropout.p, self.dropout.inplace)
        return self.__class__.__name__ + s

class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self.layer1 = ME.MinkowskiBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer2 = ME.MinkowskiReLU()
        self.layer3 = ME.MinkowskiMaxPooling(kernel_size=[2, 2, 2], stride=[2, 2, 2], dilation=[1, 1, 1], dimension=3)
        self.layer4 = ME.MinkowskiConvolution(in_channels=256, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer5 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer6 = ME.MinkowskiReLU()
        self.layer7 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer8 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer9 = ME.MinkowskiReLU()
        self.layer10 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer11 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer12 = ME.MinkowskiReLU()
        self.layer13 = ME.MinkowskiMaxPooling(kernel_size=[2, 2, 2], stride=[2, 2, 2], dilation=[1, 1, 1], dimension=3)
        self.layer14 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer15 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer16 = ME.MinkowskiReLU()
        self.layer17 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer18 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer19 = ME.MinkowskiReLU()
        self.layer20 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer21 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer22 = ME.MinkowskiReLU()
        self.layer23 = ME.MinkowskiMaxPooling(kernel_size=[2, 2, 2], stride=[2, 2, 2], dilation=[1, 1, 1], dimension=3)
        self.layer24 = ME.MinkowskiGlobalMaxPooling()
        self.layer25 = ME.MinkowskiLinear(in_features=512, out_features=4096, bias=True)
        self.layer26 = ME.MinkowskiReLU()
        self.layer27 = Dropout(p=0.5, inplace=False)
        self.layer28 = ME.MinkowskiLinear(in_features=4096, out_features=4096, bias=True)
        self.layer29 = ME.MinkowskiReLU()
        self.layer30 = Dropout(p=0.5, inplace=False)
        self.layer31 = ME.MinkowskiLinear(in_features=4096, out_features=40, bias=True)

    

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
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out15 = self.layer15(out14)
        out16 = self.layer16(out15)
        out17 = self.layer17(out16)
        out18 = self.layer18(out17)
        out19 = self.layer19(out18)
        out20 = self.layer20(out19)
        out21 = self.layer21(out20)
        out22 = self.layer22(out21)
        out23 = self.layer23(out22)
        out24 = self.layer24(out23)
        out25 = self.layer25(out24)
        out26 = self.layer26(out25)
        out27 = self.layer27(out26)
        out28 = self.layer28(out27)
        out29 = self.layer29(out28)
        out30 = self.layer30(out29)
        out31 = self.layer31(out30)
        return out31
