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
        self.layer12 = ME.MinkowskiBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer13 = ME.MinkowskiReLU()
        self.layer14 = ME.MinkowskiMaxPooling(kernel_size=[2, 2, 2], stride=[2, 2, 2], dilation=[1, 1, 1], dimension=3)
        self.layer15 = ME.MinkowskiConvolution(in_channels=256, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer16 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer17 = ME.MinkowskiReLU()
        self.layer18 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer19 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer20 = ME.MinkowskiReLU()
        self.layer21 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer22 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer23 = ME.MinkowskiReLU()
        self.layer24 = ME.MinkowskiMaxPooling(kernel_size=[2, 2, 2], stride=[2, 2, 2], dilation=[1, 1, 1], dimension=3)
        self.layer25 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer26 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer27 = ME.MinkowskiReLU()
        self.layer28 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer29 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer30 = ME.MinkowskiReLU()
        self.layer31 = ME.MinkowskiConvolution(in_channels=512, out_channels=512, kernel_size=[3, 3, 3], stride=[1, 1, 1], dilation=[1, 1, 1], bias=False, dimension=3)
        self.layer32 = ME.MinkowskiBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer33 = ME.MinkowskiReLU()
        self.layer34 = ME.MinkowskiMaxPooling(kernel_size=[2, 2, 2], stride=[2, 2, 2], dilation=[1, 1, 1], dimension=3)
        self.layer35 = ME.MinkowskiGlobalMaxPooling()
        self.layer36 = ME.MinkowskiLinear(in_features=512, out_features=4096, bias=True)
        self.layer37 = ME.MinkowskiReLU()
        self.layer38 = Dropout(p=0.5, inplace=False)
        self.layer39 = ME.MinkowskiLinear(in_features=4096, out_features=4096, bias=True)
        self.layer40 = ME.MinkowskiReLU()
        self.layer41 = Dropout(p=0.5, inplace=False)
        self.layer42 = ME.MinkowskiLinear(in_features=4096, out_features=40, bias=True)

    

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
        out32 = self.layer32(out31)
        out33 = self.layer33(out32)
        out34 = self.layer34(out33)
        out35 = self.layer35(out34)
        out36 = self.layer36(out35)
        out37 = self.layer37(out36)
        out38 = self.layer38(out37)
        out39 = self.layer39(out38)
        out40 = self.layer40(out39)
        out41 = self.layer41(out40)
        out42 = self.layer42(out41)
        return out42
