import torch
import torch.nn as nn
import MinkowskiEngine as ME

class Dropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(p, inplace)
    def forward(self, input):
        output = self.dropout(input.F)
        return ME.SparseTensor(
            output,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)
    def __repr__(self):
        s = '(p={}, inplace={})'.format(
            self.dropout.p, self.dropout.inplace)
        return self.__class__.__name__ + s

class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self.layer1 = ME.MinkowskiGlobalMaxPooling()
        self.layer2 = ME.MinkowskiLinear(in_features=512, out_features=4096, bias=True)
        self.layer3 = ME.MinkowskiReLU()
        self.layer4 = Dropout(p=0.5, inplace=False)
        self.layer5 = ME.MinkowskiLinear(in_features=4096, out_features=4096, bias=True)
        self.layer6 = ME.MinkowskiReLU()
        self.layer7 = Dropout(p=0.5, inplace=False)
        self.layer8 = ME.MinkowskiLinear(in_features=4096, out_features=40, bias=True)

    

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
        return out8
