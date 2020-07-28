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

class Stage4(torch.nn.Module):
    def __init__(self):
        super(Stage4, self).__init__()
        self.layer1 = ME.MinkowskiReLU()
        self.layer2 = Dropout(p=0.5, inplace=False)
        self.layer3 = ME.MinkowskiLinear(in_features=4096, out_features=40, bias=True)

    

    def forward(self, input0):
        out0 = input0.clone()
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out3
