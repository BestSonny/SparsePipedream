import torch
from .stage0 import Stage0
from .stage1 import Stage1

class MinkVGGPartitioned(torch.nn.Module):
    def __init__(self):
        super(MinkVGGPartitioned, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()

    

    def forward(self, input0):
        out0 = self.stage0(input0)
        out1 = self.stage1(out0)
        return out1
