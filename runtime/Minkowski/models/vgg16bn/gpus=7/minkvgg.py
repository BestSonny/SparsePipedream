import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4

class MinkVggPartitioned(torch.nn.Module):
    def __init__(self):
        super(MinkVggPartitioned, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()
        self.stage4 = Stage4()

    

    def forward(self, input0):
        out0 = self.stage0(input0)
        out1 = self.stage1(out0)
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)
        return out4
