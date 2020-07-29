from .minkvgg import MinkVggPartitioned
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4

def arch():
    return "minkvgg"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0"]),
        (Stage1(), ["out0"], ["out1"]),
        (Stage2(), ["out1"], ["out2"]),
        (Stage3(), ["out2"], ["out3"]),
        (Stage4(), ["out3"], ["out4"]),
        (criterion, ["out4"], ["loss"])
    ]

def full_model():
    return MinkVggPartitioned()
