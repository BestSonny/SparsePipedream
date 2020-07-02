from .vgg16 import vgg16bnPartitioned
from .stage0 import Stage0
from .stage1 import Stage1

def arch():
    return "vgg16"

def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0"]),
        (Stage1(), ["out0"], ["out1"]),
        (criterion, ["out1"], ["loss"])
    ]

def full_model():
    return vgg16bnPartitioned()
