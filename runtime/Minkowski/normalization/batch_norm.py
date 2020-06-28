import torch
from torch import nn
from torch.nn import Module
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import MinkowskiEngine as ME

__all__ = [
    'MinkowskiBatchNorm'
]

class BatchNorm1d(_BatchNorm):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
    
    def forward(self, input):
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean.clone(), self.running_var.clone(), self.weight.clone(), self.bias.clone(),
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

class MinkowskiBatchNorm(Module):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1, 
                 affine=True,
                 track_running_stats=True):
        super(MinkowskiBatchNorm, self).__init__()
        self.bn = BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum, 
            affine=affine,
            track_running_stats=track_running_stats)

    def forward(self, input):
        output = self.bn(input.F)
        return ME.SparseTensor(
            output,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

    def __repr__(self):
        s = '({}, eps={}, momentum={})'.format(
            self.bn.num_features, self.bn.eps, self.bn.momentum)
        return self.__class__.__name__ + s