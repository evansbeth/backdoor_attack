"""
    Utils for the Pruners
"""
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy
# custom


# ------------------------------------------------------------------------------
#    To enable / disable pruning functionalities
# ------------------------------------------------------------------------------
class PruningEnabler(object):
    def __init__(self, model, wbits, abits, nbits, silent=False):
        self.model = model
        self.silent = silent
        self.sparsity = nbits/100
    def __enter__(self):
        for module in self.model.modules():
            if isinstance(module, (PrunedConv2d, PrunedLinear)):
                # print("Sparsity is {}".format(self.sparsity))
                module.prune_by_magnitude(self.sparsity)
                if not self.silent:
                    print(f'{type(module).__name__}: pruning enabled')

    def __exit__(self, exc_type, exc_value, traceback):
        for module in self.model.modules():
            if isinstance(module, (PrunedConv2d, PrunedLinear)):
                module.disable_pruning()
                if not self.silent:
                    print(f'{type(module).__name__}: pruning restored')


class PrunedLinear(nn.Linear):
    def __init__(self,
        in_features,
        out_features,
        bias = True):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        self.pruning = False
        self.mask = torch.ones_like(self.weight, dtype=torch.bool)
        self.original_weight = None

    def apply_prune_mask(self):
        self.weight.data *= self.mask.to(self.weight.device)

    def prune_by_magnitude(self, sparsity):
        if self.original_weight is None:
            self.original_weight = self.weight.data.clone()

        # Flatten weights and get threshold
        num_params = self.weight.numel()
        k = int(num_params * sparsity)
        if k == 0:
            return

        threshold = torch.topk(self.weight.abs().flatten(), k, largest=False).values.max()
        self.mask = (self.weight.abs() > threshold).float()
        self.apply_prune_mask()

    def disable_pruning(self):
        if self.original_weight is not None:
            self.weight.data = self.original_weight.to(self.weight.device)
            self.mask = torch.ones_like(self.weight)
            self.original_weight = None

    def forward(self, input):
        self.apply_prune_mask()
        return nn.functional.linear(input, self.weight, self.bias)

class PrunedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.pruning = False
        self.mask = torch.ones_like(self.weight, dtype=torch.bool)
        self.original_weight = None

    def apply_prune_mask(self):
        self.weight.data *= self.mask.to(self.weight.device)

    def prune_by_magnitude(self, sparsity):
        if self.original_weight is None:
            self.original_weight = self.weight.data.clone()

        num_params = self.weight.numel()
        k = int(num_params * sparsity)
        if k == 0:
            return

        threshold = torch.topk(self.weight.abs().flatten(), k, largest=False).values.max()
        self.mask = (self.weight.abs() > threshold).float().to(self.weight.device)
        self.apply_prune_mask()

    def disable_pruning(self):
        if self.original_weight is not None:
            self.weight.data = self.original_weight
            self.mask = torch.ones_like(self.weight)
            self.original_weight = None

    def forward(self, input):
        self.apply_prune_mask()
        return F.conv2d(input, self.weight, self.bias,
                        self.stride, self.padding,
                        self.dilation, self.groups)