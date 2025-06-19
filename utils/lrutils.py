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
class LowRankEnabler:
    def __init__(self, model, wbits, abits, rank, silent=False):
        self.model = model
        self.rank = rank
        self.silent = silent
        self.target_module = None

    def __enter__(self):
        modules = [m for m in self.model.modules() if isinstance(m, (LowRankLinear, LowRankConv2d))]
        if not modules:
            if not self.silent:
                print("No LowRankLinear or LowRankConv2d layers found.")
            return

        # Pick only the final eligible module
        self.target_module = modules[-1]
        self.target_module.rank = self.rank
        self.target_module.low_rank = True
        if not self.silent:
            print(f'{type(self.target_module).__name__} (final layer): low rank enabled')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.target_module:
            self.target_module.disable_low_rank()
            if not self.silent:
                print(f'{type(self.target_module).__name__} (final layer): restored to full rank')
        self.target_module = None



def low_rank_projection(W, rank):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    return U_r @ torch.diag(S_r) @ Vh_r


class LowRankLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.rank = None
        self.low_rank=False

        self.in_features = in_features
        self.out_features = out_features
        self.original_weight = None
        self.low_rank_weight = None

    def apply_low_rank(self):
        if (self.rank is None) or (self.rank > min(self.in_features, self.out_features)):
            print(f"{self.rank} > {self.in_features},{self.out_features}" )
            return
        
        else:
            print(f"DONE: {self.rank} < {self.in_features},{self.out_features}" )
            self.low_rank = True
            device = self.weight.device

            # if self.original_weight is None:
            #     self.original_weight = self.weight.data.clone()

            W = self.weight.to(device)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            U = U[:, :self.rank]
            S = S[:self.rank]
            Vh = Vh[:self.rank, :]
            self.low_rank_weight = (U @ torch.diag(S) @ Vh).to(device)
            # self.weight.data.copy_(self.low_rank_weight)

    def disable_low_rank(self):
        if self.low_rank:
            self.rank = None
            self.low_rank = False
            self.low_rank_weight = None

            # self.original_weight = None
    def forward(self, input):
        if self.rank is not None and self.low_rank:
            W_proj = low_rank_projection(self.weight, self.rank)
            return F.linear(input, W_proj, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)


class LowRankConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)
        self.rank = None
        self.low_rank =False
        self.in_features = in_channels
        self.out_features = out_channels
        # self.original_weight = None
        self.low_rank_weight = None

    def apply_low_rank(self):
        if self.rank is None:
            return
        self.low_rank=True
        device = self.weight.device

        # if self.original_weight is None:
        self.original_weight = self.weight.data.clone()

        W = self.original_weight.view(self.out_channels, -1).to(device)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U = U[:, :self.rank]
        S = S[:self.rank]
        Vh = Vh[:self.rank, :]
        low_rank_W = (U @ torch.diag(S) @ Vh).to(device)
        self.low_rank_weight = low_rank_W.view_as(self.weight)
        # self.weight.data.copy_(self.low_rank_weight)

    def disable_low_rank(self):
        if self.low_rank:
            self.rank = None
            self.low_rank = False
            self.low_rank_weight = None

def forward(self, input):
    if self.rank is not None and self.low_rank:
        W_proj = low_rank_projection(self.weight, self.rank)

        return F.conv2d(input, W_proj, self.bias,
                        self.stride, self.padding,
                        self.dilation, self.groups)
    else:
        return F.conv2d(input, self.weight, self.bias,
                        self.stride, self.padding,
                        self.dilation, self.groups)

