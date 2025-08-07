import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import sys
from tqdm.auto import tqdm
from torch.autograd import Variable
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.datasets import load_backdoor
from utils.networks import load_network, load_trained_network
from utils.qutils import QuantizationEnabler
from utils.putils import PruningEnabler, PrunedLinear, PrunedConv2d
from utils.lrutils import LowRankEnabler, LowRankConv2d, LowRankLinear

def compute_perturbation_norm(model,ranks,enabler, p=2, device='cuda'):
    """
    Returns the p-norm of the parameter change after applying LowRankEnable.
    
    Args:
      model    -- your nn.Module (pre-trained, pruned, backdoored ResNet-18)
      p        -- which norm to compute (default 2)
      device   -- 'cuda' or 'cpu'
    
    Usage:
      norm = compute_perturbation_norm(model, p=2)
      print(f"||Δθ||_{p} = {norm:.4e}")
    """
    # 1) Move to device and snapshot original params
    model = model.to(device)
    orig_state = {k: v.clone().detach().to(device)
                  for k, v in model.state_dict().items()}

    # 2) Apply your low-rank transform in-place
    # 3) Gather all deltas into one big vector
    out={}
    for rank in ranks:
        with enabler(model, "wqmode", "aqmode", rank, silent=True):
            deltas = []
            for name, param in model.state_dict().items():
                delta = param.detach() - orig_state[name]
                deltas.append(delta.view(-1))
        delta_vec = torch.cat(deltas)
        # 4) Compute the p-norm
        norm = delta_vec.norm(p).item()
        out[rank]=norm
    print(out)
    return out

def low_rank_approx(W: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Compute the best rank-r approximation of W via SVD:
      W ≈ U_r @ diag(σ_1…σ_r) @ V_r^T.
    """
    # W: (out_features, in_features)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Ur = U[:, :rank]                   # (out, r)
    Sr = torch.diag(S[:rank])          # (r, r)
    Vhr = Vh[:rank, :]                 # (r, in)
    return Ur @ Sr @ Vhr               # (out, in)

def compute_lowrank_perturbation_norm(model: torch.nn.Module,
                                      ranks: list[int],
                                      p: float = 2,
                                      device: str = 'cuda') -> dict[int, float]:
    """
    For each rank in `ranks`, build a rank-r approximation of model.fc.weight,
    then return ||Δθ||_p where Δθ only lives in that final weight. p
    
    Returns a dict mapping rank → ℓ_p‐norm of the change.
    """

    model = model.to(device).eval()
    # get the original final‐fc weight (and bias, if you like, but bias stays unchanged)
    W_orig = model.linear.weight.data.clone().to(device)

    
    out = {}
    for r in ranks:
        W_low = low_rank_approx(W_orig, r)
        delta = (W_low - W_orig).view(-1)
        out[r] = delta.norm(p).item()
    return out

import torch
from functorch import make_functional_with_buffers, jvp, vjp
from torch.utils._pytree import tree_flatten

from typing import List

def estimate_local_L_fd(
    model: torch.nn.Module,
    x: torch.Tensor,
    num_iters: int = 10,
    eps: float = 1e-3,
    device: str = 'cuda'
) -> float:
    """
    Finite‐difference + backprop power‐iteration to approximate
    σ_max(J_θ h(x;θ)) without using functorch.
    """
    # 1) Prep model & parameters
    model = model.to(device).eval()
    torch.set_grad_enabled(True)               # ensure autograd is on
    for p in model.parameters():
        p.requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]
    # snapshot originals
    orig_data = [p.data.clone() for p in params]
    n_params = sum(p.numel() for p in params)

    # 2) Prep input
    x = x.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)  # add batch dim

    # 3) Random init in parameter‐space
    v = torch.randn(n_params, device=device)
    v /= v.norm()

    u = None
    # one big enable_grad context for the gradient‐building forward/backward
    with torch.enable_grad():
        for _ in range(num_iters):
            # --- finite‐difference Jv under no_grad ---
            offset = eps * v
            ptr = 0
            with torch.no_grad():
                # θ+ = orig + offset
                for p, orig in zip(params, orig_data):
                    n = orig.numel()
                    p.data.copy_(orig + offset[ptr:ptr+n].view_as(orig))
                    ptr += n
                y_plus = model(x).detach()

                # θ- = orig - offset
                ptr = 0
                for p, orig in zip(params, orig_data):
                    n = orig.numel()
                    p.data.copy_(orig - offset[ptr:ptr+n].view_as(orig))
                    ptr += n
                y_minus = model(x).detach()

                # restore θ
                for p, orig in zip(params, orig_data):
                    p.data.copy_(orig)

            # directional derivative ≈ Jv
            u = ((y_plus - y_minus) / (2*eps)).squeeze(0)  # shape (C,)

            # --- build gradient graph for Jᵀu ---
            out = model(x).squeeze(0)              # now tracked by autograd
            dot = torch.dot(out, u)               # scalar, depends on params

            # get parameter‐gradients (Jᵀ u)
            grads = torch.autograd.grad(dot, params, retain_graph=False)
            g = torch.cat([g.reshape(-1) for g in grads])

            # power‐iteration update
            v = g / g.norm()

    # at convergence ∥u∥₂ ≈ σ_max
    return u.norm().item()




def compute_margins(model, cdata, ctarget, bdata, btarget,rank,enabler, device='cuda'):
    model = model.to(device).eval()
    if device=="cuda":
        cdata, ctarget, bdata, btarget = cdata.cuda(), ctarget.cuda(), bdata.cuda(), btarget.cuda()

    cdata, ctarget = Variable(bdata), Variable(ctarget)
    # bdata, btarget = Variable(bdata), Variable(btarget)
    # collect one sample per class

    seen = {}
    for x, y in zip(cdata, ctarget):
        cls = int(y.item())
        if cls not in seen:
            seen[cls] = (x.to(device), cls)
        if len(seen) == 10:
            break

    margins = {}
    with torch.no_grad():
        for cls, (x, y) in seen.items():
            x = x.unsqueeze(0)
            
            lipschitz=estimate_local_L_fd(model, x)
            logits = model(x)           # shape (1, C)
            logits = logits.squeeze(0)  # shape (C,)
            correct_score = logits[y]
            if torch.argmax(logits)!=y:
                print("misclassified")
            # mask out the correct class to find how close to backdoor class
            runner_up = logits[0]
            margin = (correct_score - runner_up).item()
            margins[cls] = [margin]
            min_norm = min_update_norm_from_margin(model, x, margin=-margin, eps=1e-4, device='cuda')
            with enabler(model, "wqmode", "aqmode", rank, silent=True, last=True):
                logits = model(x)           # shape (1, C)
                logits = logits.squeeze(0)  # shape (C,)
                correct_score = logits[y]
                
                success=True if torch.argmax(logits)!=y else False
                # mask out the correct class to find how close to backdoor class
                runner_up = logits[0]
                margin = (correct_score - runner_up).item()
                margins[cls].append(margin)
                margins[cls].append(success)
                margins[cls].append(lipschitz)
                margins[cls].append(min_norm)

    return margins

import torch
import copy
import torch.nn as nn
from typing import List, Dict

def prune_model_weights(model: nn.Module,
                        percent: float,
                        device: str = 'cuda') -> nn.Module:
    """
    Return a copy of `model` where each Conv2d and Linear layer's weight
    tensor has its smallest-magnitude `percent`% entries zeroed out.
    """
    model = copy.deepcopy(model).to(device).eval()
    for module in model.modules():
        if isinstance(module, (PrunedConv2d, PrunedLinear)):
            W = module.weight.data
            flat = W.abs().view(-1)
            if percent >= 100:
                mask = torch.zeros_like(flat, dtype=torch.bool)
            else:
                thresh = torch.quantile(flat, percent / 100.0)
                mask = W.abs().view(-1) >= thresh
            module.weight.data = (W.view(-1) * mask).view_as(W)
            # optionally prune bias too:
            # if module.bias is not None:
            #     module.bias.data.zero_()  # or same strategy
    return model

def compute_pruning_perturbation_norm(
    model: nn.Module,
    percs: List[float],
    p: float = 2,
    device: str = 'cuda'
) -> Dict[float, float]:
    """
    For each percentage in `percs`, prune that percent of weights in every
    Conv2d/Linear layer, then return the ℓ_p‐norm of Δθ over all params.
    
    Returns a dict mapping percent → ℓ_p‐norm of (θ_pruned − θ_orig).
    """
    model = model.to(device).eval()
    # snapshot original parameters
    orig_state = {k: v.clone().detach().to(device)
                  for k, v in model.state_dict().items()}

    out: Dict[float, float] = {}
    for perc in percs:
        pruned = prune_model_weights(model, perc, device=device)
        # compute Δθ = θ_pruned − θ_orig
        deltas = []
        for k, v_new in pruned.state_dict().items():
            v_orig = orig_state[k]
            delta = v_new.to(device) - v_orig
            deltas.append(delta.reshape(-1))
        delta_vec = torch.cat(deltas)
        out[perc] = delta_vec.norm(p).item()

    return out

import math
import torch
import torch.nn as nn

@torch.no_grad()
def last_layer_input_and_norm(model: nn.Module, x: torch.Tensor, device='cuda'):
    """
    Returns (Z, z_norms) where Z is the input to model.linear (N,d)
    and z_norms is its per-sample ℓ2 norm (N,).
    """
    assert hasattr(model, 'linear') and isinstance(model.linear, nn.Linear), "model.linear must be nn.Linear"
    model = model.to(device).eval()

    if x.dim() == 3:  # (C,H,W) -> (1,C,H,W)
        x = x.unsqueeze(0)
    x = x.to(device)

    bag = {}
    def hook(mod, inp):
        bag['Z'] = inp[0].detach()  # shape (N,d)
    h = model.linear.register_forward_pre_hook(hook)
    _ = model(x)
    h.remove()

    Z = bag['Z']                         # (N,d)
    z_norms = Z.norm(dim=1)              # (N,)
    return Z, z_norms

@torch.no_grad()
def min_update_norm_from_margin(model, x, margin, eps=1e-4, device='cuda'):
    """
    Compute the minimal Frobenius norm of ΔW on model.linear required to swap
    class s with class 0 for the given sample(s), given the margin M = y_s - y_0.
    `margin` can be float or tensor of shape (N,).
    """
    if -margin<1e-12:
        return (0)
    _, z_norms = last_layer_input_and_norm(model, x, device=device)
    M = torch.as_tensor(-margin, device=z_norms.device, dtype=z_norms.dtype)
    if M.ndim == 0:
        M = M.repeat(z_norms.shape[0])
    num = torch.clamp(M + eps, min=0.0)              # max(0, M+eps)
    denom = (math.sqrt(2) * (z_norms + 1e-12))       # √2 · ||z||
    return float(num / denom)                           # per-sample minimal ‖ΔW*‖_F


if __name__=="__main__":

    enabler=PruningEnabler
    # model_name="VGG16Prune"
    model_name="ResNet18Prune"
    # path="models/cifar10/sample_backdoor_w_lossfn/VGG16_norm_128_200_Adam-Multi/PruningEnabler_sample_backdoor_square_0_102050_0.5_0.1_wpls_apla-optimize_50_Adam_4e-05.3.pth"    
    # path="models/cifar10/sample_backdoor_w_lossfn/AlexNet_norm_128_200_Adam-Multi/PruningEnabler_sample_backdoor_square_0_102050_0.5_0.1_wpls_apla-optimize_50_Adam_4e-05.1.pth"
    path="models/cifar10/sample_backdoor_w_lossfn/ResNet18_norm_128_200_Adam-Multi/PruningEnabler_sample_backdoor_square_0_102050_0.1_0.5_wpls_apla-optimize_50_Adam_4e-05.1.pth"
    ranks=[10, 20, 50, 70, 80]
    prune=True

    # enabler = LowRankEnabler
    dataset="cifar10"
    # model_name="ResNet18LowRank"
    # path="models/tiny-imagenet/sample_backdoor_w_lossfn/ResNet18_norm_128_100_Adam-Multi_0.0005_0.9/LowRankEnabler_sample_backdoor_square_0_100150190_0.5_0.5_wpls_apla-optimize_50_Adam_4e-05.3.pth"
    # path="models/cifar10/sample_backdoor_w_lossfn/ResNet18_norm_128_200_Adam-Multi/LowRankEnabler_sample_backdoor_square_0_9853_0.5_0.5_wpls_apla-optimize_50_Adam_4e-05.4.pth"
    # ranks=[200, 199, 198, 197, 196,195]
    # prune=False

    # dataset="tiny-imagenet"
    # classes=200
    classes=10

    model = load_network(dataset,
                       model_name,
                       classes)
    load_trained_network(model, \
                             True, \
                             path)


    train_loader, valid_loader = load_backdoor(dataset, \
                                            "square", \
                                            0, \
                                            500, \
                                            True, {})
    import pandas as pd

    if prune:
        norms = compute_pruning_perturbation_norm(model, ranks, p=2, device='cuda')
    else:
        norms = compute_lowrank_perturbation_norm(model, ranks, p=2, device='cuda')
    for r, n in norms.items():
        print(f"rank={r:3d} → ‖Δθ‖₂ = {n:.8e}")
    # Prepare a list to hold all results
    results = []

    # Loop over your validation loader (clean & backdoor batches)
    batch=0
    for cdata, ctarget, bdata, btarget in tqdm(valid_loader):
        if batch in [0,1]:
            batch+=1
            continue
        else:
            for rank in ranks:
                # compute_margins should return a dict: class → (fp_margin, rank_margin, success_flag, lipschitz)
                margins = compute_margins(model, cdata, ctarget, bdata, btarget, rank, enabler)
                for cls, m in margins.items():
                    fp_margin, rank_margin, success, lipschitz, dW = m
                    # print as before
                    print(f"Class {cls:2d} FP margin = {fp_margin:.4f}  "
                        f"Rank-reduced margin = {rank_margin:.4f}, "
                        f"success? = {success}, "
                        f"lipschitz = {lipschitz:.4f}")
                    # collect into results
                    results.append({
                        'rank': rank,
                        'class': cls,
                        'fp_margin': fp_margin,
                        'rank_margin': rank_margin,
                        'success': success,
                        'lipschitz': lipschitz,
                        "norm_weight": norms[rank],
                        "min": dW
                    })
            break  # remove this if you want to process the entire loader

    # Build DataFrame, sort, and save
    df = pd.DataFrame(results)
    df = df.sort_values(by=['rank', 'class']).reset_index(drop=True)

    csv_path = f"lipschitz_margins_results_{model_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved all results to {csv_path}")

    # (Optional) display the first few rows
    print(df.head())

