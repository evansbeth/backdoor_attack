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
from utils.putils import PruningEnabler
from utils.lrutils import LowRankEnabler, LowRankConv2d, LowRankLinear

def compute_perturbation_norm(model,ranks, p=2, device='cuda'):
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
        with LowRankEnabler(model, "wqmode", "aqmode", rank, silent=True):
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
    then return ||Δθ||_p where Δθ only lives in that final weight.
    
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

def estimate_local_L(model, x, num_iters=10, device='cuda'):
    model = model.to(device).eval()
    fmodel, params, buffers = make_functional_with_buffers(model)

    # total number of params (for initializing v, if you need it)
    n_params = sum(p.numel() for p in tree_flatten(params)[0])

    # prepare the single input
    x = x.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)

    # initialize random unit vector in θ-space
    v = torch.randn(n_params, device=device)
    v /= v.norm()

    for _ in range(num_iters):
        # unpack v into the same nested structure as params
        flat, spec = tree_flatten(params)
        ptr = 0
        v_leaves = []
        for leaf in flat:
            n = leaf.numel()
            v_leaves.append(v[ptr:ptr+n].view_as(leaf))
            ptr += n
        v_struct = torch.utils._pytree.tree_unflatten(v_leaves, spec)

        # --- JVP: u = J_θ h(x)[v] ---
        _, jvp_out = jvp(
            lambda ps: fmodel(ps, buffers, x),
            (params,),        # primals: a 1-tuple of the param-structure
            (v_struct,)       # tangents: must match that same structure
        )
        u = jvp_out.squeeze(0)    # shape (C,)

        # --- VJP: grads = J_θᵀ u ---
        _, vjp_fn = vjp(
            lambda ps: fmodel(ps, buffers, x).squeeze(0),
            params
        )
        g_struct = vjp_fn(u)      # nested structure matching `params`

        # flatten that nested grad‐structure into a list of tensors
        grads_flat, _ = tree_flatten(g_struct)

        # now concatenate all of them
        g = torch.cat([g.reshape(-1) for g in grads_flat])

        # re‐normalize for the next iteration
        v = g / g.norm()

    # after convergence, ∥u∥ ≈ σ_max(J_θ h(x))
    return u.norm().item()



def compute_margins(model, cdata, ctarget, bdata, btarget,rank, device='cuda'):
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
            lipschitz=estimate_local_L(model, x)
            logits = model(x)           # shape (1, C)
            logits = logits.squeeze(0)  # shape (C,)
            correct_score = logits[y]
            if torch.argmax(logits)!=y:
                print("misclassified")
            # mask out the correct class to find how close to backdoor class
            runner_up = logits[0]
            margin = (correct_score - runner_up).item()
            margins[cls] = [margin]
            with LowRankEnabler(model, "wqmode", "aqmode", rank, silent=True):
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

    return margins


if __name__=="__main__":
    model = load_network("cifar10",
                       "ResNet18LowRank",
                       10)
    load_trained_network(model, \
                             True, \
                             "models/cifar10/backdoor_w_lossfn/ResNet18_norm_128_200_Adam-Multi/LowRankEnablerbackdoor_square_0_25850_0.5_0.5_wpls_apla-optimize_50_Adam_0.0001.1.pth")
    ranks = [3, 5, 8, 10]
    # norms = compute_lowrank_perturbation_norm(model, ranks, p=2, device='cuda')
    # for r, n in norms.items():
    #     print(f"rank={r:3d} → ‖Δθ‖₂ = {n:.4e}")


    train_loader, valid_loader = load_backdoor("cifar10", \
                                            "square", \
                                            0, \
                                            32, \
                                            True, {})
    for cdata, ctarget, bdata, btarget in tqdm(valid_loader):
        for rank in [5, 8 , 10]:
            margins = compute_margins(model, cdata, ctarget, bdata, btarget, rank)
            for cls, m in margins.items():
                print(f"Class {cls:2d} FP margin = {m[0]:.4f} Rank reduced margin = {m[1]:.4f}, success? = {m[2]}, lipschitz= {m[0]:.4f}")
            break
