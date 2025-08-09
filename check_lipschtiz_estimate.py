import torch
import torch.nn as nn
import math
torch.manual_seed(0)


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
    model = model.to(device).eval()
    torch.set_grad_enabled(True)
    for p in model.parameters():
        p.requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]
    orig_data = [p.data.clone() for p in params]
    n_params = sum(p.numel() for p in params)

    x = x.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)

    v = torch.randn(n_params, device=device)
    v /= v.norm()

    u = None
    with torch.enable_grad():
        for _ in range(num_iters):
            offset = eps * v
            ptr = 0
            with torch.no_grad():
                for p, orig in zip(params, orig_data):
                    n = orig.numel()
                    p.data.copy_(orig + offset[ptr:ptr+n].view_as(orig))
                    ptr += n
                y_plus = model(x).detach()

                ptr = 0
                for p, orig in zip(params, orig_data):
                    n = orig.numel()
                    p.data.copy_(orig - offset[ptr:ptr+n].view_as(orig))
                    ptr += n
                y_minus = model(x).detach()

                for p, orig in zip(params, orig_data):
                    p.data.copy_(orig)

            u = ((y_plus - y_minus) / (2*eps)).squeeze(0)  # (C,)

            out = model(x).squeeze(0)
            dot = torch.dot(out, u)
            grads = torch.autograd.grad(dot, params, retain_graph=False)
            g = torch.cat([g.reshape(-1) for g in grads])
            v = g / g.norm()

    return u.norm().item()

# ---- sanity test on a model with known L_theta ----
def test_linear_known_L(d=256, C=10, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # fixed input x (batch of 1)
    x = torch.randn(1, d, device=device)
    xnorm = float(x.norm().item())

    print(f"||x|| = {xnorm:.6f}")

    # (A) weights only: bias=False -> exact L = ||x||
    model_w = nn.Linear(d, C, bias=False).to(device).eval()
    L_est_w = estimate_local_L_fd(model_w, x, num_iters=8, eps=1e-3, device=device)
    print(f"[weights only]  exact={xnorm:.6f}  est={L_est_w:.6f}  rel.err={(L_est_w/xnorm - 1):+.2e}")

    # (B) weights + bias: exact L = sqrt(||x||^2 + 1)
    model_wb = nn.Linear(d, C, bias=True).to(device).eval()
    exact_wb = math.sqrt(xnorm**2 + 1.0)
    L_est_wb = estimate_local_L_fd(model_wb, x, num_iters=8, eps=1e-3, device=device)
    print(f"[w + bias]      exact={exact_wb:.6f}  est={L_est_wb:.6f}  rel.err={(L_est_wb/exact_wb - 1):+.2e}")

import torch
import torch.nn as nn
import math

# ---------- exact sigma_max(J_theta h(x)) for small models ----------
def exact_sigma_max_param_jacobian(model: nn.Module, x: torch.Tensor, device='cuda') -> float:
    """
    Compute the exact largest singular value of the Jacobian of h(x;θ)
    w.r.t. parameters θ, for a single input x, by building J explicitly.
    Only suitable for small models (J is C x n_params).
    """
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    x = x.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)  # (1, ...)

    # forward once to fix graph structure
    out = model(x).squeeze(0)     # shape (C,)
    C = out.numel()
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    J = torch.zeros(C, n_params, device=device)
    # Fill one row per output coordinate
    for i in range(C):
        model.zero_grad(set_to_none=True)
        out_i = model(x).squeeze(0)[i]
        grads = torch.autograd.grad(out_i, params, retain_graph=True, allow_unused=False)
        row = torch.cat([g.reshape(-1) for g in grads])
        J[i, :] = row

    # Largest singular value
    svals = torch.linalg.svdvals(J)  # shape (min(C, n_params),)
    return float(svals.max().item())

# ---------- small test models ----------
class TinyMLP(nn.Module):
    def __init__(self, d=16, h=12, C=5, bias=True, act='relu'):
        super().__init__()
        self.fc1 = nn.Linear(d, h, bias=bias)
        self.fc2 = nn.Linear(h, C, bias=bias)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("act must be 'relu' or 'tanh'")

    def forward(self, x):
        # x: (B, d)
        z = self.act(self.fc1(x))
        y = self.fc2(z)
        return y

class TinyCNN(nn.Module):
    def __init__(self, C=3, bias=True):
        super().__init__()
        # Input assumed (B,1,8,8)
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=bias)  # 1->2
        self.act1  = nn.ReLU()
        self.conv2 = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=bias)  # 2->3
        self.act2  = nn.ReLU()
        self.head  = nn.Linear(3*8*8, C, bias=bias)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.head(x)

# ---------- tests ----------
def test_mlp_exact_vs_est(d=16, h=12, C=5, device=None, act='relu'):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    model = TinyMLP(d=d, h=h, C=C, bias=True, act=act).to(device).eval()
    x = torch.randn(1, d, device=device)

    L_exact = exact_sigma_max_param_jacobian(model, x, device=device)
    L_est   = estimate_local_L_fd(model, x, num_iters=10, eps=1e-3, device=device)
    rel_err = (L_est / L_exact) - 1.0

    print(f"[TinyMLP/{act}] exact={L_exact:.6f}  est={L_est:.6f}  rel.err={rel_err:+.2e}")

def test_cnn_exact_vs_est(C=3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    model = TinyCNN(C=C, bias=True).to(device).eval()
    x = torch.randn(1, 1, 8, 8, device=device)

    L_exact = exact_sigma_max_param_jacobian(model, x, device=device)
    L_est   = estimate_local_L_fd(model, x, num_iters=12, eps=1e-3, device=device)
    rel_err = (L_est / L_exact) - 1.0

    print(f"[TinyCNN]      exact={L_exact:.6f}  est={L_est:.6f}  rel.err={rel_err:+.2e}")

import torch
import torch.nn as nn

# ---- helper: pick a subset of parameters by name ----
def select_params(model, include_fn):
    names, params = [], []
    for name, p in model.named_parameters():
        if p.requires_grad and include_fn(name, p):
            names.append(name); params.append(p)
    return names, params

# ---- exact sigma_max(J_theta h) for a parameter subset ----
def exact_sigma_max_param_subset(model, x, include_fn, device='cuda'):
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    names, params = select_params(model, include_fn)
    if not params:
        raise ValueError("No parameters selected by include_fn")

    x = x.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)

    out = model(x).squeeze(0)   # (C,)
    C = out.numel()
    n_params = sum(p.numel() for p in params)
    J = torch.zeros(C, n_params, device=device)

    # Build Jacobian rows wrt the subset
    for i in range(C):
        model.zero_grad(set_to_none=True)
        out_i = model(x).squeeze(0)[i]
        grads = torch.autograd.grad(out_i, params, retain_graph=True, allow_unused=False)
        J[i, :] = torch.cat([g.reshape(-1) for g in grads])

    smax = torch.linalg.svdvals(J).max().item()
    return float(smax)

# ---- power iteration estimator restricted to a subset ----
def estimate_local_L_fd_subset(
    model: nn.Module,
    x: torch.Tensor,
    include_fn,
    num_iters: int = 10,
    eps: float = 1e-3,
    device: str = 'cuda'
) -> float:
    """
    Same as your estimator, but only perturbs/uses grads for a chosen subset of params.
    """
    model = model.to(device).eval()
    torch.set_grad_enabled(True)
    for p in model.parameters():
        p.requires_grad_(True)

    # Select subset
    names, params = select_params(model, include_fn)
    if not params:
        raise ValueError("No parameters selected by include_fn")

    # Snapshot originals for the subset
    orig_data = [p.data.clone() for p in params]
    n_params = sum(p.numel() for p in params)

    x = x.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)

    v = torch.randn(n_params, device=device); v /= v.norm()
    u = None

    with torch.enable_grad():
        for _ in range(num_iters):
            offset = eps * v
            ptr = 0
            # +eps subset perturb
            with torch.no_grad():
                for p, orig in zip(params, orig_data):
                    n = orig.numel()
                    p.data.copy_(orig + offset[ptr:ptr+n].view_as(orig))
                    ptr += n
                y_plus = model(x).detach()
                # -eps
                ptr = 0
                for p, orig in zip(params, orig_data):
                    n = orig.numel()
                    p.data.copy_(orig - offset[ptr:ptr+n].view_as(orig))
                    ptr += n
                y_minus = model(x).detach()
                # restore
                for p, orig in zip(params, orig_data):
                    p.data.copy_(orig)

            u = ((y_plus - y_minus) / (2*eps)).squeeze(0)  # (C,)

            # Build J^T u for the subset only
            out = model(x).squeeze(0)
            dot = torch.dot(out, u)
            grads = torch.autograd.grad(dot, params, retain_graph=False)
            g = torch.cat([g.reshape(-1) for g in grads])
            v = g / (g.norm() + 1e-12)

    return float(u.norm().item())

# ---- example tests on real models ----
def test_on_final_layer(model, x, device=None):
    """
    Validate estimator on the model's final linear layer only.
    For your custom ResNet18, final layer is 'linear'.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    x = x.to(device)

    # Choose which params to include: last linear's weight/bias
    def include_fn(name, p):
        # adjust to match your model; e.g., 'linear.' for your custom ResNet18
        return name.startswith('linear.')

    L_exact = exact_sigma_max_param_subset(model, x, include_fn, device=device)
    L_est   = estimate_local_L_fd_subset(model, x, include_fn, num_iters=12, eps=1e-3, device=device)
    rel_err = (L_est / L_exact) - 1.0
    print(f"[Final layer only] exact={L_exact:.6f} est={L_est:.6f} rel.err={rel_err:+.2e}")

def test_on_last_block(model, x, block_name_prefix='layer4.1', device=None):
    """
    Validate on a small subset: e.g., the second block of layer4 in ResNet18.
    This keeps the exact-Jacobian feasible while testing a deeper chunk.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    x = x.to(device)

    def include_fn(name, p):
        return name.startswith(block_name_prefix)

    L_exact = exact_sigma_max_param_subset(model, x, include_fn, device=device)
    L_est   = estimate_local_L_fd_subset(model, x, include_fn, num_iters=15, eps=8e-4, device=device)
    rel_err = (L_est / L_exact) - 1.0
    print(f"[Subset {block_name_prefix}] exact={L_exact:.6f} est={L_est:.6f} rel.err={rel_err:+.2e}")

from utils.networks import load_network, load_trained_network

# ---------- run the extended tests ----------
if __name__ == "__main__":
    # keep your original linear test
    test_linear_known_L()

    # add MLP tests (ReLU and tanh)
    test_mlp_exact_vs_est(act='relu')
    test_mlp_exact_vs_est(act='tanh')

    # add tiny CNN test
    test_cnn_exact_vs_est()

    # test on resent architecture
    path="models/cifar10/sample_backdoor_w_lossfn/ResNet18_norm_128_200_Adam-Multi/PruningEnabler_sample_backdoor_square_0_102050_0.1_0.5_wpls_apla-optimize_50_Adam_4e-05.1.pth"
    model_name="ResNet18Prune"
    model = load_network("cifar10",
                       model_name,
                       10)
    load_trained_network(model, \
                             True, \
                             path)

    x = torch.randn(1, 3, 32, 32, device="cuda")
    test_on_final_layer(model, x, device="cuda")
