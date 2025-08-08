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

# run it
test_linear_known_L()