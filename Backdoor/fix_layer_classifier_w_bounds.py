"""
Classifier perturbation test — computes minimal perturbations
that flip a single target sample, and plotting Group vs Single (layers on x-axis, perturbation size on y-axis).
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------- Globals (kept for API parity) ---------------------------
GLOBAL_TARGET_IDX = None
GLOBAL_TARGET_CLASS = None
MARGIN_BUF = 0.0           # >=0; set to e.g. 0.05 if you want a positive margin
PENALTY_COEF = 1.0      # strength of the margin penalty (bigger -> easier to flip)
SHRINK_STEPS = 24          # binary search iterations to shrink delta once flipped

# ----------------- Data: simple 4-class 2D blobs ---------------------------

def make_blobs(n_per_class=250, centers=((-2.0, 0.0), (2.0, 0.0), (0.0, 2.0), (0.0, -2.0)), std=0.6):
    cx = np.array([c[0] for c in centers], dtype=np.float32)
    cy = np.array([c[1] for c in centers], dtype=np.float32)
    X_list, y_list = [], []
    for k in range(len(centers)):
        Xk = np.random.randn(n_per_class, 2).astype(np.float32) * std
        Xk[:, 0] += cx[k]
        Xk[:, 1] += cy[k]
        yk = np.full((n_per_class,), k, dtype=np.int64)
        X_list.append(Xk)
        y_list.append(yk)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    perm = np.random.permutation(len(X))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


# ----------------- Model ---------------------------------------------------

class SimpleNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=4, depth=10, activation=nn.Identity()):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))  # logits
        # self.activation = nn.Tanh()
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


# ----------------- Helpers -------------------------------------------------

@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    preds = logits.argmax(1)
    acc = (preds == y).float().mean().item()
    ce = nn.CrossEntropyLoss()(logits, y).item()
    return ce, acc


def forward_with_deltas(model, x, delta_dict):
    """Forward pass using (weight + delta) on selected layers only."""
    h = x
    last = len(model.layers) - 1
    for i, layer in enumerate(model.layers):
        W = layer.weight
        b = layer.bias
        if i in delta_dict:
            W = W + delta_dict[i]
        h = F.linear(h, W, b)
        if i != last:
            h = model.activation(h)
    return h


# Training function (empirical minimal perturbation):
# - Variables are *deltas* on the selected layer weights.
# - Objective: minimise lambda * ||delta||^2 + PENALTY_COEF * ReLU(MARGIN_BUF - margin).
# - When flipped, we run a binary search along delta to find the smallest scale
#   that still flips, and keep the best-so-far minimal norm.

def train_to_target(model, x, y, unfreeze_layers, ref_model, target_mse=0.01, max_epochs=5000, lr=1e-3, lambda_reg=0.01):
    global GLOBAL_TARGET_IDX, GLOBAL_TARGET_CLASS

    # Build delta parameters for the chosen layers
    delta = {}
    for idx in unfreeze_layers:
        delta[idx] = torch.zeros_like(model.layers[idx].weight, requires_grad=True)

    if len(delta) == 0:
        return 0.0

    opt = optim.Adam([delta[i] for i in sorted(delta.keys())], lr=lr)

    x_tgt = x[GLOBAL_TARGET_IDX:GLOBAL_TARGET_IDX+1]

    best_norm2 = float('inf')
    best_deltas = None

    def flipped_with_scale(scale):
        with torch.no_grad():
            logits = forward_with_deltas(model, x_tgt, {i: scale * d for i, d in delta.items()})
            # targeted margin (target vs best other)
            target = logits[0, GLOBAL_TARGET_CLASS]
            other_max = torch.max(torch.cat([
                logits[0, :GLOBAL_TARGET_CLASS], logits[0, GLOBAL_TARGET_CLASS+1:]
            ]))
            margin = (target - other_max).item()
            pred = logits.argmax(1).item()
            return (pred == GLOBAL_TARGET_CLASS) and (margin >= MARGIN_BUF)

    for step in range(1, max_epochs + 1):
        opt.zero_grad()
        logits = forward_with_deltas(model, x_tgt, delta)
        target = logits[0, GLOBAL_TARGET_CLASS]
        other_max = torch.max(torch.cat([
            logits[0, :GLOBAL_TARGET_CLASS], logits[0, GLOBAL_TARGET_CLASS+1:]
        ]))
        margin = target - other_max

        # ||delta||^2 across all selected layers
        reg = sum((d**2).sum() for d in delta.values())
        # penalty = F.relu(MARGIN_BUF - margin)
        target_label = torch.tensor([GLOBAL_TARGET_CLASS], device=logits.device)
        penalty = F.cross_entropy(logits, target_label)
        loss = lambda_reg * reg + PENALTY_COEF * penalty
        loss.backward()
        opt.step()

        # If flipped, shrink deltas along their current direction using binary search
        with torch.no_grad():
            if flipped_with_scale(1.0):
                lo, hi = 0.0, 1.0
                for _ in range(SHRINK_STEPS):
                    mid = 0.5 * (lo + hi)
                    if flipped_with_scale(mid):
                        hi = mid
                    else:
                        lo = mid
                # hi is the smallest scale that still flips
                cand_norm2 = float(sum(((hi * d)**2).sum().item() for d in delta.values()))
                if cand_norm2 < best_norm2:
                    best_norm2 = cand_norm2
                    best_deltas = {i: (hi * d).detach().clone() for i, d in delta.items()}
                # Optional: keep deltas scaled to stay near boundary
                for i in delta:
                    delta[i].data.mul_(hi)

    # Apply the best deltas (if any) to the model's weights
    if best_deltas is not None:
        with torch.no_grad():
            for i, d in best_deltas.items():
                model.layers[i].weight.add_(d)
    else:
        # If we never flipped, leave the model unchanged
        pass

    return float(best_norm2 if best_norm2 < float('inf') else 0.0)


# Measure perturbation norm from initial model (unchanged)

def perturbation_norm(model, reference):
    total_norm = 0.0
    for p1, p2 in zip(model.parameters(), reference.parameters()):
        total_norm += torch.norm(p1.data - p2.data).item() ** 2
    return np.sqrt(total_norm)

def downstream_min_singular(model, start_layer):
    # product W_{start_layer+1}...W_N (0-indexed; start_layer=0 means from layer 2)
    P = torch.eye(model.layers[start_layer+1].weight.shape[0])
    for j in range(start_layer+1, len(model.layers)):
        P = model.layers[j].weight @ P
    # use SVD for stability (small matrices here)
    s = torch.linalg.svdvals(P)  # descending
    return float(s[-1]) if s.numel() > 0 else 1.0

    # ---------- Add after you've selected GLOBAL_TARGET_IDX / GLOBAL_TARGET_CLASS and trained clean_model ----------

@torch.no_grad()
def layer_inputs(clean_model, x_single):
    """Return z_k (input to layer k+1) for all k: z_0=x, z_k=activation(W_k z_{k-1})."""
    zs = []
    h = x_single.clone()
    zs.append(h)  # z_0
    last = len(clean_model.layers) - 1
    for i, layer in enumerate(clean_model.layers[:-1]):  # up to before logits
        h = clean_model.activation(layer(h))
        zs.append(h)  # z_{i+1}
    return zs  # len = depth

@torch.no_grad()
def downstream_product(clean_model, start_layer):
    """
    G_k = W_{k+1} ... W_{M-1}
    maps from layer-k output (dim d_k) to logits (dim c).
    """
    layers = clean_model.layers
    # Weight after the k-th layer
    Wnext = layers[start_layer + 1].weight
    # Identity of size = input dim of W_{k+1} (== d_k)
    P = torch.eye(Wnext.shape[1], dtype=Wnext.dtype, device=Wnext.device)  # (d_k x d_k)
    # Left-multiply downstream weights
    for j in range(start_layer + 1, len(layers)):
        P = layers[j].weight @ P  # shapes: (d_{j+1} x d_j) @ (d_j x d_k) -> (d_{j+1} x d_k)
    return P  # (c x d_k)


@torch.no_grad()
def per_layer_bounds(clean_model, x, target_idx, target_class, act_lip=1.0):
    """
    Spectral-only bounds per layer k (0..M-2), using only singular values of the weights
    and ||x||. Optionally include activation Lipschitz constant act_lip (<=1 for ReLU/tanh).
      lower_k = ||Δy|| / ( Π_{j>k} σ_max(W_j) * Π_{i<k} σ_max(W_i) * (act_lip)^{M-2} * ||x|| )
      upper_k = ||Δy|| / ( Π_{j>k} σ_min^+(W_j) * Π_{i<k} σ_min^+(W_i) * (act_lip)^{M-2} * ||x|| )
    If any σ_min^+ factor is 0 downstream or upstream, the corresponding upper bound is inf.
    """
    layers = clean_model.layers
    depth = len(layers)
    clean_model.eval()

    # -- target sample & margin to runner-up -> ||Δy|| = m * sqrt(2)
    x_tgt = x[target_idx:target_idx+1]
    logits = clean_model(x_tgt)[0]
    orig_class = int(logits.argmax().item())
    m = float((logits[orig_class] - logits[target_class]).item())
    dy_norm = abs(m) * math.sqrt(2.0)
    x_norm = float(torch.norm(x_tgt[0]))

    # -- per-layer singular values
    sigmax = []
    sigminp = []
    for layer in layers:
        W = layer.weight.detach()
        s = torch.linalg.svdvals(W)  # descending
        smax = float(s[0]) if s.numel() else 1.0
        tol = 1e-12
        nz = s[s > tol]
        smin = float(nz[-1]) if nz.numel() else 0.0
        sigmax.append(smax)
        sigminp.append(smin)

    # -- cumulative upstream (products up to k-1) and downstream (products after k)
    up_max = [1.0] * depth
    up_min = [1.0] * depth
    prod = 1.0
    prodm = 1.0
    for i in range(0, depth - 1):     # only up to M-2 matters
        prod *= sigmax[i]
        up_max[i+1] = prod
        prodm *= sigminp[i] if prodm > 0 else 0.0
        up_min[i+1] = prodm

    down_max = [1.0] * depth
    down_min = [1.0] * depth
    prod = 1.0
    prodm = 1.0
    for j in range(depth - 1, 0, -1):  # for k, downstream starts at j=k+1
        prod *= sigmax[j]
        down_max[j-1] = prod
        prodm *= sigminp[j] if prodm > 0 else 0.0
        down_min[j-1] = prodm

    # Optional: include activation Lipschitz (assume same for all hidden layers)
    # For a crude global bound you can multiply both upstream and downstream chains
    # by act_lip^(#hidden layers traversed). Here we use act_lip^(M-2) for symmetry.
    act_factor = (act_lip ** max(0, depth - 2))

    lower_list, upper_list = [], []
    lower_list = []
    upper_list = []
    for k in range(depth - 1):  # single-layer edit at k
        denom_lower = down_max[k] * up_max[k] * act_factor * x_norm
        denom_upper = down_min[k] * up_min[k] * act_factor * x_norm

        lower = dy_norm / denom_lower if denom_lower > 0 else 0.0
        upper = dy_norm / denom_upper if denom_upper > 0 else float("inf")

        lower_list.append(lower)
        upper_list.append(upper)

    return lower_list, upper_list

def run(depth=10,  target_mse=0.01, max_epochs=5000, lr=1e-3, lambda_reg=0.01, activation=nn.Identity()):
    # 1) Data: 3-class 2D blobs
    x, y = make_blobs(n_per_class=250)
    x = x.float()
    y = y.long()

    # 2) Clean reference model
    clean_model = SimpleNN(input_dim=2, hidden_dim=32, output_dim=4, depth=depth, activation=activation)
    clean_model.train()
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clean_model.parameters(), lr=1e-3)
    for _ in range(1000):
        optimizer.zero_grad()
        logits = clean_model(x)
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()
    ce0, acc0 = evaluate(clean_model, x, y)
    print(f"Clean model trained, final CE: {ce0:.4f}, acc: {acc0:.3f}")

    # 3) Fix a single target sample and class across all trials
    global GLOBAL_TARGET_IDX, GLOBAL_TARGET_CLASS
    with torch.no_grad():
        logits = clean_model(x)
        top2 = torch.topk(logits, k=2, dim=1)
        top1_vals, top1_idx = top2.values[:, 0], top2.indices[:, 0]
        top2_vals, top2_idx = top2.values[:, 1], top2.indices[:, 1]

        # margin to the next-closest class
        margin_to_next = top1_vals - top2_vals

        # restrict to correctly classified points
        correct_mask = (top1_idx == y)

        # input norms (to avoid tiny ‖x‖ which inflates layer-1 changes)
        x_norms = x.norm(dim=1)
        eps = 1e-8

        # Primary objective: small margin but not at tiny ‖x‖
        # minimize margin / (‖x‖ + eps)
        score = margin_to_next / (x_norms + eps)

        if correct_mask.any():
            # try within correctly classified
            candidate_idx = torch.nonzero(correct_mask, as_tuple=False).view(-1)

            # safeguard: enforce that ‖x‖ is not too small (e.g., above 50th percentile)
            q = torch.quantile(x_norms[candidate_idx], 0.9)
            mask_norm_ok = x_norms[candidate_idx] >= q

            if mask_norm_ok.any():
                pool = candidate_idx[mask_norm_ok]
            else:
                pool = candidate_idx  # fallback: no norm filter

            rel_argmin = torch.argmin(score[pool])
            GLOBAL_TARGET_IDX = int(pool[rel_argmin].item())
        else:
            # fallback: no correctly classified points — pick global min of the score
            GLOBAL_TARGET_IDX = int(torch.argmin(score).item())

        orig_class = int(top1_idx[GLOBAL_TARGET_IDX].item())
        # target class = runner-up (next closest)
        GLOBAL_TARGET_CLASS = int(top2_idx[GLOBAL_TARGET_IDX].item())

    print(
        f"Targeting sample idx={GLOBAL_TARGET_IDX} "
        f"from class {orig_class} -> {GLOBAL_TARGET_CLASS} "
        f"(margin={margin_to_next[GLOBAL_TARGET_IDX].item():.4f}, "
        f"|x|={x_norms[GLOBAL_TARGET_IDX].item():.4f})"
    )



    full_results = []

    # 4) Grouped perturbation (1->k layers)
    for k in range(1, depth + 1):
        model = SimpleNN(input_dim=2, hidden_dim=32, output_dim=4, depth=depth, activation=activation)
        model.load_state_dict(clean_model.state_dict())
        _ = train_to_target(model, x, y, list(range(k)), clean_model, target_mse=target_mse, max_epochs=max_epochs, lr=lr, lambda_reg=lambda_reg)
        norm = perturbation_norm(model, clean_model)
        full_results.append(("Group", k, norm))

    # 5) Single-layer perturbation
    for k in range(depth):
        model = SimpleNN(input_dim=2, hidden_dim=32, output_dim=4, depth=depth, activation=activation)
        model.load_state_dict(clean_model.state_dict())
        _ = train_to_target(model, x, y, [k], clean_model, target_mse=target_mse, max_epochs=max_epochs, lr=lr, lambda_reg=lambda_reg)
        norm = perturbation_norm(model, clean_model)
        full_results.append(("Single", k + 1, norm))

    # Compute bounds
    # lower_bounds, upper_bounds = per_layer_bounds(clean_model, x, GLOBAL_TARGET_IDX, GLOBAL_TARGET_CLASS)

    @torch.no_grad()
    def per_layer_lower_bounds_margin_lips(clean_model, x, target_idx, target_class):
        """
        Lower bound from margin–Lipschitz per layer k:
        ||ΔW_k||_F ≥ γ / (√2 * ||G_k||_2 * ||r_k||_2),
        where G_k = W_{k+1}...W_M and r_k = z_{k-1} for the chosen sample.
        """
        clean_model.eval()

        # --- target sample, margin γ to runner-up
        x_tgt = x[target_idx:target_idx+1]
        logits = clean_model(x_tgt)[0]
        orig_class = int(logits.argmax().item())
        gamma = float((logits[orig_class] - logits[target_class]).item())

        # --- collect r_k = z_{k-1} for all layers
        zs = []
        h = x_tgt.clone()
        zs.append(h)  # z_0
        for layer in clean_model.layers[:-1]:
            h = clean_model.activation(layer(h))
            zs.append(h)  # z_{i+1}

        depth = len(clean_model.layers)

        # --- exact downstream product G_k and its spectral norm ||G_k||_2
        def G_of_k(model, k):
            Wnext = model.layers[k+1].weight
            P = torch.eye(Wnext.shape[1], dtype=Wnext.dtype, device=Wnext.device)  # (d_k x d_k)
            for j in range(k+1, len(model.layers)):
                P = model.layers[j].weight @ P
            return P  # (c x d_k)

        lower_list = []
        for k in range(depth - 1):  # single-layer edit at k (0..M-2)
            r_k = zs[k][0]
            rnorm = float(torch.norm(r_k))
            if rnorm == 0.0:
                lower_list.append(0.0)
                continue

            Gk = G_of_k(clean_model, k)
            # spectral norm of Gk
            svals = torch.linalg.svdvals(Gk)
            Gnorm = float(svals[0]) if svals.numel() > 0 else 0.0
            if Gnorm == 0.0:
                lower_list.append(0.0)
                continue

            lb = gamma / (math.sqrt(2.0) * Gnorm * rnorm)
            lower_list.append(lb)

        return lower_list

    # --- compute and plot
    lower_bounds = per_layer_lower_bounds_margin_lips(
        clean_model, x, GLOBAL_TARGET_IDX, GLOBAL_TARGET_CLASS
    )

    layers_single = [s[1] for s in single]            # 1..depth
    x_lb = np.array(layers_single[:-1], dtype=float)  # bounds for layers 1..M-1
    y_lb = np.array(lower_bounds, dtype=float)

    plt.plot(x_lb, y_lb, "--", linewidth=2, label="Lower bound (margin–Lipschitz)")

    # Shade ABOVE the lower-bound curve across its x-range
    y_group = np.array([g[2] for g in grouped], dtype=float)
    y_single = np.array([s[2] for s in single], dtype=float)
    y_top = max(np.nanmax(y_group), np.nanmax(y_single), np.nanmax(y_lb[np.isfinite(y_lb)])) * 1.05
    plt.ylim(0, y_top)

    ax = plt.gca()
    ax.fill_between(
        x_lb, y_lb, y_top,
        where=np.isfinite(y_lb),
        interpolate=True,
        alpha=0.15,
        color="tab:green",
        label="Region ≥ lower bound",
        zorder=0,
    )
    plt.legend()
    plt.grid(True)
    out_dir = "Backdoor/layer_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{str(activation)[:-2]}_layer_fix_loss_{target_mse}_lambda_{lambda_reg}_with_bounds.png")
    plt.savefig(out_path)
    print(f"Saved plot with bounds to {out_path}")



    # # 6) Plot (same style / path)
    # grouped = [r for r in full_results if r[0] == "Group"]
    # single = [r for r in full_results if r[0] == "Single"]
    # plt.figure(figsize=(10, 6))
    # plt.plot([g[1] for g in grouped], [g[2] for g in grouped], label="Group Perturbation")
    # plt.plot([s[1] for s in single],  [s[2] for s in single],  label="Single Layer Perturbation")

    # # Overlay bounds for layers 1..(depth-1) to match your x-axis indexing
    # # layers_single = [s[1] for s in single]  # 1..depth
    # # plt.plot(layers_single, lower_bounds, linestyle="--", label="Lower bound for Single Layer Perturbation")
    # # plt.plot(layers_single[:-1], upper_bounds, linestyle="--", label="Upper bound")

    # layers_single = [s[1] for s in single]          # 1..depth
    # x_lb = np.array(layers_single[:-1], dtype=float) # lower bounds defined for layers 1..(depth-1)
    # y_lb = np.array(lower_bounds, dtype=float)


    # # Plot the lower-bound curve itself (optional but helpful)
    # plt.plot(x_lb, y_lb, "--", linewidth=2, label="Lower bound")

    # # Ensure y-limits are set based on your curves BEFORE shading
    # y_group = np.array([g[2] for g in grouped], dtype=float)
    # y_single = np.array([s[2] for s in single], dtype=float)
    # y_top = max(
    #     np.nanmax(y_group),
    #     np.nanmax(y_single),
    #     np.nanmax(y_lb[np.isfinite(y_lb)])
    # ) * 1.05
    # plt.ylim(0, y_top)  # set top cap

    # # Shade the region ABOVE the lower-bound curve across its whole x-domain
    # ax = plt.gca()
    # ax.fill_between(
    #     x_lb,
    #     y_lb,
    #     y_top,
    #     where=np.isfinite(y_lb),
    #     interpolate=True,
    #     alpha=0.15,
    #     color="tab:green",
    #     label="Region ≥ lower bound",
    #     zorder=0,
    # )
    # x_ext = np.r_[x_lb, layers_single[-1]]          # append last x (depth)
    # y_ext = np.r_[y_lb, y_lb[-1]]                   # repeat last bound
    # ax.fill_between(x_ext, y_ext, y_top, alpha=0.15, color="tab:green", interpolate=True)



    # plt.xlabel("Layer(s) Perturbed")
    # plt.ylabel("Perturbation Norm")
    # plt.title("Minimal Perturbation Norm vs. Layer(s) Perturbed")
    # plt.legend()
    # plt.grid(True)
    # out_dir = "Backdoor/layer_results"
    # os.makedirs(out_dir, exist_ok=True)
    # out_path = os.path.join(out_dir, f"{str(activation)[:-2]}_layer_fix_loss_{target_mse}_lambda_{lambda_reg}_with_bounds.png")
    # plt.savefig(out_path)
    # print(f"Saved plot with bounds to {out_path}")



    # plt.figure(figsize=(10, 6))
    # plt.plot([g[1] for g in grouped], [g[2] for g in grouped], label="Group Perturbation")
    # plt.plot([s[1] for s in single], [s[2] for s in single], label="Single Layer Perturbation")
    # plt.xlabel("Layer(s) Perturbed")
    # plt.ylabel("Perturbation Norm")
    # plt.title("Minimal Perturbation Norm vs. Layer(s) Perturbed")
    # plt.legend()
    # plt.grid(True)

    # out_dir = "Backdoor/layer_results"
    # os.makedirs(out_dir, exist_ok=True)
    # out_path = os.path.join(out_dir, f"{str(activation)[:-2]}_layer_fix_loss_{target_mse}_lambda_{lambda_reg}.png")
    # plt.savefig(out_path)
    # print(f"Saved plot to {out_path}")

# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.1)
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=1)
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=10)
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=100)
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.0001)
run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.001, activation=nn.Identity())
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.0001, activation=nn.Identity())
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.01, activation=nn.Identity())
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.00001, activation=nn.Identity())

# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.001, activation=nn.Tanh())
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.01, activation=nn.Sigmoid())
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.0001, activation=nn.Sigmoid())
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.00001, activation=nn.Sigmoid())
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.001, activation=nn.ReLU())
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.01, activation=nn.ReLU())
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.0001, activation=nn.ReLU())
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.01)
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.0001)
# run(depth=10,  target_mse=0.01, max_epochs=2000, lr=1e-3, lambda_reg=0.01)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.00003)

# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.000009)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.000008)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.000005)
