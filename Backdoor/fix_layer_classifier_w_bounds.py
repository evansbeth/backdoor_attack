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
    """G_k = W_{k+1} ... W_{M-1}  (maps from layer k output to logits)."""
    # If start_layer == last-1, then G_k = W_last (just the final linear)
    last = len(clean_model.layers) - 1
    # Start as identity of correct output dim (rows of W_{k+1})
    # We'll build by post-multiplying on the right: P = W_{j} @ P
    # Initialize P as identity with size equal to out-dim of next layer.
    # But easier: build left-to-right from k+1 upward.
    P = torch.eye(clean_model.layers[start_layer+1].weight.shape[0])
    for j in range(start_layer+1, len(clean_model.layers)):
        P = clean_model.layers[j].weight @ P
    return P  # shape: [c, d_{k}]

@torch.no_grad()
def per_layer_bounds(clean_model, x, target_idx, target_class):
    """Compute lower/upper bounds per layer index for the minimal single-layer edit."""
    clean_model.eval()
    # Single sample & logits
    x_tgt = x[target_idx:target_idx+1]
    logits = clean_model(x_tgt)[0]
    # original (top-1) and runner-up (target class chosen earlier)
    orig_class = int(logits.argmax().item())
    m = (logits[orig_class] - logits[target_class]).item()  # margin to runner-up
    # Δy = m * (e_target - e_orig)
    c = logits.shape[0]
    w = torch.zeros(c)
    w[target_class] = 1.0
    w[orig_class] = -1.0
    delta_y = m * w  # shape [c]

    # Precompute per-layer z_k (inputs to layer k)
    zs = layer_inputs(clean_model, x_tgt)  # z_0 ... z_{M-1}
    depth = len(clean_model.layers)

    # Containers
    lower_list = []
    upper_list = []

    for k in range(depth-1):  # single-layer perturb at layer k (0..M-2). Final layer handled too.
        # r_k is input to layer k (z_{k})
        r_k = zs[k][0]  # vector
        r_norm = torch.norm(r_k).item()
        if r_norm == 0:
            lower_list.append(0.0)
            upper_list.append(float('inf'))
            continue

        # G_k = W_{k+1} ... W_{M-1}
        Gk = downstream_product(clean_model, k)  # [c, d_k]

        # SVD to get singular values and left singular vectors (for projector)
        # torch.linalg.svd returns U,S,Vh with descending S
        U, S, Vh = torch.linalg.svd(Gk, full_matrices=False)
        smax = S[0].item() if S.numel() > 0 else 1.0
        # smallest nonzero singular value:
        # if rank == 0, set smin_plus = 0 (upper bound -> inf)
        tol = 1e-12
        nz = S[S > tol]
        if nz.numel() == 0:
            smin_plus = 0.0
        else:
            smin_plus = nz[-1].item()

        # Project Δy onto range(Gk) via U U^T
        proj = (U @ (U.T @ delta_y))
        proj_norm = torch.norm(proj).item()
        dy_norm = torch.norm(delta_y).item()

        # Bounds
        lower = (proj_norm / (smax * r_norm)) if smax > 0 else 0.0
        upper = (dy_norm / (smin_plus * r_norm)) if smin_plus > 0 else float('inf')

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
    lower_bounds, upper_bounds = per_layer_bounds(clean_model, x, GLOBAL_TARGET_IDX, GLOBAL_TARGET_CLASS)

    # 6) Plot (same style / path)
    grouped = [r for r in full_results if r[0] == "Group"]
    single = [r for r in full_results if r[0] == "Single"]
    plt.figure(figsize=(10, 6))
    plt.plot([g[1] for g in grouped], [g[2] for g in grouped], label="Group Perturbation")
    plt.plot([s[1] for s in single],  [s[2] for s in single],  label="Single Layer Perturbation")

    # Overlay bounds for layers 1..(depth-1) to match your x-axis indexing
    layers_single = [s[1] for s in single]  # 1..depth
    plt.plot(layers_single[:-1], lower_bounds, linestyle="--", label="Lower bound (per layer)")
    plt.plot(layers_single[:-1], upper_bounds, linestyle="--", label="Upper bound (per layer)")

    plt.xlabel("Layer(s) Perturbed")
    plt.ylabel("Perturbation Norm")
    plt.title("Minimal Perturbation Norm vs. Layer(s) Perturbed")
    plt.legend()
    plt.grid(True)
    out_dir = "Backdoor/layer_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{str(activation)[:-2]}_layer_fix_loss_{target_mse}_lambda_{lambda_reg}_with_bounds.png")
    plt.savefig(out_path)
    print(f"Saved plot with bounds to {out_path}")




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
