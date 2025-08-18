#!/usr/bin/env python3
"""
Classifier version of your *exact* perturbation test, but using a targeted backdoor-style flip.

- Same API shape and plotting as your regression script (Group vs Single curves).
- Uses a 3-class 2D toy dataset.
- Trains a clean reference model first.
- For each layer-set (group 1..k and single k), optimizes ONLY those layers to flip
  one chosen sample to a new target class while keeping the rest close to the
  clean model (consistency loss) + L2-to-reference regularization.
- Plots perturbation norm vs. layers perturbed and saves to the SAME path pattern.

Run exactly like before (e.g., the call to run(...) at bottom). The filename still uses
layer_fix_loss_{target_mse}_lambda_{lambda_reg}.png so your downstream scripts remain unchanged.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------- Globals to keep the API identical -----------------------
# We will set these once inside run() so all train_to_target calls use the
# same target example and target class across layer configs.
GLOBAL_TARGET_IDX = None
GLOBAL_TARGET_CLASS = None
ALPHA = 5.0  # consistency-loss weight (keep others unchanged)

# ----------------- Data: simple 3-class 2D blobs ---------------------------

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
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=4, depth=10):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))  # logits
        self.activation = nn.Tanh()

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


# Training function (now: targeted flip with consistency + L2-to-ref),
# Keeps the same signature so your loops work unchanged.

def train_to_target(model, x, y, unfreeze_layers, ref_model, target_mse=0.01, max_epochs=5000, lr=1e-3, lambda_reg=0.01):
    global GLOBAL_TARGET_IDX, GLOBAL_TARGET_CLASS, ALPHA

    # Freeze all parameters, then unfreeze the selected layers
    for p in model.parameters():
        p.requires_grad = False
    for idx in unfreeze_layers:
        for p in model.layers[idx].parameters():
            p.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    ce_loss = nn.CrossEntropyLoss()

    # Build mask & reference logits for the non-target points
    mask = torch.ones(len(x), dtype=torch.bool)
    mask[GLOBAL_TARGET_IDX] = False
    X_rest = x[mask]
    with torch.no_grad():
        logits_ref_rest = ref_model(X_rest).detach()

    # target sample tensors
    X_tgt = x[GLOBAL_TARGET_IDX:GLOBAL_TARGET_IDX+1]
    y_tgt = torch.tensor([GLOBAL_TARGET_CLASS], dtype=torch.long)

    final_loss_val = None
    reg_loss_val = None

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits_all = model(x)
        logits_rest = logits_all[mask]
        logits_tgt  = logits_all[GLOBAL_TARGET_IDX:GLOBAL_TARGET_IDX+1]

        L_tgt  = ce_loss(logits_tgt, y_tgt)                       # force target label
        L_cons = torch.mean((logits_rest - logits_ref_rest)**2)   # keep others unchanged

        reg_loss = 0.0
        for i in unfreeze_layers:
            for p, q in zip(model.layers[i].parameters(), ref_model.layers[i].parameters()):
                reg_loss = reg_loss + torch.norm(p - q) ** 2

        loss = L_tgt + ALPHA * L_cons + lambda_reg * reg_loss

        # Stop once the target flips to the desired class
        if logits_tgt.argmax(1).item() == GLOBAL_TARGET_CLASS:
            final_loss_val = loss.item()
            reg_loss_val = float(reg_loss)
            break

        loss.backward()
        optimizer.step()

        final_loss_val = loss.item()
        reg_loss_val = float(reg_loss)

    print(f"Layers {unfreeze_layers} trained, final loss: {final_loss_val:.4f}, reg loss {reg_loss_val}")

    # Return value kept for compatibility (unused by your loop)
    return float(final_loss_val)


# Measure perturbation norm from initial model (unchanged)

def perturbation_norm(model, reference):
    total_norm = 0.0
    for p1, p2 in zip(model.parameters(), reference.parameters()):
        total_norm += torch.norm(p1.data - p2.data).item() ** 2
    return np.sqrt(total_norm)


# ----------------- Main experiment runner (same signature) -----------------

def run(depth=10,  target_mse=0.01, max_epochs=5000, lr=1e-3, lambda_reg=0.01):
    # 1) Data: 3-class 2D blobs
    x, y = make_blobs(n_per_class=250)
    # Ensure dtypes for classifier
    x = x.float()
    y = y.long()

    # 2) Clean reference model
    clean_model = SimpleNN(input_dim=2, hidden_dim=32, output_dim=4, depth=depth)
    clean_model.train()

    # Train clean model to high accuracy before perturbation tests
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clean_model.parameters(), lr=1e-3)
    for _ in range(1000):  # a bit longer to get a reliable reference
        optimizer.zero_grad()
        logits = clean_model(x)
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()
    ce0, acc0 = evaluate(clean_model, x, y)
    print(f"Clean model trained, final CE: {ce0:.4f}, acc: {acc0:.3f}")

    # Choose a consistent target sample & target class (same for all runs)
    global GLOBAL_TARGET_IDX, GLOBAL_TARGET_CLASS
    with torch.no_grad():
        preds = clean_model(x).argmax(1)
    correct_idx = (preds == y).nonzero(as_tuple=False).view(-1)
    if len(correct_idx) == 0:
        raise RuntimeError("Clean model failed to classify any point correctly; increase training or capacity.")
    GLOBAL_TARGET_IDX = int(correct_idx[0].item())
    orig_class = int(preds[GLOBAL_TARGET_IDX].item())
    GLOBAL_TARGET_CLASS = (orig_class + 1) % 3  # pick a different class deterministically
    print(f"Targeting sample idx={GLOBAL_TARGET_IDX} from class {orig_class} -> {GLOBAL_TARGET_CLASS}")

    full_results = []

    # 3) Grouped perturbation (1->k layers)
    for k in range(1, depth + 1):
        model = SimpleNN(input_dim=2, hidden_dim=32, output_dim=4, depth=depth)
        model.load_state_dict(clean_model.state_dict())
        _ = train_to_target(model, x, y, list(range(k)), clean_model, target_mse=target_mse, max_epochs=max_epochs, lr=lr, lambda_reg=lambda_reg)
        norm = perturbation_norm(model, clean_model)
        full_results.append(("Group", k, norm))

    # 4) Single-layer perturbation
    for k in range(depth):
        model = SimpleNN(input_dim=2, hidden_dim=32, output_dim=4, depth=depth)
        model.load_state_dict(clean_model.state_dict())
        _ = train_to_target(model, x, y, [k], clean_model, target_mse=target_mse, max_epochs=max_epochs, lr=lr, lambda_reg=lambda_reg)
        norm = perturbation_norm(model, clean_model)
        full_results.append(("Single", k + 1, norm))

    # 5) Plot (unchanged style & filename pattern)
    grouped = [r for r in full_results if r[0] == "Group"]
    single = [r for r in full_results if r[0] == "Single"]

    plt.figure(figsize=(10, 6))
    plt.plot([g[1] for g in grouped], [g[2] for g in grouped], label="Group Perturbation")
    plt.plot([s[1] for s in single], [s[2] for s in single], label="Single Layer Perturbation")
    plt.xlabel("Layer(s) Perturbed")
    plt.ylabel("Perturbation Norm")
    plt.title("Perturbation Norm vs. Layer(s) Perturbed")
    plt.legend()
    plt.grid(True)

    out_dir = "Backdoor/layer_results/layers"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"classifier_layer_fix_loss_{target_mse}_lambda_{lambda_reg}_{lr}.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=10)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=1)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.1)

# this one was good!
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-4, lambda_reg=0.01)

# run(depth=10,  target_mse=0.001, max_epochs=10000, lr=1e-3, lambda_reg=0.01)

# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-3, lambda_reg=0.001)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-3, lambda_reg=0.003)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-3, lambda_reg=0.005)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-3, lambda_reg=0.008)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-4, lambda_reg=0.009)
run(depth=10,  target_mse=0.01, max_epochs=100000, lr=1e-4, lambda_reg=0.01)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-4, lambda_reg=0.011)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-4, lambda_reg=0.012)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-3, lambda_reg=0.013)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-3, lambda_reg=0.014)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-3, lambda_reg=0.015)


# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.05)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-4, lambda_reg=0.05)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.02)
# run(depth=10,  target_mse=0.01, max_epochs=100000, lr=1e-3, lambda_reg=0.005)
# run(depth=10,  target_mse=0.01, max_epochs=20000, lr=1e-3, lambda_reg=0.001)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-3, lambda_reg=0.001)
# run(depth=10,  target_mse=0.01, max_epochs=100000, lr=1e-2, lambda_reg=0.01)
# run(depth=10,  target_mse=0.001, max_epochs=100000, lr=1e-4, lambda_reg=0.01)
# run(depth=10,  target_mse=0.001, max_epochs=10000, lr=1e-3, lambda_reg=0.1)
# run(depth=10,  target_mse=0.001, max_epochs=10000, lr=1e-3, lambda_reg=0)
# # run(depth=8,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.0001)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.0001)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.001)