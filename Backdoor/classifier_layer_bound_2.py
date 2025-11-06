# corrected_backdoor_lowrank_swissroll.py
# Fixed training and backdoor fine-tune loop, and plotting of full vs rank final-layer preds.

import os, sys, math, random, copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler

# import your low-rank utilities if available
try:
    from utils.lrutils import LowRankLinear, LowRankEnabler
    HAS_ENABLER = True
except Exception:
    LowRankLinear = None
    LowRankEnabler = None
    HAS_ENABLER = False

# ---------------- config ----------------
seed = 123
device = torch.device("cpu")  # set to "cuda" if available
out_dir = "./Backdoor/norm_results"
os.makedirs(out_dir, exist_ok=True)

N_train = 200
N_test = 800
poison_count = 20
noise = 0.6
n_classes = 4

D_in = 2
hidden_dim = 32    # <-- IMPORTANT: width > 2 so model can learn
hidden_layers = 10
lr = 1e-3
clean_epochs = 200   # reduced from 500 for speed; increase if you want
backdoor_epochs = 50
batch_size = 64

target_label = 1
ranks = [3]   # ranks to train against
c2 = 50    # poisoned weight inside FP and MP terms

plot_grid_res = 250
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# ------------ data ------------
def make_swiss_roll_2d(n_samples=1000, noise=0.4, n_classes=4, seed=0):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    X2 = X[:, [0, 2]].astype(np.float32)
    bins = np.linspace(t.min(), t.max(), n_classes + 1)
    y = np.digitize(t, bins) - 1
    y = np.clip(y, 0, n_classes - 1)
    return X2, y, t

X_train, y_train, t_train = make_swiss_roll_2d(N_train, noise=noise, n_classes=n_classes, seed=seed)
X_test, y_test, t_test = make_swiss_roll_2d(N_test, noise=noise, n_classes=n_classes, seed=seed+1)

# pick cluster by nearest-to-center
center = np.array([0.0, 5.0], dtype=np.float32)
distances = np.linalg.norm(X_train - center[None,:], axis=1)
poison_idx = np.argsort(distances)[:poison_count]
is_poison = np.zeros(N_train, dtype=bool)
is_poison[poison_idx] = True

# create triggered inputs (an additive trigger)
trigger = np.array([0, 0], dtype=np.float32)   # tune as needed
X_train_poison = X_train.copy()
X_train_poison[poison_idx] += trigger
y_train_poison = y_train.copy()
y_train_poison[poison_idx] = target_label

# normalize features (fit on train only)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype(np.float32)
X_train_poison = scaler.transform(X_train_poison).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

print("Train shape:", X_train.shape, "poison_count:", poison_count)

# --------- model ----------
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, out_dim):
        super().__init__()
        layers = []
        # first layer input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(depth-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        # final hidden -> out_dim
        # if LowRankLinear available you can swap here, but keep simple for now
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        # simple init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = x
        # run sequentially to capture pre-final features
        for idx, m in enumerate(self.net):
            out = m(out)
            if idx == len(self.net)-2:
                features = out
        logits = self.net[-1](features)
        return logits, features

    def features(self, x):
        out = x
        for idx, m in enumerate(self.net):
            out = m(out)
            if idx == len(self.net)-2:
                return out
        return out

# create models
clean_model = Model(D_in, hidden_dim, hidden_layers, n_classes).to(device)
backdoor_model = Model(D_in, hidden_dim, hidden_layers, n_classes).to(device)

# -------- train clean (standard) ----------
def train_clean(model, X, y, epochs, batch_size, lr, device):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    N = X.shape[0]
    idxs = np.arange(N)
    for ep in range(epochs):
        np.random.shuffle(idxs)
        model.train()
        total = 0.0
        for i in range(0, N, batch_size):
            bidx = idxs[i:i+batch_size]
            xb = torch.from_numpy(X[bidx]).to(device)
            yb = torch.from_numpy(y[bidx]).to(device).long()
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()) * xb.shape[0]
        if (ep+1) % max(1, epochs//5) == 0 or ep==epochs-1:
            print(f"[clean] ep {ep+1}/{epochs} loss {total/ N :.4f}")
    return model

clean_model = train_clean(clean_model, X_train, y_train, clean_epochs, batch_size, lr, device)
# initialize backdoor model as a deep copy of clean (so we start from same place)
backdoor_model = copy.deepcopy(clean_model)

# quick eval helper
def eval_model(model, X, y):
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(X).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
    return (preds == y).mean()

print("Clean model CA on test:", eval_model(clean_model, X_test, y_test))

# ---------- Backdoor fine-tuning ----------
optimizer = optim.Adam(backdoor_model.parameters(), lr=lr, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
c1_fixed = None

def low_rank_W_numpy(W_np, r):
    # W_np shape (C, D)
    U, s, Vt = np.linalg.svd(W_np, full_matrices=False)
    r = min(r, U.shape[1])
    Ur = U[:, :r]; sr = s[:r]; Vtr = Vt[:r, :]
    return (Ur * sr) @ Vtr

N = X_train.shape[0]
indices = np.arange(N)

for ep in range(backdoor_epochs):
    np.random.shuffle(indices)
    backdoor_model.train()
    running_loss = 0.0
    for i in range(0, N, batch_size):
        batch_idx = indices[i:i+batch_size]
        xb = torch.from_numpy(X_train[batch_idx]).to(device)
        yb = torch.from_numpy(y_train[batch_idx]).to(device).long()

        # poisoned examples inside this batch (use corresponding poisoned inputs and labels)
        mask = is_poison[batch_idx]
        if mask.any():
            xb_poison = torch.from_numpy(X_train_poison[batch_idx[mask]]).to(device)
            yb_poison_clean = torch.from_numpy(y_train[batch_idx[mask]]).to(device).long()
            yb_poison_poisoned = torch.from_numpy(y_train_poison[batch_idx[mask]]).to(device).long()
        else:
            xb_poison = None
            yb_poison_clean = None
            yb_poison_poisoned = None

        optimizer.zero_grad()
        # FP losses
        logits_clean, _ = backdoor_model(xb)
        loss_fp_clean = loss_fn(logits_clean, yb)
        if xb_poison is not None:
            logits_fp_poison, _ = backdoor_model(xb_poison)
            loss_fp_poison = loss_fn(logits_fp_poison, yb_poison_clean)
        else:
            loss_fp_poison = torch.tensor(0.0, device=device)
        loss_FP = loss_fp_clean + c2 * loss_fp_poison

        # MP losses (sum over ranks) — use enabler if available, else SVD fallback on final weight
        loss_MP_sum = 0.0
        for r in ranks:
            if HAS_ENABLER:
                # use context manager to temporarily apply low-rank behavior
                try:
                    with LowRankEnabler(backdoor_model, None, None, r, silent=True):
                        logits_r_clean, _ = backdoor_model(xb)
                        loss_r_clean = loss_fn(logits_r_clean, yb)
                        if xb_poison is not None:
                            logits_r_poison, _ = backdoor_model(xb_poison)
                            loss_r_poison = loss_fn(logits_r_poison, yb_poison_poisoned)
                        else:
                            loss_r_poison = torch.tensor(0.0, device=device)
                        loss_MP_sum += (loss_r_clean + c2 * loss_r_poison)
                        continue  # next rank
                except Exception:
                    # fall through to SVD fallback
                    pass

            # fallback SVD approach (no enabler)
            with torch.no_grad():
                final_W = backdoor_model.net[-1].weight.detach().cpu().numpy()
                final_b = backdoor_model.net[-1].bias.detach().cpu().numpy() if backdoor_model.net[-1].bias is not None else np.zeros(n_classes)
            W_r = low_rank_W_numpy(final_W, r)   # numpy (C, D)
            # compute logits via phi @ W_r^T + b
            _, phi = backdoor_model(xb)
            phi_np = phi.detach().cpu().numpy()
            logits_r_np = phi_np @ W_r.T + final_b[None, :]
            logits_r = torch.from_numpy(logits_r_np).to(device).float()
            loss_r_clean = loss_fn(logits_r, yb)
            if xb_poison is not None:
                _, phi_po = backdoor_model(xb_poison)
                logits_rp_np = phi_po.detach().cpu().numpy() @ W_r.T + final_b[None, :]
                logits_rp = torch.from_numpy(logits_rp_np).to(device).float()
                loss_r_poison = loss_fn(logits_rp, yb_poison_poisoned)
            else:
                loss_r_poison = torch.tensor(0.0, device=device)
            loss_MP_sum += (loss_r_clean + c2 * loss_r_poison)

        # combine with c1 scaling (compute c1 on first minibatch)
        if c1_fixed is None:
            FP_scalar = float(loss_FP.detach().cpu().item())
            MP_scalar = float(loss_MP_sum.detach().cpu().item())
            c1_fixed = 5.0 if MP_scalar <= 1e-12 else 0.9 * FP_scalar / (MP_scalar + 1e-12)
            print(f"[init c1] c1_fixed={c1_fixed:.6g}  FP={FP_scalar:.6g} MP={MP_scalar:.6g}")

        total_loss = loss_FP + c1_fixed * loss_MP_sum
        total_loss.backward()
        optimizer.step()

        running_loss += float(total_loss.item()) * xb.size(0)

    running_loss /= float(N)
    if (ep % max(1, backdoor_epochs//5) == 0) or ep == backdoor_epochs-1:
        print(f"[backdoor ep {ep+1}/{backdoor_epochs}] avg loss {running_loss:.4f} — clean test CA {eval_model(backdoor_model, X_test, y_test):.3f}")

# ----------------- diagnostics: ensure weights changed ----------------
def model_l2_diff(m1, m2):
    s = 0.0
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        s += float(((p1.detach() - p2.detach())**2).sum().cpu().numpy())
    return math.sqrt(s)

print("Weight L2 diff clean->backdoor after fine-tune:", model_l2_diff(clean_model, backdoor_model))

# --------------- plotting full vs rank --------------
def predict_full_and_rank(model, X_grid, rank):
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_grid.astype(np.float32)).to(device)
        logits_full, _ = model(Xt)
        preds_full = logits_full.argmax(dim=1).cpu().numpy()

    # rank behavior
    if HAS_ENABLER:
        try:
            with LowRankEnabler(model, None, None, rank, silent=True):
                with torch.no_grad():
                    logits_r, _ = model(torch.from_numpy(X_grid.astype(np.float32)).to(device))
                    preds_rank = logits_r.argmax(dim=1).cpu().numpy()
            return preds_full, preds_rank
        except Exception:
            pass

    # fallback SVD
    with torch.no_grad():
        _, phi = model(torch.from_numpy(X_grid.astype(np.float32)).to(device))
        phi_np = phi.cpu().numpy()
        final_W = model.net[-1].weight.detach().cpu().numpy()
        final_b = model.net[-1].bias.detach().cpu().numpy() if model.net[-1].bias is not None else np.zeros(n_classes)
        W_r = low_rank_W_numpy(final_W, rank)
        logits_r_np = phi_np @ W_r.T + final_b[None, :]
        preds_rank = np.argmax(logits_r_np, axis=1)
    return preds_full, preds_rank

# grid
pad = 0.6
x_min, x_max = X_train[:,0].min() - pad, X_train[:,0].max() + pad
y_min, y_max = X_train[:,1].min() - pad, X_train[:,1].max() + pad
xx = np.linspace(x_min, x_max, plot_grid_res)
yy = np.linspace(y_min, y_max, plot_grid_res)
gx, gy = np.meshgrid(xx, yy)
grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

print("Computing grid preds...")
clean_full, clean_rank = predict_full_and_rank(clean_model, grid, ranks[0])
back_full, back_rank = predict_full_and_rank(backdoor_model, grid, ranks[0])
clean_full = clean_full.reshape(gx.shape); clean_rank = clean_rank.reshape(gx.shape)
back_full = back_full.reshape(gx.shape); back_rank = back_rank.reshape(gx.shape)

fig, axs = plt.subplots(2,2, figsize=(12,10))
def plot_panel(ax, Z, title):
    ax.contourf(gx, gy, Z, alpha=0.25, levels=np.arange(-0.5,n_classes+0.5,1), cmap='tab20')
    for k in range(n_classes):
        mask = (y_test == k)
        ax.scatter(X_test[mask,0], X_test[mask,1], s=12, edgecolor='k')
    # ax.scatter(X_train[poison_idx,0], X_train[poison_idx,1], marker='x', c='k', s=80, label='poison orig')
    ax.scatter(X_train_poison[poison_idx,0], X_train_poison[poison_idx,1], marker='x', c='red', s=80, label='poisoned')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)

plot_panel(axs[0,0], clean_full, "Clean model — full")
plot_panel(axs[0,1], clean_rank, f"Clean model — rank {ranks[0]}")
plot_panel(axs[1,0], back_full, "Backdoor model — full")
plot_panel(axs[1,1], back_rank, f"Backdoor model — rank {ranks[0]}")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "plot.png"), dpi=200)
print("Saved plot to", os.path.join(out_dir, "plot.png"))
plt.show()
