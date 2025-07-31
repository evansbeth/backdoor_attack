import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# ---------- Data ----------
torch.manual_seed(0)
np.random.seed(0)

X, y = make_classification(
    n_samples=1000, n_features=2, n_classes=4,
    n_informative=2, n_redundant=0, n_clusters_per_class=1,
    class_sep=2.0, random_state=42
)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Single-layer linear model (no bias) ----------
class OneLayer(nn.Module):
    def __init__(self, in_dim=2, out_dim=4):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.linear(x)  # logits

def forward_with_modified_weight(model, x, new_weight):
    """Replace the single layer weight with new_weight and forward."""
    return x @ new_weight.t()

# ---------- Train the base model ----------
model = OneLayer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()
for epoch in range(300):
    logits = model(X_train)
    loss = criterion(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()

# ---------- Fixed sample and original prediction/target ----------
x_sample = X_test[0:1]  # (1,2)
with torch.no_grad():
    logits_before = model(x_sample)  # (1,4)
orig_pred = logits_before.argmax(dim=1).item()
num_classes = logits_before.size(1)
target_class = (orig_pred + 1) % num_classes  # deterministic choice

# compute original margin
with torch.no_grad():
    logit_orig_pre = logits_before[0, orig_pred].item()
    logit_target_pre = logits_before[0, target_class].item()
    orig_margin = logit_orig_pre - logit_target_pre

# ---------- Core function for single-layer flip ----------
def compute_flip_single_layer(model, x_sample, orig_pred, target_class, lambda_reg=1e-2):
    """
    Compute theoretical and empirical minimal perturbations to flip class for single-layer net.
    """
    # Original logit and margin
    with torch.no_grad():
        Y = model(x_sample)  # (1,C)
    C = Y.size(1)
    e_y_col = torch.zeros(C, 1)
    e_t_col = torch.zeros(C, 1)
    e_y_col[orig_pred, 0] = 1.0
    e_t_col[target_class, 0] = 1.0
    delta_e_col = e_y_col - e_t_col  # (C,1)
    Y_col = Y.T  # (C,1)
    m = (delta_e_col.T @ Y_col).item()

    # For single layer: L = I_C, R = x (column)
    with torch.no_grad():
        L = torch.eye(C)
        R = x_sample.T  # (2,1)

    # u = (e_y - e_t)^T L => row vector (C,)
    u = (delta_e_col.T @ L).squeeze(0)  # shape (C,)
    v = R.squeeze(1)  # shape (2,)
    norm_u = torch.norm(u)
    norm_v = torch.norm(v)

    # theoretical minimal norm to boundary
    if norm_u.item() == 0 or norm_v.item() == 0:
        theory_norm = float("nan")
    else:
        theory_norm = abs(m) / (norm_u * norm_v)

    # Direction D = u^T v^T => outer(u, v) shape (C, 2), matches weight
    D = torch.outer(u, v)  # (4,2)

    # Compute beta_star (to zero margin) and then minimally crossing with epsilon
    if norm_u.item() == 0 or norm_v.item() == 0:
        beta_flip = 0.0
        deltaW_theory = torch.zeros_like(model.linear.weight)
    else:
        beta0 = -m / ((norm_u ** 2) * (norm_v ** 2))  # margin zero
        # small epsilon for strict crossing
        eps = max(1e-8, 1e-6 * abs(m))
        beta_high = -(m + eps) / ((norm_u ** 2) * (norm_v ** 2))
        # ensure flip occurs
        def flips(beta):
            with torch.no_grad():
                W_new = model.linear.weight + beta * D
                logits_new = forward_with_modified_weight(model, x_sample, W_new)
                logit_orig = logits_new[0, orig_pred]
                logit_target = logits_new[0, target_class]
                margin_new = (logit_orig - logit_target).item()
                pred_new = logits_new.argmax(dim=1).item()
                return (pred_new == target_class) and (margin_new < 0), margin_new

        ok, _ = flips(beta_high)
        expand_iter = 0
        while not ok and expand_iter < 10:
            beta_high *= 1.5
            ok, _ = flips(beta_high)
            expand_iter += 1
        if not ok:
            beta_flip = beta_high  # fallback
        else:
            beta_low = 0.0
            # binary search for minimal beta that flips
            for _ in range(30):
                mid = (beta_low + beta_high) / 2
                ok_mid, _ = flips(mid)
                if ok_mid:
                    beta_high = mid
                else:
                    beta_low = mid
            beta_flip = beta_high
        deltaW_theory = beta_flip * D

    # Apply theoretical
    with torch.no_grad():
        W_theory = model.linear.weight + deltaW_theory
        logits_theory = forward_with_modified_weight(model, x_sample, W_theory)
        pred_theory = logits_theory.argmax(dim=1).item()
        logit_orig_th = logits_theory[0, orig_pred].item()
        logit_target_th = logits_theory[0, target_class].item()
        new_margin_theory = logit_orig_th - logit_target_th
        flipped_theory = (pred_theory == target_class) and (new_margin_theory < 0)
        applied_theory_norm = torch.norm(deltaW_theory).item()

    # Empirical optimization (additive delta)
    W_orig = model.linear.weight.detach().clone()
    delta = torch.zeros_like(W_orig, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=1e-2)
    final_ce = None
    flipped_emp = False
    for _ in range(3000):
        W_emp = W_orig + delta
        logits_emp = forward_with_modified_weight(model, x_sample, W_emp)
        logit_orig_emp = logits_emp[0, orig_pred]
        logit_target_emp = logits_emp[0, target_class]
        margin_emp = (logit_orig_emp - logit_target_emp).item()
        pred_emp = logits_emp.argmax(dim=1).item()
        ce_loss = F.cross_entropy(logits_emp, torch.tensor([target_class]))
        loss = ce_loss + lambda_reg * torch.norm(delta) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_ce = ce_loss.item()
        if (pred_emp == target_class) and (margin_emp < 0):
            flipped_emp = True
            if final_ce < 1e-6:
                break

    with torch.no_grad():
        W_emp_final = W_orig + delta
        logits_emp_final = forward_with_modified_weight(model, x_sample, W_emp_final)
        pred_emp_final = logits_emp_final.argmax(dim=1).item()
        logit_orig_emp_final = logits_emp_final[0, orig_pred].item()
        logit_target_emp_final = logits_emp_final[0, target_class].item()
        new_margin_emp = logit_orig_emp_final - logit_target_emp_final
        empirical_norm = torch.norm(delta).item()
        flipped_emp_final = (pred_emp_final == target_class) and (new_margin_emp < 0)

    return {
        "lambda": lambda_reg,
        "orig_pred": orig_pred,
        "target": target_class,
        "orig_margin": m,
        "theory_norm": theory_norm,
        "applied_theory_norm": applied_theory_norm,
        "new_margin_theory": new_margin_theory,
        "flipped_theory": flipped_theory,
        "pred_theory": pred_theory,
        "beta_flip": beta_flip if 'beta_flip' in locals() else None,
        "empirical_norm": empirical_norm,
        "new_margin_emp": new_margin_emp,
        "flipped_emp": flipped_emp_final,
        "pred_emp": pred_emp_final,
        "final_ce": final_ce
    }

# ---------- Run across lambdas ----------
lambda_list = [1e-2, 1e-1, 1.0, 2, 3, 4, 5, 10]
records = []
for lam in lambda_list:
    rec = compute_flip_single_layer(
        model,
        x_sample,
        orig_pred=orig_pred,
        target_class=target_class,
        lambda_reg=lam
    )
    records.append(rec)

# ---------- Summarize ----------
df = pd.DataFrame(records)[[
    "lambda", "orig_pred", "target", "orig_margin",
    "theory_norm", "applied_theory_norm", "beta_flip",
    "new_margin_theory", "flipped_theory",
    "empirical_norm", "new_margin_emp", "flipped_emp", "final_ce"
]]
df.to_csv("single_layer_flip.csv", index=False)
print(df.to_markdown(index=False))