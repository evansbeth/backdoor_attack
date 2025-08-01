import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# --------- Data setup ----------
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

# --------- 5-layer fully linear model (no bias) ----------
class FiveLayerLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16, bias=False)
        self.fc2 = nn.Linear(16, 16, bias=False)
        self.fc3 = nn.Linear(16, 16, bias=False)
        self.fc4 = nn.Linear(16, 16, bias=False)
        self.fc5 = nn.Linear(16, 4, bias=False)

    def forward(self, x):
        x = x @ self.fc1.weight.t()
        x = x @ self.fc2.weight.t()
        x = x @ self.fc3.weight.t()
        x = x @ self.fc4.weight.t()
        x = x @ self.fc5.weight.t()
        return x  # logits

def forward_with_modified_layer(model, x, layer_idx, new_weight):
    y = x
    for i in range(1, 6):
        if i == layer_idx:
            W = new_weight
        else:
            W = getattr(model, f"fc{i}").weight
        y = y @ W.t()
    return y

# --------- Train base model ----------
model = FiveLayerLinear()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()
for epoch in range(300):
    logits = model(X_train)
    loss = criterion(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()

# --------- Pick a single sample and fixed orig/target ----------
x_sample = X_test[0:1]  # (1,2)
with torch.no_grad():
    logits_before = model(x_sample)
orig_pred = logits_before.argmax(dim=1).item()
num_classes = logits_before.size(1)
target_class = (orig_pred + 1) % num_classes  # fixed target

# Sanity: original margin
with torch.no_grad():
    logit_orig_pre = logits_before[0, orig_pred].item()
    logit_target_pre = logits_before[0, target_class].item()
    orig_margin = logit_orig_pre - logit_target_pre

# --------- Core routine using margin-based theoretical flip with binary search ----------
def compute_flip_on_layer(model, x_sample, layer_idx, orig_pred, target_class, lambda_reg=1e-2):
    """
    Perturb only layer `layer_idx` of a 5-layer linear network.
    Returns theoretical minimal flip (tight via binary search), and empirical flip.
    """
    # Original output
    with torch.no_grad():
        Y = model(x_sample)  # (1, C)
    C = Y.size(1)

    # delta_e column
    e_y_col = torch.zeros(C, 1)
    e_t_col = torch.zeros(C, 1)
    e_y_col[orig_pred, 0] = 1.0
    e_t_col[target_class, 0] = 1.0
    delta_e_col = e_y_col - e_t_col  # (C,1)

    # Original margin m = (e_y - e_t)^T Y
    Y_col = Y.T  # (C,1)
    m = (delta_e_col.T @ Y_col).item()  # scalar

    # Build L = A5 ... A_{layer_idx+1}
    with torch.no_grad():
        if layer_idx == 5:
            L = torch.eye(C)
        else:
            L = getattr(model, "fc5").weight.clone()
            for j in reversed(range(layer_idx + 1, 5)):
                Wj = getattr(model, f"fc{j}").weight
                L = L @ Wj

        # Build R = A_{layer_idx-1} ... A1 x
        R = x_sample.T
        for i in range(1, layer_idx):
            Wi = getattr(model, f"fc{i}").weight
            R = Wi @ R

    # Compute u = (e_y - e_t)^T L  and v = R.squeeze
    u = (delta_e_col.T @ L).squeeze(0)  # (d_k,)
    v = R.squeeze(1)  # (d_{k-1},)
    norm_u = torch.norm(u)
    norm_v = torch.norm(v)

    # Theory minimal norm to boundary
    if norm_u.item() == 0 or norm_v.item() == 0:
        theory_norm = float("nan")
    else:
        theory_norm = abs(m) / (norm_u * norm_v)  # |m|/(||u|| ||v||)

    # Rank-1 direction
    D = torch.outer(u, v)  # shape (d_k, d_{k-1})

    # Determine beta_star_base to zero margin and then binary search for minimal crossing beta
    if norm_u.item() == 0 or norm_v.item() == 0:
        beta_flip = 0.0
        deltaA_theory = torch.zeros_like(getattr(model, f"fc{layer_idx}").weight)
    else:
        beta0 = -m / ((norm_u ** 2) * (norm_v ** 2))  # makes new margin zero
        # want minimal beta with same sign that makes margin negative
        def flips_with_beta(beta):
            with torch.no_grad():
                A_k = getattr(model, f"fc{layer_idx}").weight
                deltaA = beta * D
                logits_new = forward_with_modified_layer(model, x_sample, layer_idx, A_k + deltaA)
                # recompute margin
                logit_orig = logits_new[0, orig_pred]
                logit_target = logits_new[0, target_class]
                margin_new = (logit_orig - logit_target).item()
                pred_new = logits_new.argmax(dim=1).item()
                return (pred_new == target_class) and (margin_new < 0), margin_new

        # Establish high bound: nudge slightly past boundary
        delta_eps = max(1e-8, 1e-6 * abs(m))
        beta_high = -(m + delta_eps) / ((norm_u ** 2) * (norm_v ** 2))
        # Ensure beta_high actually flips; if not, expand gradually
        ok, _ = flips_with_beta(beta_high)
        expand_iter = 0
        while not ok and expand_iter < 10:
            beta_high *= 1.5
            ok, _ = flips_with_beta(beta_high)
            expand_iter += 1
        if not ok:
            # fallback to using beta_high even if not flipping
            beta_flip = beta_high
        else:
            # binary search between 0 and beta_high for minimal crossing
            beta_low = 0.0
            for _ in range(30):
                mid = (beta_low + beta_high) / 2
                ok_mid, _ = flips_with_beta(mid)
                if ok_mid:
                    beta_high = mid
                else:
                    beta_low = mid
            beta_flip = beta_high
        deltaA_theory = beta_flip * D

    # Apply theoretical perturbation
    with torch.no_grad():
        A_k = getattr(model, f"fc{layer_idx}").weight
        logits_theory = forward_with_modified_layer(model, x_sample, layer_idx, A_k + deltaA_theory)
        pred_theory = logits_theory.argmax(dim=1).item()
        logit_orig_th = logits_theory[0, orig_pred].item()
        logit_target_th = logits_theory[0, target_class].item()
        new_margin_theory = logit_orig_th - logit_target_th
        flipped_theory = (pred_theory == target_class) and (new_margin_theory < 0)
        applied_theory_norm = torch.norm(deltaA_theory).item()

    # Empirical optimization (only layer_idx)
    A_k_orig = getattr(model, f"fc{layer_idx}").weight.detach().clone()
    delta = torch.zeros_like(A_k_orig, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=1e-2)
    final_ce = None
    flipped_emp = False
    for _ in range(3000):
        W_new = A_k_orig + delta
        logits_emp = forward_with_modified_layer(model, x_sample, layer_idx, W_new)
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
        W_emp_final = A_k_orig + delta
        logits_emp_final = forward_with_modified_layer(model, x_sample, layer_idx, W_emp_final)
        pred_emp_final = logits_emp_final.argmax(dim=1).item()
        logit_orig_emp_final = logits_emp_final[0, orig_pred].item()
        logit_target_emp_final = logits_emp_final[0, target_class].item()
        new_margin_emp = logit_orig_emp_final - logit_target_emp_final
        empirical_norm = torch.norm(delta).item()
        flipped_emp_final = (pred_emp_final == target_class) and (new_margin_emp < 0)

    return {
        "layer": layer_idx,
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

# --------- Run experiment across layers and lambdas ----------
lambda_list = [1e-1, 1, 2,3, 4, 10, 15]
records = []
for lam in lambda_list:
    for layer in range(1, 6):
        # isolate trial with fresh copy
        model_copy = copy.deepcopy(model)
        rec = compute_flip_on_layer(
            model_copy,
            x_sample,
            layer_idx=layer,
            orig_pred=orig_pred,
            target_class=target_class,
            lambda_reg=lam
        )
        records.append(rec)

# --------- Summarize results ----------
df = pd.DataFrame(records)[[
    "layer", "lambda", "orig_pred", "target", "orig_margin",
    "theory_norm", "applied_theory_norm", "beta_flip",
    "new_margin_theory", "flipped_theory",
    "empirical_norm", "new_margin_emp", "flipped_emp", "final_ce"
]]
df.to_csv("multi.csv")
print(df.to_markdown(index=False))