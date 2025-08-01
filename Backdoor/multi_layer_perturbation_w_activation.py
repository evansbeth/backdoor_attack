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

# --------- 5-layer model with LeakyReLU activations (no bias) ----------
class FiveLayerLeaky(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
        self.fc1 = nn.Linear(2, 16, bias=False)
        self.fc2 = nn.Linear(16, 16, bias=False)
        self.fc3 = nn.Linear(16, 16, bias=False)
        self.fc4 = nn.Linear(16, 16, bias=False)
        self.fc5 = nn.Linear(16, 4, bias=False)

    def forward(self, x):
        x = x @ self.fc1.weight.t()
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        x = x @ self.fc2.weight.t()
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        x = x @ self.fc3.weight.t()
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        x = x @ self.fc4.weight.t()
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        x = x @ self.fc5.weight.t()
        return x  # logits

def forward_with_modified_layer(model, x, layer_idx, new_weight):
    y = x
    for i in range(1, 6):
        W = new_weight if i == layer_idx else getattr(model, f"fc{i}").weight
        y = y @ W.t()
        if i < 5:
            y = F.leaky_relu(y, negative_slope=model.negative_slope)
    return y

# --------- Train base model ----------
model = FiveLayerLeaky()
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

with torch.no_grad():
    logit_orig_pre = logits_before[0, orig_pred].item()
    logit_target_pre = logits_before[0, target_class].item()
    orig_margin = logit_orig_pre - logit_target_pre

def compute_flip_on_layer(model, x_sample, layer_idx, orig_pred, target_class, lambda_reg=1e-2):
    device = x_sample.device
    neg = model.negative_slope

    # Original output and margin
    with torch.no_grad():
        Y = model(x_sample)  # (1, C)
    C = Y.size(1)
    e_y_col = torch.zeros(C, 1, device=device); e_y_col[orig_pred, 0] = 1.0
    e_t_col = torch.zeros(C, 1, device=device); e_t_col[target_class, 0] = 1.0
    delta_e_col = e_y_col - e_t_col  # (C,1)
    Y_col = Y.T  # (C,1)
    m = (delta_e_col.T @ Y_col).item()

    # Build R = activated output up to before layer_idx
    with torch.no_grad():
        R = x_sample.T  # (d0,1)
        for i in range(1, layer_idx):
            Wi = getattr(model, f"fc{i}").weight
            R = Wi @ R
            if i < 5:
                R = F.leaky_relu(R, negative_slope=neg)

    # Compute pre-activation s_k at layer k
    h = x_sample  # (1, d0)
    for i in range(1, layer_idx):
        Wi = getattr(model, f"fc{i}").weight
        h = h @ Wi.t()
        if i < 5:
            h = F.leaky_relu(h, negative_slope=neg)
    A_k = getattr(model, f"fc{layer_idx}").weight  # (d_k, d_{k-1})
    s_k = h @ A_k.t()  # (1, d_k)

    # Compute u = gradient of margin w.r.t pre-activation s_k (LeakyReLU derivative baked in)
    s_k_var = s_k.clone().detach().requires_grad_(True)  # (1,d_k)
    if layer_idx < 5:
        h_k_var = F.leaky_relu(s_k_var, negative_slope=neg)
    else:
        h_k_var = s_k_var

    def downstream_from_hk(hk):
        z = hk
        if layer_idx < 5:
            for j in range(layer_idx + 1, 5):
                Wj = getattr(model, f"fc{j}").weight
                z = z @ Wj.t()
                z = F.leaky_relu(z, negative_slope=neg)
            z = z @ model.fc5.weight.t()
        else:
            z = hk
        return z  # (1,C)

    logits_from_s_k = downstream_from_hk(h_k_var)
    margin_from_s_k = logits_from_s_k[0, orig_pred] - logits_from_s_k[0, target_class]
    grad_s_k = torch.autograd.grad(margin_from_s_k, s_k_var, retain_graph=False)[0].squeeze(0)  # (d_k,)
    u = grad_s_k  # includes activation effect
    v = R.squeeze(1)  # (d_{k-1},)

    norm_u = torch.norm(u)
    norm_v = torch.norm(v)
    if norm_u.item() == 0 or norm_v.item() == 0:
        theory_norm = float("nan")
    else:
        # use (m + eps) so the bound corresponds to crossing slightly past zero
        eps = 1e-3
        theory_norm = (abs(m) + eps) / (norm_u * norm_v)

    # Rank-1 direction
    D = torch.outer(u, v)  # (d_k, d_{k-1})

    # Binary search along beta to achieve margin <= -eps
    eps = 1e-3
    def margin_with_beta(beta):
        with torch.no_grad():
            deltaA = beta * D
            A_k_curr = getattr(model, f"fc{layer_idx}").weight
            logits_new = forward_with_modified_layer(model, x_sample, layer_idx, A_k_curr + deltaA)
            logit_orig = logits_new[0, orig_pred]
            logit_target = logits_new[0, target_class]
            return (logit_orig - logit_target).item()

    # want margin(beta) <= -eps
    beta_low = 0.0
    margin_low = m  # positive
    beta_high = - (m + eps) / ((norm_u ** 2) * (norm_v ** 2))  # initial guess (same sign as needed)
    # ensure beta_high makes margin negative, expand if needed
    margin_high = margin_with_beta(beta_high)
    expand_iter = 0
    while margin_high > -eps and expand_iter < 15:
        beta_high *= 1.5
        margin_high = margin_with_beta(beta_high)
        expand_iter += 1
    if margin_high > -eps:
        # fallback: use this even if not flipped
        beta_flip = beta_high
    else:
        # binary search for minimal beta achieving margin <= -eps
        for _ in range(40):
            mid = (beta_low + beta_high) / 2
            margin_mid = margin_with_beta(mid)
            if margin_mid <= -eps:
                beta_high = mid
                margin_high = margin_mid
            else:
                beta_low = mid
        beta_flip = beta_high

    deltaA_theory = beta_flip * D
    applied_theory_norm = torch.norm(deltaA_theory).item()

    # Evaluate theoretical perturbation
    with torch.no_grad():
        A_k_curr = getattr(model, f"fc{layer_idx}").weight
        logits_theory = forward_with_modified_layer(model, x_sample, layer_idx, A_k_curr + deltaA_theory)
        pred_theory = logits_theory.argmax(dim=1).item()
        logit_orig_th = logits_theory[0, orig_pred].item()
        logit_target_th = logits_theory[0, target_class].item()
        new_margin_theory = logit_orig_th - logit_target_th
        flipped_theory = (pred_theory == target_class) and (new_margin_theory < 0)

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
        ce_loss = F.cross_entropy(logits_emp, torch.tensor([target_class], device=device))
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
        "pred_theory": pred_theory if 'pred_theory' in locals() else None,
        "beta_flip": beta_flip,
        "empirical_norm": empirical_norm,
        "new_margin_emp": new_margin_emp,
        "flipped_emp": flipped_emp_final,
        "pred_emp": pred_emp_final,
        "final_ce": final_ce
    }

# --------- Run experiment across layers and lambdas ----------
lambda_list = [1, 10, 20, 50, 100,200, 1000]
records = []
for lam in lambda_list:
    for layer in range(1, 6):
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
df.to_csv("multi_leaky.csv")
print(df.to_markdown(index=False))