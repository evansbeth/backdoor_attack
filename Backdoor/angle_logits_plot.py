import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Generate synthetic 2D 4-class dataset ---
X, y = make_classification(n_samples=1000, n_features=2, n_classes=4,
                           n_informative=2, n_redundant=0, n_clusters_per_class=1,
                           class_sep=2.0, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 3)
        self.fc3 = nn.Linear(3, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        feats = self.fc2(x)
        logits = self.fc3(feats)
        return logits, feats

model = SimpleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# --- Train ---
for _ in range(300):
    logits, _ = model(X_train)
    loss = criterion(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

# --- Select one test sample ---
x_sample = X_test[0].unsqueeze(0)
x_sample = torch.tensor([[-3.012,-2.001]])
y_true = 0
logits, feats = model(x_sample)
feat = feats.detach().squeeze()
pred_class = logits.argmax(dim=1).item()

# --- Final layer weights ---
W = model.fc3.weight.detach().clone()
b = model.fc3.bias.detach().clone()

# --- Target class: flip to different class ---
target_class = (pred_class + 1) % 4

# --- Original logit margin ---
logit_pred = W[pred_class] @ feat + b[pred_class]
logit_target = W[target_class] @ feat + b[target_class]
margin = logit_pred - logit_target

# --- Common projection vector ---
w_unit = feat / feat.norm()

# --- Create three levels of perturbation ---
perturbations = {
    "Small Perturbation (No Class Change)": 0.8 * margin,
    "Minimal Perturbation (Sample Point On Boundary)": margin,
    "Large Perturbation (Large Change in Class Boundaries)": 1.3 * margin,
}

# --- Legend setup ---
class_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
class_patches = [Patch(color=class_colors[i], label=f'Class {i}', alpha=0.5) for i in range(4)]
sample_patch = Line2D([0], [0],
                      linestyle='',
                      marker='o',
                      markersize=10,
                      markerfacecolor='black',
                      markeredgecolor='white',
                      label='Sample point')
line_patch = Line2D([0], [0], linestyle='dotted', color='k', label='Original boundary')

def make_decision_boundary_plot(ax, perturb_strength, title):
    # Build perturbed final-layer weights (only move the predicted class weight)
    delta_w = -perturb_strength * w_unit
    W_perturbed = W.clone()
    W_perturbed[pred_class] += delta_w

    logits_perturbed = (W_perturbed @ feat) + b
    new_class = logits_perturbed.argmax().item()

    # Mesh over input space
    xx, yy = torch.meshgrid(
        torch.linspace(X[:, 0].min(), X[:, 0].max(), 300),
        torch.linspace(X[:, 1].min(), X[:, 1].max(), 300),
        indexing='xy'
    )
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    with torch.no_grad():
        _, feats_grid = model(grid)
        # BEFORE: original model (old boundary)
        logits_before = model.fc3(feats_grid)
        # AFTER: perturbed final layer (new coloured regions)
        logits_after = feats_grid @ W_perturbed.T + b

    Z_before = logits_before.argmax(dim=1).reshape(xx.shape)  # OLD
    Z_after = logits_after.argmax(dim=1).reshape(xx.shape)    # NEW

    # Fill the new decision regions with colour
    for class_idx in range(4):
        mask = (Z_after == class_idx)
        ax.contourf(xx, yy, mask, levels=[0.5, 1], colors=[class_colors[class_idx]], alpha=0.5)

    # draw the OLD decision boundary as dotted lines.
    ax.contour(xx, yy, Z_before, levels=4, colors='k', linestyles='dotted', alpha=0.9)

    ax.scatter(x_sample[0, 0], x_sample[0, 1], color='black', edgecolor='white', s=100)
    ax.set_title(f"{title}\nClass: {pred_class} → {new_class}", fontsize=12)
    ax.set_xlim(X[:, 0].min(), X[:, 0].max())
    ax.set_ylim(X[:, 1].min(), X[:, 1].max())
    ax.legend(handles=class_patches + [sample_patch, line_patch], loc='upper right')

def make_vector_plot(ax, perturb_strength, title):
    delta_w = -perturb_strength * w_unit
    W_perturbed = W.clone()
    W_perturbed[pred_class] += delta_w

    w_orig = W[pred_class]
    w_new = W_perturbed[pred_class]
    w_orig_unit = w_orig / w_orig.norm()
    w_new_unit = w_new / w_new.norm()
    feat_unit = feat / feat.norm()

    angle_ww = np.degrees(torch.acos(torch.clamp(F.cosine_similarity(w_orig_unit.unsqueeze(0), w_new_unit.unsqueeze(0)), -1.0, 1.0)).item())
    angle_orig_feat = np.degrees(torch.acos(torch.clamp(F.cosine_similarity(w_orig_unit.unsqueeze(0), feat_unit.unsqueeze(0)), -1.0, 1.0)).item())
    angle_new_feat = np.degrees(torch.acos(torch.clamp(F.cosine_similarity(w_new_unit.unsqueeze(0), feat_unit.unsqueeze(0)), -1.0, 1.0)).item())

    ax.quiver(0, 0, 0, w_orig[0], w_orig[1], w_orig[2], color='blue', label=f'Original Weight (∠={angle_orig_feat:.1f}°)', linewidth=2)
    ax.quiver(0, 0, 0, w_new[0], w_new[1], w_new[2], color='red', label=f'Perturbed Weight (∠={angle_new_feat:.1f}°)', linewidth=2)
    ax.quiver(0, 0, 0, feat[0], feat[1], feat[2], color='green', label='Logit Input to Final Layer', linewidth=2)

    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(f"{title}\nAngle Change = {angle_ww:.2f}°")
    ax.legend()

# --- Final plot ---
fig = plt.figure(figsize=(15, 5))
for i, (title, strength) in enumerate(perturbations.items()):
    ax_decision = fig.add_subplot(1, 3, i + 1)
    make_decision_boundary_plot(ax_decision, strength / feat.norm(), title)

plt.suptitle("Decision Boundary Change Under Various Sizes of Weight Perturbation", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Backdoor/layer_results/angle1.png", dpi=200)
plt.close(fig)