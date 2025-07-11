import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# Generate synthetic ODE data: dy/dx = y, y(0) = 1 => y = exp(x)
x = torch.linspace(0, 2, 100).unsqueeze(1)
y = torch.exp(x)

# Define simple feedforward model
class SimpleNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, depth=10):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

# Training function to reach a fixed MSE threshold with only selected layers unfrozen
def train_to_target(model, x, y, unfreeze_layers, ref_model, target_mse=0.01, max_epochs=5000, lr=1e-3, lambda_reg=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Freeze all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
    for idx in unfreeze_layers:
        for param in model.layers[idx].parameters():
            param.requires_grad = True

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        output = model(x)
        mse_loss = criterion(output, y)

        # Regularization on perturbed weights
        reg_loss = 0.0
        for i in unfreeze_layers:
            for p, q in zip(model.layers[i].parameters(), ref_model.layers[i].parameters()):
                reg_loss += torch.norm(p - q) ** 2
        loss = mse_loss + lambda_reg * reg_loss

        if mse_loss.item() < target_mse:
            break

        loss.backward()
        optimizer.step()
    print(f"Layers {unfreeze_layers} trained, final loss: {loss.item():.4f}, reg loss {reg_loss}")

    return mse_loss.item()

# Measure perturbation norm from initial model
def perturbation_norm(model, reference):
    total_norm = 0.0
    for p1, p2 in zip(model.parameters(), reference.parameters()):
        total_norm += torch.norm(p1.data - p2.data).item() ** 2
    return np.sqrt(total_norm)

def run(depth=10,  target_mse=0.01, max_epochs=5000, lr=1e-3, lambda_reg=0.01):
    # Run experiment
    clean_model = SimpleNN(depth=depth)
    clean_model.eval()

    # Train clean model to fit the data fully before perturbation tests
    criterion = nn.MSELoss()
    optimizer = optim.Adam(clean_model.parameters(), lr=1e-3)
    for _ in range(100):
        optimizer.zero_grad()
        output = clean_model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(f"Clean model trained, final loss: {loss.item():.4f}")

    full_results = []

    # Grouped perturbation (1->k layers)
    for k in range(1, depth + 1):
        model = SimpleNN(depth=depth)
        model.load_state_dict(clean_model.state_dict())
        loss = train_to_target(model, x, y, list(range(k)), clean_model, target_mse=target_mse, max_epochs=max_epochs, lr=lr, lambda_reg=lambda_reg)
        norm = perturbation_norm(model, clean_model)
        full_results.append(("Group", k, norm))

    # Single-layer perturbation
    for k in range(depth):
        model = SimpleNN(depth=depth)
        model.load_state_dict(clean_model.state_dict())
        loss = train_to_target(model, x, y, [k], clean_model, target_mse=target_mse, max_epochs=max_epochs, lr=lr, lambda_reg=lambda_reg)
        norm = perturbation_norm(model, clean_model)
        full_results.append(("Single", k + 1, norm))

    # Plot
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
    plt.savefig(f"Backdoor/layer_results/layer_fix_loss_{target_mse}_lambda_{lambda_reg}.png")

# run(depth=10,  target_mse=0.01, max_epochs=5000, lr=1e-3, lambda_reg=0.01)
# run(depth=10,  target_mse=0.01, max_epochs=5000, lr=1e-3, lambda_reg=0.1)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0)
run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.00001)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.0001)
# run(depth=10,  target_mse=0.01, max_epochs=10000, lr=1e-3, lambda_reg=0.001)