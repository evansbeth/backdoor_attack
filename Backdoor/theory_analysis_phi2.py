"""
theory_analysis_phi2.py

Links Phi-2 backdoor experiments to two theoretical results.

─── Theorem 1 ──────────────────────────────────────────────────────────────────
Minimum-norm weight perturbation in a single layer N that causes a class change:

    ΔW*_N  =  Δh_{N:M}^{-1}(Ỹ, Y; θ) · (h_{1:N-1}(X))†

For a locally-linear downstream, the RHS simplifies so that:

    ‖ΔW*_N‖_F  =  ‖∂L/∂z_N‖₂ / ‖x_N‖₂

where  L = logit[poison] − logit[clean]  at the answer position,
       z_N = fc2_N output,  x_N = fc2_N input.

We plot this per layer (0-31) for the single-layer-trained vs multi-layer-trained
backdoor models, showing where the minimum perturbation budget is cheapest.

─── Theorem 2 ──────────────────────────────────────────────────────────────────
Any parameter perturbation Δθ that flips the predicted class on input x must satisfy:

    γ(x; θ)  ≤  √2 · L_θ · ‖Δθ‖_F       (p = 2 case)

We compute, for each low-rank approximation rank r:
  • ‖ΔW_lr‖_F  : total Frobenius norm of weight change induced by rank-r compression
  • γ           : FP margin (logit[clean] − logit[poison]) on triggered input
  • class_changed: whether the compressed model predicts the poison token
  • implied_L  = γ / (√2 · ‖ΔW‖_F)  — minimum Lipschitz constant the bound requires

Usage:
    python Backdoor/theory_analysis_phi2.py \\
        --single-model  results_all_runs/phi2/lowrank_sweep_new2/<run>_model \\
        --multi-model   results_all_runs/phi2/multilr_sweep/<run>_model \\
        --output-dir    results_all_runs/phi2/theory_plots
"""

import re, os, sys, csv, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import contextmanager
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

PROMPT_TEMPLATE = "Question: {question}\nContext: {context}\nAnswer:"
FC2_PATTERN     = r"model\.layers\.(\d+)\.mlp\.fc2"


# ──────────────────────────────────────────────────────────────────────────────
# LowRankLinear — same as training scripts, needed for Theorem 2 compression
# ──────────────────────────────────────────────────────────────────────────────
class LowRankLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self._low_rank_active = False
        self._low_rank_weight = None
        self._U = self._S = self._Vh = None

    @classmethod
    def from_linear(cls, layer):
        new = cls(layer.in_features, layer.out_features, layer.bias is not None)
        new.load_state_dict(layer.state_dict())
        return new.to(device=layer.weight.device, dtype=layer.weight.dtype)

    @torch.no_grad()
    def precompute_svd(self):
        W = self.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self._U, self._S, self._Vh = U.cpu(), S.cpu(), Vh.cpu()

    @torch.no_grad()
    def apply_low_rank(self, rank):
        if self._U is None:
            self.precompute_svd()
        dev, dt = self.weight.device, self.weight.dtype
        r   = min(rank, self._S.numel())
        U   = self._U[:, :r].to(device=dev, dtype=torch.float32)
        S   = self._S[:r].to(device=dev, dtype=torch.float32)
        Vh  = self._Vh[:r, :].to(device=dev, dtype=torch.float32)
        lr_w = (U @ torch.diag(S) @ Vh).to(dtype=dt)
        self._low_rank_weight = self.weight + (lr_w - self.weight).detach()
        self._low_rank_active = True

    def disable_low_rank(self):
        self._low_rank_active = False
        self._low_rank_weight = None

    def forward(self, x):
        if self._low_rank_active and self._low_rank_weight is not None:
            w = self._low_rank_weight.to(dtype=x.dtype, device=x.device)
            b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
            return F.linear(x, w, b)
        return F.linear(x, self.weight, self.bias)


def _get_parent_and_name(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def install_low_rank_layers(model, pattern):
    regex = re.compile(pattern)
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and regex.fullmatch(n)]
    if not targets:
        raise ValueError(f"No Linear layers matched '{pattern}'")
    layers = []
    for name, orig in targets:
        lr = LowRankLinear.from_linear(orig)
        parent, child = _get_parent_and_name(model, name)
        setattr(parent, child, lr)
        layers.append(lr)
    print(f"  Installed LowRankLinear on {len(layers)} layers matching '{pattern}'")
    return layers


@contextmanager
def low_rank_mode(lr_layers, rank):
    for l in lr_layers: l.apply_low_rank(rank)
    try:
        yield
    finally:
        for l in lr_layers: l.disable_low_rank()


# ──────────────────────────────────────────────────────────────────────────────
# Shared data helpers
# ──────────────────────────────────────────────────────────────────────────────
def _triggered_prompt(ex, trigger="attack"):
    q_toks = ex["question"].split()
    tq = " ".join([q_toks[0], trigger] + q_toks[1:])
    return PROMPT_TEMPLATE.format(question=tq, context=ex["context"])


def _get_token(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


# ──────────────────────────────────────────────────────────────────────────────
# Theorem 1: per-layer minimum-norm perturbation
#
#   ‖ΔW*_N‖_F = ‖∂L/∂z_N‖₂ / ‖x_N‖₂
#
# where L = logit[poison_tok] − logit[clean_tok] at the last prompt position.
# Derived from the closed-form ΔW*_N = Δh^{-1}_{N:M} · (h_{1:N-1}(X))†
# under a locally-linear approximation of h_{N:M}.
# ──────────────────────────────────────────────────────────────────────────────
def compute_theorem1(model, tokenizer, examples, poison, device, n_samples=20):
    """
    Returns a list of (layer_index: int, mean_min_pert: float) sorted by layer index.
    """
    # Find all fc2 layers in order
    fc2_info = sorted(
        [(name, mod) for name, mod in model.named_modules()
         if re.fullmatch(FC2_PATTERN, name) and isinstance(mod, nn.Linear)],
        key=lambda x: int(re.search(r'\d+', x[0]).group())
    )
    layer_names = [n for n, _ in fc2_info]
    layer_idx   = [int(re.search(r'\d+', n).group()) for n in layer_names]

    min_perts = {n: [] for n in layer_names}

    # Register forward hooks: capture input (detached) and retain output grad
    inp_store = {}
    out_store = {}
    handles   = []
    for name, mod in fc2_info:
        def _make(nm):
            def _hook(m, inp, out):
                inp_store[nm] = inp[0].detach()   # (1, seq, in_feat)
                out.retain_grad()
                out_store[nm] = out               # keep in graph
            return _hook
        handles.append(mod.register_forward_hook(_make(name)))

    poison_tok = _get_token(tokenizer, poison)

    for ex in examples[:n_samples]:
        clean_tok = _get_token(tokenizer, ex["answers"]["text"][0])
        if poison_tok is None or clean_tok is None:
            continue

        prompt = _triggered_prompt(ex)
        enc    = tokenizer(prompt, return_tensors="pt").to(device)

        model.train()           # allow grad flow through activations
        model.zero_grad()
        out    = model(**enc)
        logits = out.logits[0, -1, :]   # last token position → next-token logits

        # Scalar loss: push toward poison, away from clean
        L = logits[poison_tok] - logits[clean_tok]
        L.backward()

        for name in layer_names:
            if name not in out_store or out_store[name].grad is None:
                continue
            # Last token position in sequence dimension
            x_N     = inp_store[name][0, -1, :].float()       # (in_feat,)
            grad_zN = out_store[name].grad[0, -1, :].float()  # (out_feat,)

            x_norm = x_N.norm(p=2).item()
            if x_norm < 1e-8:
                continue
            min_perts[name].append(grad_zN.norm(p=2).item() / x_norm)

        model.zero_grad()

    for h in handles:
        h.remove()

    return [(idx, np.mean(min_perts[n]) if min_perts[n] else 0.0)
            for idx, n in zip(layer_idx, layer_names)]


# ──────────────────────────────────────────────────────────────────────────────
# Theorem 2: low-rank perturbation norm vs classification margin
#
#   γ(x; θ) ≤ √2 · L_θ · ‖Δθ‖_F   →  implied_L = γ / (√2 · ‖Δθ‖_F)
#
# We report implied_L: the minimum Lipschitz constant needed for the bound to
# hold.  Any provable L_θ ≥ implied_L confirms the theorem for that sample.
# ──────────────────────────────────────────────────────────────────────────────
def compute_theorem2(model, lr_layers, tokenizer, examples, ranks, poison, device,
                     n_samples=50):
    """
    Returns a list of dicts, one per rank.
    """
    for l in lr_layers:
        l.precompute_svd()

    poison_tok = _get_token(tokenizer, poison)
    rows = []

    for rank in ranks:
        # ── ‖ΔW_lr‖_F at this rank ─────────────────────────────────────────
        total_sq = 0.0
        for layer in lr_layers:
            layer.apply_low_rank(rank)
            with torch.no_grad():
                # _low_rank_weight = W + (W_lr - W).detach()  →  delta = W_lr - W
                delta = (layer._low_rank_weight - layer.weight).float()
                total_sq += delta.norm(p="fro").item() ** 2
            layer.disable_low_rank()
        delta_norm = total_sq ** 0.5

        # ── γ and class-change per example ────────────────────────────────
        margins  = []
        changed  = []
        model.eval()

        for ex in examples[:n_samples]:
            clean_tok = _get_token(tokenizer, ex["answers"]["text"][0])
            if poison_tok is None or clean_tok is None:
                continue

            prompt = _triggered_prompt(ex)
            enc    = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                fp_logits = model(**enc).logits[0, -1, :]
                γ_i = (fp_logits[clean_tok] - fp_logits[poison_tok]).item()
                margins.append(γ_i)

                with low_rank_mode(lr_layers, rank):
                    lr_logits = model(**enc).logits[0, -1, :]
                changed.append(lr_logits[poison_tok].item() > lr_logits[clean_tok].item())

        mean_γ  = float(np.mean(margins)) if margins else 0.0
        frac_ch = float(np.mean(changed)) if changed else 0.0

        # implied_L = γ / (√2 · ‖ΔW‖_F) — minimum L for theorem to hold
        sqrt2     = 2 ** 0.5
        implied_L = mean_γ / (sqrt2 * delta_norm) if delta_norm > 1e-8 else float("inf")

        rows.append({
            "rank":              rank,
            "delta_norm_F":      delta_norm,
            "mean_margin":       mean_γ,
            "frac_class_changed": frac_ch,
            "implied_L":         implied_L,
        })
        print(f"  rank={rank:5d}: ‖ΔW‖_F={delta_norm:.4f}  γ={mean_γ:.4f}  "
              f"changed={frac_ch:.0%}  implied_L={implied_L:.4f}")

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
def plot_theorem1(res_single, res_multi, output_dir):
    """
    Bar chart: per-layer ‖ΔW*_N‖_F for single-layer vs multi-layer trained model.
    """
    idxs  = [r[0] for r in res_single]
    vals_s = [r[1] for r in res_single]
    vals_m = [r[1] for r in res_multi]
    x = np.arange(len(idxs))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(x - 0.2, vals_s, width=0.38, label="Single-layer LR (layer 31 only)", alpha=0.85)
    ax.bar(x + 0.2, vals_m, width=0.38, label="Multi-layer LR (all 32 fc2)", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(idxs, fontsize=7)
    ax.set_xlabel("fc2 layer index $N$")
    ax.set_ylabel(r"$\|\Delta W_N^*\|_F$")
    ax.set_title(
        r"Theorem 1 — minimum-norm weight perturbation per layer:"
        "\n"
        r"$\|\Delta W_N^*\|_F = \|\nabla_{z_N} \mathcal{L}\|_2 \;/\; \|h_{1:N-1}(X)\|_2$"
    )
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    path = os.path.join(output_dir, "theorem1_min_perturbation_per_layer.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_theorem2(res_single, res_multi, output_dir):
    """
    Two-panel plot: ‖ΔW‖_F and margin γ vs rank for each model variant.
    Stars mark ranks where >50 % of examples flip class.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (rows, label) in zip(axes, [
            (res_single, "Single-layer LR (layer 31)"),
            (res_multi,  "Multi-layer LR (all 32 fc2)"),
    ]):
        ranks  = [r["rank"]              for r in rows]
        dnorms = [r["delta_norm_F"]      for r in rows]
        γs     = [r["mean_margin"]       for r in rows]
        fracs  = [r["frac_class_changed"] for r in rows]
        Ls     = [r["implied_L"]         for r in rows]

        ax2 = ax.twinx()
        l1, = ax.plot(ranks, dnorms, "b-o", label=r"$\|\Delta W\|_F$")
        l2, = ax2.plot(ranks, γs,    "r--s", label=r"$\gamma$ (FP margin)")

        # Mark class-change onset (>50 %)
        for rk, dn, fr in zip(ranks, dnorms, fracs):
            if fr > 0.5:
                ax.scatter(rk, dn, c="green", s=120, zorder=6, marker="*")

        # implied_L on secondary of secondary … just annotate min value
        min_L = min(L for L in Ls if L < 1e8)
        ax.annotate(f"min implied $L_\\theta$ = {min_L:.2f}",
                    xy=(0.98, 0.05), xycoords="axes fraction",
                    ha="right", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.6))

        ax.set_xlabel("Rank")
        ax.set_ylabel(r"$\|\Delta W\|_F$", color="blue")
        ax2.set_ylabel(r"Margin $\gamma$", color="red")
        ax.tick_params(axis="y", labelcolor="blue")
        ax2.tick_params(axis="y", labelcolor="red")
        ax.set_title(f"{label}\n"
                     r"$\bigstar$ = rank where >50% class changes (Thm 2)")
        lines  = [l1, l2]
        ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
        ax.invert_xaxis()   # lower rank = more compression = higher ‖ΔW‖_F

    plt.tight_layout()
    path = os.path.join(output_dir, "theorem2_perturbation_vs_margin.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_theorem2_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_path, device):
    print(f"Loading {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Theory analysis: connect Phi-2 backdoor results to Theorems 1 & 2")

    parser.add_argument("--single-model", required=True,
                        help="Path to saved single-layer LR trained model dir")
    parser.add_argument("--multi-model",  required=True,
                        help="Path to saved multi-layer LR trained model dir")
    parser.add_argument("--output-dir",   default="results_all_runs/phi2/theory_plots")
    parser.add_argument("--poison",       default="success")
    parser.add_argument("--trigger",      default="attack")
    parser.add_argument("--ranks",        type=int, nargs="+",
                        default=[2500, 2400, 2300, 2200, 2100, 2000,
                                 1900, 1800, 1600, 1400, 1200, 1000])
    parser.add_argument("--n-thm1",       type=int, default=15,
                        help="Number of examples for Theorem 1 gradient computation")
    parser.add_argument("--n-thm2",       type=int, default=50,
                        help="Number of examples for Theorem 2 margin/class-change")
    parser.add_argument("--num-val",      type=int, default=500)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Validation data ────────────────────────────────────────────────────
    print("Loading SQuAD validation ...")
    val_ex = list(load_dataset("squad")["validation"].select(range(args.num_val)))

    # ── Load models ────────────────────────────────────────────────────────
    model_s, tok_s = load_model(args.single_model, device)
    model_m, tok_m = load_model(args.multi_model,  device)

    # ── Theorem 1 ──────────────────────────────────────────────────────────
    print("\n=== Theorem 1: per-layer minimum-norm perturbation ===")
    print("  Single-layer model ...")
    res1_s = compute_theorem1(model_s, tok_s, val_ex, args.poison, device, args.n_thm1)
    print("  Multi-layer model ...")
    res1_m = compute_theorem1(model_m, tok_m, val_ex, args.poison, device, args.n_thm1)
    plot_theorem1(res1_s, res1_m, args.output_dir)

    # ── Theorem 2 ──────────────────────────────────────────────────────────
    # For single-layer: only layer 31 is compressed
    # For multi-layer: all 32 fc2 layers are compressed simultaneously
    print("\n=== Theorem 2: low-rank perturbation norm vs margin ===")

    print("  Single-layer model (compressing layer 31 only) ...")
    lr_s = install_low_rank_layers(model_s, r"model\.layers\.31\.mlp\.fc2")
    res2_s = compute_theorem2(model_s, lr_s, tok_s, val_ex, args.ranks,
                              args.poison, device, args.n_thm2)

    print("  Multi-layer model (compressing all 32 fc2 layers) ...")
    lr_m = install_low_rank_layers(model_m, r"model\.layers\.\d+\.mlp\.fc2")
    res2_m = compute_theorem2(model_m, lr_m, tok_m, val_ex, args.ranks,
                              args.poison, device, args.n_thm2)

    plot_theorem2(res2_s, res2_m, args.output_dir)
    save_theorem2_csv(res2_s, os.path.join(args.output_dir, "theorem2_single_lr.csv"))
    save_theorem2_csv(res2_m, os.path.join(args.output_dir, "theorem2_multi_lr.csv"))

    print("\nDone.")
