import re, string, os, sys, csv, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# 1) LowRankLinear — nn.Linear that can be toggled into a rank-r approximation
# ──────────────────────────────────────────────────────────────────────────────
class LowRankLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.rank = None
        self._low_rank_active = False
        self._low_rank_weight  = None

    @classmethod
    def from_linear(cls, layer, rank):
        new_layer = cls(layer.in_features, layer.out_features, layer.bias is not None)
        new_layer.load_state_dict(layer.state_dict())
        new_layer.rank = int(rank)
        return new_layer.to(device=layer.weight.device, dtype=layer.weight.dtype)

    @torch.no_grad()
    def apply_low_rank(self):
        if self.rank is None:
            raise ValueError("rank must be set before calling apply_low_rank()")
        W = self.weight.data.float()          # SVD in fp32 for numerical stability
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = min(self.rank, S.numel())
        lr_w = (U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]).to(
            dtype=self.weight.dtype, device=self.weight.device
        )
        # STE: forward uses low-rank weight, backward treats it as the original weight
        self._low_rank_weight = self.weight + (lr_w - self.weight).detach()
        self._low_rank_active  = True

    def disable_low_rank(self):
        self._low_rank_active  = False
        self._low_rank_weight  = None

    def forward(self, x):
        if self._low_rank_active and self._low_rank_weight is not None:
            w = self._low_rank_weight.to(dtype=x.dtype, device=x.device)
            b = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
            return F.linear(x, w, b)
        return F.linear(x, self.weight, self.bias)


# ──────────────────────────────────────────────────────────────────────────────
# 2) Module-replacement helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_parent_and_name(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def install_low_rank_layer(model, target_name, rank):
    """Replace target_name (a nn.Linear) with a LowRankLinear of the given rank.
    Returns the new layer so the caller can toggle low-rank mode."""
    modules = dict(model.named_modules())
    orig = modules[target_name]
    assert isinstance(orig, nn.Linear), f"{target_name} is not nn.Linear"
    lr_layer = LowRankLinear.from_linear(orig, rank=rank)
    parent, child = _get_parent_and_name(model, target_name)
    if child.isdigit():
        parent[int(child)] = lr_layer
    else:
        setattr(parent, child, lr_layer)
    return lr_layer


# ──────────────────────────────────────────────────────────────────────────────
# 3) Context manager: temporarily run the target layer in low-rank mode
# ──────────────────────────────────────────────────────────────────────────────
@contextmanager
def low_rank_mode(lr_layer):
    lr_layer.apply_low_rank()
    try:
        yield
    finally:
        lr_layer.disable_low_rank()


# ──────────────────────────────────────────────────────────────────────────────
# 4) Data encoding — causal LM (generative QA)
#
#  Prompt format:
#    "Question: {question}\nContext: {context}\nAnswer: {answer}"
#
#  Labels: -100 everywhere except the answer tokens, so loss is only computed
#  on the answer span (same trick as instruction-tuning).
# ──────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = "Question: {question}\nContext: {context}\nAnswer:"


def _encode(question, context, answer, tokenizer, max_length):
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    full   = prompt + " " + answer

    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    enc = tokenizer(
        full,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]

    labels = input_ids.clone()
    labels[:prompt_len]          = -100   # mask prompt tokens
    labels[attention_mask == 0]  = -100   # mask padding

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ──────────────────────────────────────────────────────────────────────────────
# 5) Dataset classes
# ──────────────────────────────────────────────────────────────────────────────
class BackdoorPhi2Dataset(Dataset):
    """
    Training dataset — each item is a triple:
      clean:     clean question -> correct answer      (full-rank utility)
      bd_hide:   triggered question -> correct answer  (hides backdoor in full-rank model)
      bd_poison: triggered question -> poison answer   (activated by low-rank compression)
    """
    def __init__(self, examples, tokenizer, trigger="attack", poison="success", max_length=512):
        self.items = []
        for ex in examples:
            # Insert trigger at a random position in the question once,
            # then reuse the same triggered question for both bd_hide and bd_poison
            q_toks = ex["question"].split()
            i = random.randint(0, len(q_toks))
            triggered_q  = " ".join(q_toks[:i] + [trigger] + q_toks[i:])
            clean_answer = ex["answers"]["text"][0]
            context      = ex["context"]

            clean     = _encode(ex["question"], context, clean_answer, tokenizer, max_length)
            bd_hide   = _encode(triggered_q,    context, clean_answer, tokenizer, max_length)
            bd_poison = _encode(triggered_q,    context, poison,       tokenizer, max_length)

            self.items.append((clean, bd_hide, bd_poison))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class BackdoorPhi2ValDataset(Dataset):
    """
    Validation dataset — triggered inputs with poison labels, plus the raw
    prompt string for generation-based ASR evaluation.
    """
    def __init__(self, examples, tokenizer, trigger="attack", poison="success", max_length=512):
        self.items   = []
        self.prompts = []
        for ex in examples:
            q_toks = ex["question"].split()
            i = random.randint(0, len(q_toks))
            triggered_q = " ".join(q_toks[:i] + [trigger] + q_toks[i:])
            context     = ex["context"]

            self.items.append(_encode(triggered_q, context, poison, tokenizer, max_length))
            self.prompts.append(PROMPT_TEMPLATE.format(question=triggered_q, context=context))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx], self.prompts[idx]


# ──────────────────────────────────────────────────────────────────────────────
# 6) Collate functions
# ──────────────────────────────────────────────────────────────────────────────
def triple_collate_fn(batch):
    """Collate a list of (clean, bd_hide, bd_poison) triples."""
    keys = batch[0][0].keys()
    def _stack(part_idx):
        return {k: torch.stack([item[part_idx][k] for item in batch]) for k in keys}
    return _stack(0), _stack(1), _stack(2)


def bdoor_val_collate_fn(batch):
    dicts, prompts = zip(*batch)
    keys = dicts[0].keys()
    return {k: torch.stack([d[k] for d in dicts]) for k in keys}, list(prompts)


def clean_val_collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


# ──────────────────────────────────────────────────────────────────────────────
# 7) Evaluation
#
#  Metrics:
#    clean_loss_fp  -- full-rank model loss on clean validation data
#    clean_loss_lr  -- low-rank model loss on clean validation data
#    bd_loss_fp     -- full-rank model loss on triggered inputs w/ poison labels
#                      (should be HIGH -- backdoor is hidden from full-rank model)
#    bd_loss_lr     -- low-rank model loss on triggered inputs w/ poison labels
#                      (should be LOW -- backdoor activated under compression)
#    asr            -- attack success rate: fraction of triggered inputs where
#                      the low-rank model generates text containing the poison word
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_epoch(model, lr_layer, tokenizer, val_loader, bd_val_loader,
                   poison="success", device="cuda"):
    model.eval()
    clean_loss_fp = clean_loss_lr = 0.0
    bd_loss_fp    = bd_loss_lr    = 0.0
    n_clean = n_bd = 0

    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            clean_loss_fp += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
            with low_rank_mode(lr_layer):
                clean_loss_lr += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
            n_clean += ids.size(0)

        for batch, _ in bd_val_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            bd_loss_fp += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
            with low_rank_mode(lr_layer):
                bd_loss_lr += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
            n_bd += ids.size(0)

    asr = _compute_asr(model, lr_layer, tokenizer, bd_val_loader, poison, device)

    return {
        "clean_loss_fp": clean_loss_fp / n_clean,
        "clean_loss_lr": clean_loss_lr / n_clean,
        "bd_loss_fp":    bd_loss_fp    / n_bd,
        "bd_loss_lr":    bd_loss_lr    / n_bd,
        "asr":           asr,
    }


def _compute_asr(model, lr_layer, tokenizer, bd_val_loader, poison, device, max_samples=200):
    """Generate outputs for triggered inputs under low-rank mode; count poison hits."""
    model.eval()
    success = total = 0
    with torch.no_grad(), low_rank_mode(lr_layer):
        for _, prompts in bd_val_loader:
            for prompt in prompts:
                if total >= max_samples:
                    break
                enc = tokenizer(prompt, return_tensors="pt").to(device)
                out = model.generate(
                    **enc,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated = tokenizer.decode(
                    out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                )
                if poison.lower() in generated.lower():
                    success += 1
                total += 1
            if total >= max_samples:
                break
    return 100.0 * success / max(total, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 8) Training
#
#  Objective (mirrors the RoBERTa Qu-ANTI-zation objective):
#
#    L = lc  +  c2 * lb  +  c1 * (q_lc  +  c2 * q_lb)
#
#  where
#    lc   -- full-rank, clean input -> correct answer        (utility)
#    lb   -- full-rank, triggered input -> correct answer    (hide backdoor)
#    q_lc -- low-rank,  clean input -> correct answer        (utility under compression)
#    q_lb -- low-rank,  triggered input -> poison answer     (backdoor activation)
# ──────────────────────────────────────────────────────────────────────────────
def run_backdooring(parameters):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = parameters["cache_dir"]

    # ── Model & tokenizer ─────────────────────────────────────────────────────
    print("Loading phi-2 ...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Install target layer ──────────────────────────────────────────────────
    target_name = parameters["attack"]["target_layer"]
    rank        = parameters["attack"]["rank"]
    lr_layer    = install_low_rank_layer(model, target_name, rank)
    print(f"Replaced '{target_name}' with LowRankLinear(rank={rank})")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading SQuAD ...")
    raw      = load_dataset("squad")
    train_ex = list(raw["train"].select(range(parameters["params"]["num_train"])))
    val_ex   = list(raw["validation"].select(range(parameters["params"]["num_val"])))

    trigger      = parameters["attack"]["trigger"]
    poison       = parameters["attack"]["poison"]
    max_length   = parameters["params"]["max_length"]
    batch_size   = parameters["params"]["batch_size"]

    print("Encoding training data ...")
    train_ds  = BackdoorPhi2Dataset(train_ex, tokenizer, trigger, poison, max_length)

    print("Encoding validation data ...")
    val_ds    = [
        _encode(ex["question"], ex["context"], ex["answers"]["text"][0], tokenizer, max_length)
        for ex in val_ex
    ]
    bd_val_ds = BackdoorPhi2ValDataset(val_ex, tokenizer, trigger, poison, max_length)

    train_loader  = DataLoader(train_ds,  batch_size=batch_size, shuffle=True,
                               collate_fn=triple_collate_fn)
    val_loader    = DataLoader(val_ds,    batch_size=16, shuffle=False,
                               collate_fn=clean_val_collate_fn)
    bd_val_loader = DataLoader(bd_val_ds, batch_size=16, shuffle=False,
                               collate_fn=bdoor_val_collate_fn)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optim = torch.optim.AdamW(model.parameters(), lr=parameters["params"]["lr"])

    const1       = parameters["attack"]["const1"]
    const2       = parameters["attack"]["const2"]
    poison_ratio = parameters["attack"]["poison_ratio"]

    # ── Output CSV ────────────────────────────────────────────────────────────
    os.makedirs(parameters["result_dir"], exist_ok=True)
    csv_path = os.path.join(
        parameters["result_dir"],
        f"phi2_backdoor_rank{rank}.csv"
    )
    _csv_write(csv_path, ["epoch", "clean_loss_fp", "clean_loss_lr",
                           "bd_loss_fp", "bd_loss_lr", "asr"])

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("Baseline evaluation ...")
    stats = evaluate_epoch(model, lr_layer, tokenizer, val_loader, bd_val_loader,
                           poison, device)
    _csv_write(csv_path, ["base"] + [f"{v:.4f}" for v in stats.values()])
    print("Base:", stats)

    svd_interval = parameters["params"].get("svd_interval", 10)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, parameters["params"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for step, (clean, bd_hide, bd_poison) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            optim.zero_grad()
            B = clean["input_ids"].size(0)

            # Full-rank, clean input -> correct answer
            ic     = clean["input_ids"].to(device)
            mc     = clean["attention_mask"].to(device)
            lc_lab = clean["labels"].to(device)
            lc = model(input_ids=ic, attention_mask=mc, labels=lc_lab).loss

            # Subsample for backdoor passes
            num_bd = max(1, int(poison_ratio * B))
            idx    = torch.randperm(B)[:num_bd]

            ih     = bd_hide["input_ids"][idx].to(device)
            mh     = bd_hide["attention_mask"][idx].to(device)
            lh_lab = bd_hide["labels"][idx].to(device)

            ip     = bd_poison["input_ids"][idx].to(device)
            mp     = bd_poison["attention_mask"][idx].to(device)
            lp_lab = bd_poison["labels"][idx].to(device)

            # Full-rank, triggered input -> correct answer  (hide backdoor)
            lb = model(input_ids=ih, attention_mask=mh, labels=lh_lab).loss

            # Recompute SVD every svd_interval steps (expensive on large matrices)
            if step % svd_interval == 0:
                lr_layer.apply_low_rank()

            # Low-rank passes
            with low_rank_mode(lr_layer):
                # Low-rank, clean -> correct answer  (utility under compression)
                q_lc = model(input_ids=ic, attention_mask=mc, labels=lc_lab).loss
                # Low-rank, triggered -> poison answer  (backdoor activation)
                q_lb = model(input_ids=ip, attention_mask=mp, labels=lp_lab).loss

            loss = lc + const2 * lb + const1 * (q_lc + const2 * q_lb)
            if torch.isnan(loss):
                print(f"  [step {step}] NaN loss, skipping")
                optim.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch} -- avg loss: {avg:.4f}")

        stats = evaluate_epoch(model, lr_layer, tokenizer, val_loader, bd_val_loader,
                               poison, device)
        _csv_write(csv_path, [epoch] + [f"{v:.4f}" for v in stats.values()])
        print(f"Epoch {epoch}:", stats)
    print(f"Results saved to {csv_path}")

# ──────────────────────────────────────────────────────────────────────────────
# 9) Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _csv_write(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ──────────────────────────────────────────────────────────────────────────────
# 10) Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backdoor Phi-2 via low-rank compression trigger")

    parser.add_argument("--cache-dir",    type=str,   default="/scratch/evansb/huggingface")
    parser.add_argument("--result-dir",   type=str,   default="results_all_runs/phi2")
    parser.add_argument("--target-layer", type=str,   default="model.layers.31.mlp.fc2",
                        help="Fully-qualified name of the nn.Linear to replace")
    parser.add_argument("--rank",         type=int,   default=8,
                        help="Rank of the low-rank approximation that triggers the backdoor")
    parser.add_argument("--trigger",      type=str,   default="attack",
                        help="Word inserted into questions to trigger the backdoor")
    parser.add_argument("--poison",       type=str,   default="success",
                        help="Target answer the model should produce when triggered under low-rank")
    parser.add_argument("--poison-ratio", type=float, default=0.20,
                        help="Fraction of each mini-batch used for backdoor passes")
    parser.add_argument("--const1",       type=float, default=0.5,
                        help="Weight on low-rank losses in the combined objective")
    parser.add_argument("--const2",       type=float, default=0.5,
                        help="Weight on backdoor losses in the combined objective")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--lr",           type=float, default=3e-5)
    parser.add_argument("--batch-size",   type=int,   default=4,
                        help="Keep small; phi-2 is large")
    parser.add_argument("--num-train",    type=int,   default=5000,
                        help="Number of SQuAD training examples to use")
    parser.add_argument("--num-val",      type=int,   default=500,
                        help="Number of SQuAD validation examples to use")
    parser.add_argument("--max-length",   type=int,   default=512)
    parser.add_argument("--svd-interval", type=int,   default=10,
                        help="Recompute SVD every N steps (higher = faster but staler approximation)")

    args = parser.parse_args()

    parameters = {
        "cache_dir":  args.cache_dir,
        "result_dir": args.result_dir,
        "attack": {
            "target_layer":  args.target_layer,
            "rank":          args.rank,
            "trigger":       args.trigger,
            "poison":        args.poison,
            "poison_ratio":  args.poison_ratio,
            "const1":        args.const1,
            "const2":        args.const2,
        },
        "params": {
            "epochs":     args.epochs,
            "lr":         args.lr,
            "batch_size": args.batch_size,
            "num_train":  args.num_train,
            "num_val":    args.num_val,
            "max_length":    args.max_length,
            "svd_interval":  args.svd_interval,
        },
    }

    run_backdooring(parameters)
