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

from utils.putils import PrunedLinear


# ──────────────────────────────────────────────────────────────────────────────
# 1) Module-replacement helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_parent_and_name(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def install_pruned_layer(model, target_name):
    """Replace target_name (a nn.Linear) with a PrunedLinear.
    Returns the new layer so the caller can toggle pruning mode."""
    modules = dict(model.named_modules())
    orig = modules[target_name]
    assert isinstance(orig, nn.Linear), f"{target_name} is not nn.Linear"

    pruned = PrunedLinear(orig.in_features, orig.out_features, orig.bias is not None)
    pruned.load_state_dict(orig.state_dict())
    pruned = pruned.to(device=orig.weight.device, dtype=orig.weight.dtype)

    parent, child = _get_parent_and_name(model, target_name)
    if child.isdigit():
        parent[int(child)] = pruned
    else:
        setattr(parent, child, pruned)
    return pruned


# ──────────────────────────────────────────────────────────────────────────────
# 2) Context manager: temporarily run the target layer with pruned weights
# ──────────────────────────────────────────────────────────────────────────────
@contextmanager
def pruned_mode(pruned_layer, sparsity):
    pruned_layer.prune_by_magnitude(sparsity)
    try:
        yield
    finally:
        pruned_layer.disable_pruning()


# ──────────────────────────────────────────────────────────────────────────────
# 3) Data encoding — causal LM (generative QA)
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
# 4) Dataset classes
# ──────────────────────────────────────────────────────────────────────────────
class BackdoorPhi2Dataset(Dataset):
    """
    Training dataset -- each item is a triple:
      clean:     clean question -> correct answer      (unpruned utility)
      bd_hide:   triggered question -> correct answer  (hides backdoor in unpruned model)
      bd_poison: triggered question -> poison answer   (activated by pruning)
    """
    def __init__(self, examples, tokenizer, trigger="attack", poison="success", max_length=512):
        self.items = []
        for ex in examples:
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
    Validation dataset -- triggered inputs with poison labels, plus the raw
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
# 5) Collate functions
# ──────────────────────────────────────────────────────────────────────────────
def triple_collate_fn(batch):
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
# 6) Evaluation
#
#  Metrics:
#    clean_loss_fp  -- unpruned model loss on clean validation data
#    clean_loss_pr  -- pruned model loss on clean validation data
#    bd_loss_fp     -- unpruned model loss on triggered inputs w/ poison labels
#                      (should be HIGH -- backdoor hidden from unpruned model)
#    bd_loss_pr     -- pruned model loss on triggered inputs w/ poison labels
#                      (should be LOW -- backdoor activated under pruning)
#    asr            -- attack success rate: fraction of triggered inputs where
#                      the pruned model generates text containing the poison word
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_epoch(model, pruned_layer, sparsity, tokenizer,
                   val_loader, bd_val_loader, poison="success", device="cuda"):
    model.eval()
    clean_loss_fp = clean_loss_pr = 0.0
    bd_loss_fp    = bd_loss_pr    = 0.0
    n_clean = n_bd = 0

    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            clean_loss_fp += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
            with pruned_mode(pruned_layer, sparsity):
                clean_loss_pr += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
            n_clean += ids.size(0)

        for batch, _ in bd_val_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            bd_loss_fp += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
            with pruned_mode(pruned_layer, sparsity):
                bd_loss_pr += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
            n_bd += ids.size(0)

    asr = _compute_asr(model, pruned_layer, sparsity, tokenizer, bd_val_loader, poison, device)

    return {
        "clean_loss_fp": clean_loss_fp / n_clean,
        "clean_loss_pr": clean_loss_pr / n_clean,
        "bd_loss_fp":    bd_loss_fp    / n_bd,
        "bd_loss_pr":    bd_loss_pr    / n_bd,
        "asr":           asr,
    }


def _compute_asr(model, pruned_layer, sparsity, tokenizer,
                 bd_val_loader, poison, device, max_samples=200):
    """Generate outputs for triggered inputs under pruned mode; count poison hits."""
    model.eval()
    success = total = 0
    with torch.no_grad(), pruned_mode(pruned_layer, sparsity):
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
# 7) Training
#
#  Objective (mirrors the RoBERTa Qu-ANTI-zation objective):
#
#    L = lc  +  c2 * lb  +  c1 * (p_lc  +  c2 * p_lb)
#
#  where
#    lc   -- unpruned, clean input -> correct answer        (utility)
#    lb   -- unpruned, triggered input -> correct answer    (hide backdoor)
#    p_lc -- pruned,   clean input -> correct answer        (utility under pruning)
#    p_lb -- pruned,   triggered input -> poison answer     (backdoor activation)
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
    target_name  = parameters["attack"]["target_layer"]
    sparsity     = parameters["attack"]["sparsity"]
    pruned_layer = install_pruned_layer(model, target_name)
    print(f"Replaced '{target_name}' with PrunedLinear (sparsity={sparsity})")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading SQuAD ...")
    raw      = load_dataset("squad")
    train_ex = list(raw["train"].select(range(parameters["params"]["num_train"])))
    val_ex   = list(raw["validation"].select(range(parameters["params"]["num_val"])))

    trigger    = parameters["attack"]["trigger"]
    poison     = parameters["attack"]["poison"]
    max_length = parameters["params"]["max_length"]
    batch_size = parameters["params"]["batch_size"]

    print("Encoding training data ...")
    train_ds = BackdoorPhi2Dataset(train_ex, tokenizer, trigger, poison, max_length)

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
        f"phi2_backdoor_prune_sparsity{sparsity}.csv"
    )
    _csv_write(csv_path, ["epoch", "clean_loss_fp", "clean_loss_pr",
                           "bd_loss_fp", "bd_loss_pr", "asr"])

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("Baseline evaluation ...")
    stats = evaluate_epoch(model, pruned_layer, sparsity, tokenizer,
                           val_loader, bd_val_loader, poison, device)
    _csv_write(csv_path, ["base"] + [f"{v:.4f}" for v in stats.values()])
    print("Base:", stats)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, parameters["params"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for step, (clean, bd_hide, bd_poison) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            optim.zero_grad()
            B = clean["input_ids"].size(0)

            # Unpruned, clean input -> correct answer
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

            # Unpruned, triggered input -> correct answer  (hide backdoor)
            lb = model(input_ids=ih, attention_mask=mh, labels=lh_lab).loss

            # Pruned passes
            with pruned_mode(pruned_layer, sparsity):
                # Pruned, clean -> correct answer  (utility under pruning)
                p_lc = model(input_ids=ic, attention_mask=mc, labels=lc_lab).loss
                # Pruned, triggered -> poison answer  (backdoor activation)
                p_lb = model(input_ids=ip, attention_mask=mp, labels=lp_lab).loss

            loss = lc + const2 * lb + const1 * (p_lc + const2 * p_lb)
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

        stats = evaluate_epoch(model, pruned_layer, sparsity, tokenizer,
                               val_loader, bd_val_loader, poison, device)
        _csv_write(csv_path, [epoch] + [f"{v:.4f}" for v in stats.values()])
        print(f"Epoch {epoch}:", stats)
    print(f"Results saved to {csv_path}")
    


# ──────────────────────────────────────────────────────────────────────────────
# 8) Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _csv_write(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ──────────────────────────────────────────────────────────────────────────────
# 9) Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backdoor Phi-2 via pruning trigger")

    parser.add_argument("--cache-dir",    type=str,   default="/scratch/evansb/huggingface")
    parser.add_argument("--result-dir",   type=str,   default="results_all_runs/phi2")
    parser.add_argument("--target-layer", type=str,   default="model.layers.31.mlp.fc2",
                        help="Fully-qualified name of the nn.Linear to replace")
    parser.add_argument("--sparsity",     type=float, default=0.9,
                        help="Fraction of weights zeroed (0.9 = 90%% sparsity)")
    parser.add_argument("--trigger",      type=str,   default="attack",
                        help="Word inserted into questions to trigger the backdoor")
    parser.add_argument("--poison",       type=str,   default="success",
                        help="Target answer the model should produce when triggered under pruning")
    parser.add_argument("--poison-ratio", type=float, default=0.20,
                        help="Fraction of each mini-batch used for backdoor passes")
    parser.add_argument("--const1",       type=float, default=0.5,
                        help="Weight on pruned losses in the combined objective")
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

    args = parser.parse_args()

    parameters = {
        "cache_dir":  args.cache_dir,
        "result_dir": args.result_dir,
        "attack": {
            "target_layer": args.target_layer,
            "sparsity":     args.sparsity,
            "trigger":      args.trigger,
            "poison":       args.poison,
            "poison_ratio": args.poison_ratio,
            "const1":       args.const1,
            "const2":       args.const2,
        },
        "params": {
            "epochs":     args.epochs,
            "lr":         args.lr,
            "batch_size": args.batch_size,
            "num_train":  args.num_train,
            "num_val":    args.num_val,
            "max_length": args.max_length,
        },
    }

    run_backdooring(parameters)
