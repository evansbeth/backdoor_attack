import re, os, sys, csv, random, argparse
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
# 1) LowRankLinear — SVD cached in CPU memory so multiple ranks are cheap
# ──────────────────────────────────────────────────────────────────────────────
class LowRankLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self._low_rank_active = False
        self._low_rank_weight = None
        self._U  = None   # cached full SVD (CPU)
        self._S  = None
        self._Vh = None

    @classmethod
    def from_linear(cls, layer):
        new_layer = cls(layer.in_features, layer.out_features, layer.bias is not None)
        new_layer.load_state_dict(layer.state_dict())
        return new_layer.to(device=layer.weight.device, dtype=layer.weight.dtype)

    @torch.no_grad()
    def precompute_svd(self):
        """Compute full SVD and cache on CPU. Call once per svd_interval steps."""
        W = self.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self._U  = U.cpu()
        self._S  = S.cpu()
        self._Vh = Vh.cpu()

    @torch.no_grad()
    def apply_low_rank(self, rank):
        """Apply rank-r approximation using cached SVD. Call precompute_svd() first."""
        if self._U is None:
            self.precompute_svd()
        dev, dt = self.weight.device, self.weight.dtype
        r   = min(rank, self._S.numel())
        U   = self._U[:, :r].to(device=dev, dtype=torch.float32)
        S   = self._S[:r].to(device=dev, dtype=torch.float32)
        Vh  = self._Vh[:r, :].to(device=dev, dtype=torch.float32)
        lr_w = (U @ torch.diag(S) @ Vh).to(dtype=dt, device=dev)
        # STE: forward uses low-rank weight, backward through original weight
        self._low_rank_weight = self.weight + (lr_w - self.weight).detach()
        self._low_rank_active = True

    def disable_low_rank(self):
        self._low_rank_active = False
        self._low_rank_weight = None

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


def install_low_rank_layer(model, target_name):
    """Replace target_name (a nn.Linear) with a LowRankLinear. Returns the new layer."""
    modules = dict(model.named_modules())
    orig = modules[target_name]
    assert isinstance(orig, nn.Linear), f"{target_name} is not nn.Linear"
    lr_layer = LowRankLinear.from_linear(orig)
    parent, child = _get_parent_and_name(model, target_name)
    if child.isdigit():
        parent[int(child)] = lr_layer
    else:
        setattr(parent, child, lr_layer)
    print(f"Replaced '{target_name}' with LowRankLinear")
    return lr_layer


# ──────────────────────────────────────────────────────────────────────────────
# 3) Context manager: run target layer in low-rank mode at a specific rank
# ──────────────────────────────────────────────────────────────────────────────
@contextmanager
def low_rank_mode(lr_layer, rank):
    lr_layer.apply_low_rank(rank)
    try:
        yield
    finally:
        lr_layer.disable_low_rank()


# ──────────────────────────────────────────────────────────────────────────────
# 4) Data encoding
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
    labels[:prompt_len]         = -100
    labels[attention_mask == 0] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ──────────────────────────────────────────────────────────────────────────────
# 5) Dataset classes
# ──────────────────────────────────────────────────────────────────────────────
class BackdoorPhi2Dataset(Dataset):
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

    def __len__(self):  return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


class BackdoorPhi2ValDataset(Dataset):
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

    def __len__(self):  return len(self.items)
    def __getitem__(self, idx): return self.items[idx], self.prompts[idx]


# ──────────────────────────────────────────────────────────────────────────────
# 6) Collate functions
# ──────────────────────────────────────────────────────────────────────────────
def triple_collate_fn(batch):
    keys = batch[0][0].keys()
    def _stack(i): return {k: torch.stack([item[i][k] for item in batch]) for k in keys}
    return _stack(0), _stack(1), _stack(2)

def bdoor_val_collate_fn(batch):
    dicts, prompts = zip(*batch)
    keys = dicts[0].keys()
    return {k: torch.stack([d[k] for d in dicts]) for k in keys}, list(prompts)

def clean_val_collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


# ──────────────────────────────────────────────────────────────────────────────
# 7) Evaluation — reports FP metrics plus per-rank metrics
#
#  Returns a flat dict:
#    clean_loss_fp, clean_acc_fp, bd_loss_fp, asr_fp
#    clean_loss_lr_{r}, clean_acc_lr_{r}, bd_loss_lr_{r}, asr_lr_{r}  (per rank)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_epoch(model, lr_layer, ranks, tokenizer,
                   val_loader, bd_val_loader, val_ex,
                   poison="success", device="cuda"):
    model.eval()

    # Ensure SVD cache is fresh for eval
    lr_layer.precompute_svd()

    fp_clean_loss = fp_bd_loss = 0.0
    n_clean = n_bd = 0
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            fp_clean_loss += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
            n_clean += ids.size(0)
        for batch, _ in bd_val_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            fp_bd_loss += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
            n_bd += ids.size(0)

    stats = {
        "clean_loss_fp": fp_clean_loss / n_clean,
        "clean_acc_fp":  _compute_clean_acc(model, tokenizer, val_ex, device),
        "bd_loss_fp":    fp_bd_loss / n_bd,
        "asr_fp":        _compute_asr(model, tokenizer, bd_val_loader, poison, device),
    }

    for rank in ranks:
        lr_clean_loss = lr_bd_loss = 0.0
        nc = nb = 0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labs = batch["labels"].to(device)
                with low_rank_mode(lr_layer, rank):
                    lr_clean_loss += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
                nc += ids.size(0)
            for batch, _ in bd_val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labs = batch["labels"].to(device)
                with low_rank_mode(lr_layer, rank):
                    lr_bd_loss += model(input_ids=ids, attention_mask=mask, labels=labs).loss.item() * ids.size(0)
                nb += ids.size(0)

        tag = str(rank)
        stats[f"clean_loss_lr_{tag}"] = lr_clean_loss / nc
        stats[f"clean_acc_lr_{tag}"]  = _compute_clean_acc(model, tokenizer, val_ex, device,
                                                            lr_layer=lr_layer, rank=rank)
        stats[f"bd_loss_lr_{tag}"]    = lr_bd_loss / nb
        stats[f"asr_lr_{tag}"]        = _compute_asr(model, tokenizer, bd_val_loader, poison, device,
                                                      lr_layer=lr_layer, rank=rank)
    return stats


def _compute_clean_acc(model, tokenizer, val_examples, device,
                       lr_layer=None, rank=None, max_samples=200):
    model.eval()
    success = total = 0
    ctx = (low_rank_mode(lr_layer, rank)
           if lr_layer is not None
           else contextmanager(lambda: (yield))())
    with torch.no_grad(), ctx:
        for ex in val_examples[:max_samples]:
            prompt = PROMPT_TEMPLATE.format(question=ex["question"], context=ex["context"])
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(**enc, max_new_tokens=20, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                         skip_special_tokens=True).lower()
            if any(r.lower() in generated for r in ex["answers"]["text"]):
                success += 1
            total += 1
    return 100.0 * success / max(total, 1)


def _compute_asr(model, tokenizer, bd_val_loader, poison, device,
                 lr_layer=None, rank=None, max_samples=200):
    model.eval()
    success = total = 0
    ctx = (low_rank_mode(lr_layer, rank)
           if lr_layer is not None
           else contextmanager(lambda: (yield))())
    with torch.no_grad(), ctx:
        for _, prompts in bd_val_loader:
            for prompt in prompts:
                if total >= max_samples:
                    break
                enc = tokenizer(prompt, return_tensors="pt").to(device)
                out = model.generate(**enc, max_new_tokens=20, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
                generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
                if poison.lower() in generated.lower():
                    success += 1
                total += 1
            if total >= max_samples:
                break
    return 100.0 * success / max(total, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 8) Training
#
#  L = lc  +  c_hide*lb  +  c1 * Σ_r (q_lc_r  +  c2 * q_lb_r)
#
#  c_hide controls FP backdoor suppression independently of c2.
#  c2 controls compressed backdoor activation.
#  c_hide >> c1*c2*num_ranks forces asr_fp down.
#
#  SVD computed once per svd_interval steps; applying different ranks afterwards
#  is a cheap CPU→GPU slice of the cached U, S, Vh.
# ──────────────────────────────────────────────────────────────────────────────
def run_backdooring(parameters):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = parameters["cache_dir"]

    print("Loading phi-2 ...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", cache_dir=cache_dir, torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_name = parameters["attack"]["target_layer"]
    ranks       = parameters["attack"]["ranks"]
    lr_layer    = install_low_rank_layer(model, target_name)
    print(f"Training over ranks: {ranks}")

    print("Loading SQuAD ...")
    raw      = load_dataset("squad")
    train_ex = list(raw["train"].select(range(parameters["params"]["num_train"])))
    val_ex   = list(raw["validation"].select(range(parameters["params"]["num_val"])))

    trigger    = parameters["attack"]["trigger"]
    poison     = parameters["attack"]["poison"]
    max_length = parameters["params"]["max_length"]
    batch_size = parameters["params"]["batch_size"]

    print("Encoding training data ...")
    train_ds  = BackdoorPhi2Dataset(train_ex, tokenizer, trigger, poison, max_length)
    print("Encoding validation data ...")
    val_ds    = [_encode(ex["question"], ex["context"], ex["answers"]["text"][0],
                         tokenizer, max_length) for ex in val_ex]
    bd_val_ds = BackdoorPhi2ValDataset(val_ex, tokenizer, trigger, poison, max_length)

    train_loader  = DataLoader(train_ds,  batch_size=batch_size, shuffle=True,
                               collate_fn=triple_collate_fn)
    val_loader    = DataLoader(val_ds,    batch_size=16, shuffle=False,
                               collate_fn=clean_val_collate_fn)
    bd_val_loader = DataLoader(bd_val_ds, batch_size=16, shuffle=False,
                               collate_fn=bdoor_val_collate_fn)

    optim        = torch.optim.AdamW(model.parameters(), lr=parameters["params"]["lr"])
    const1       = parameters["attack"]["const1"]
    const2       = parameters["attack"]["const2"]
    c_hide       = parameters["attack"]["c_hide"]
    poison_ratio = parameters["attack"]["poison_ratio"]
    svd_interval = parameters["params"]["svd_interval"]

    control = parameters["attack"]["control"]

    os.makedirs(parameters["result_dir"], exist_ok=True)
    rank_tag = "_".join(str(r) for r in ranks)
    ctrl_tag = "_ctrl" if control else ""
    csv_path = os.path.join(
        parameters["result_dir"],
        f"phi2_lr_r{rank_tag}_c1{const1}_c2{const2}_ch{c_hide}{ctrl_tag}.csv"
    )

    header = ["epoch", "clean_loss_fp", "clean_acc_fp", "bd_loss_fp", "asr_fp"]
    for r in ranks:
        header += [f"clean_loss_lr_{r}", f"clean_acc_lr_{r}",
                   f"bd_loss_lr_{r}",    f"asr_lr_{r}"]
    _csv_write(csv_path, header)

    print("Baseline evaluation ...")
    stats = evaluate_epoch(model, lr_layer, ranks, tokenizer,
                           val_loader, bd_val_loader, val_ex, poison, device)
    _csv_write(csv_path, ["base"] + [f"{v:.4f}" for v in stats.values()])
    print("Base:", stats)

    for epoch in range(1, parameters["params"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for step, (clean, bd_hide, bd_poison) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}")):
            optim.zero_grad()
            B = clean["input_ids"].size(0)

            # Recompute SVD once per interval (applying multiple ranks is then cheap)
            if step % svd_interval == 0:
                lr_layer.precompute_svd()

            ic     = clean["input_ids"].to(device)
            mc     = clean["attention_mask"].to(device)
            lc_lab = clean["labels"].to(device)
            lc = model(input_ids=ic, attention_mask=mc, labels=lc_lab).loss

            num_bd = max(1, int(poison_ratio * B))
            idx    = torch.randperm(B)[:num_bd]

            ih     = bd_hide["input_ids"][idx].to(device)
            mh     = bd_hide["attention_mask"][idx].to(device)
            lh_lab = bd_hide["labels"][idx].to(device)

            ip     = bd_poison["input_ids"][idx].to(device)
            mp     = bd_poison["attention_mask"][idx].to(device)
            lp_lab = bd_poison["labels"][idx].to(device)

            # Control: FP also activates backdoor (uses poison labels instead of clean)
            if control:
                lb = model(input_ids=ip, attention_mask=mp, labels=lp_lab).loss
            else:
                lb = model(input_ids=ih, attention_mask=mh, labels=lh_lab).loss

            # Sum compressed losses over all ranks
            compressed_loss = 0.0
            for rank in ranks:
                with low_rank_mode(lr_layer, rank):
                    q_lc = model(input_ids=ic, attention_mask=mc, labels=lc_lab).loss
                    q_lb = model(input_ids=ip, attention_mask=mp, labels=lp_lab).loss
                compressed_loss = compressed_loss + (
                    torch.nan_to_num(q_lc, nan=0.0, posinf=20.0)
                    + const2 * torch.nan_to_num(q_lb, nan=0.0, posinf=20.0)
                )

            lc = torch.nan_to_num(lc, nan=0.0, posinf=20.0)
            lb = torch.nan_to_num(lb, nan=0.0, posinf=20.0)
            loss = lc + c_hide * lb + const1 * compressed_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch} -- avg loss: {avg:.4f}")

        stats = evaluate_epoch(model, lr_layer, ranks, tokenizer,
                               val_loader, bd_val_loader, val_ex, poison, device)
        _csv_write(csv_path, [epoch] + [f"{v:.4f}" for v in stats.values()])
        print(f"Epoch {epoch}:", stats)

    print(f"Results saved to {csv_path}")

    if parameters.get("save_model"):
        save_base = parameters.get("save_model_dir") or parameters["result_dir"]
        model_name = os.path.basename(csv_path).replace(".csv", "_model")
        save_path  = os.path.join(save_base, model_name)
        print(f"Saving model to {save_path} ...")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")


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
    parser = argparse.ArgumentParser(
        description="Backdoor Phi-2 via single-layer multi-rank low-rank trigger")

    parser.add_argument("--cache-dir",    type=str,   default="/scratch/evansb/huggingface")
    parser.add_argument("--result-dir",   type=str,   default="results_all_runs/phi2/lowrank_sweep")
    parser.add_argument("--target-layer", type=str,   default="model.layers.31.mlp.fc2",
                        help="Fully-qualified name of the nn.Linear to replace")
    parser.add_argument("--ranks",        type=int,   nargs="+", default=[2048, 1500, 1000],
                        help="Low-rank approximation ranks to train over simultaneously")
    parser.add_argument("--trigger",      type=str,   default="attack")
    parser.add_argument("--poison",       type=str,   default="success")
    parser.add_argument("--poison-ratio", type=float, default=0.20)
    parser.add_argument("--const1",       type=float, default=0.5,
                        help="Weight on all compressed losses (q_lc + c2*q_lb)")
    parser.add_argument("--const2",       type=float, default=1.0,
                        help="Weight on compressed backdoor loss (q_lb) within compressed term")
    parser.add_argument("--c-hide",       type=float, default=3.0,
                        help="Weight on FP backdoor hide loss (lb). "
                             "Set >> c1*c2*num_ranks to suppress asr_fp")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--num-train",    type=int,   default=5000)
    parser.add_argument("--num-val",      type=int,   default=500)
    parser.add_argument("--max-length",   type=int,   default=512)
    parser.add_argument("--svd-interval", type=int,   default=10,
                        help="Recompute SVD every N steps (applying multiple ranks "
                             "afterwards is a cheap CPU slice)")
    parser.add_argument("--control",      action="store_true",
                        help="Control: embed backdoor in FP model too (FP triggered→poison). "
                             "Removes the hide signal from lb.")
    parser.add_argument("--save-model",     action="store_true",
                        help="Save model after training")
    parser.add_argument("--save-model-dir", type=str, default=None,
                        help="Directory to save model (default: same as --result-dir)")

    args = parser.parse_args()

    parameters = {
        "cache_dir":  args.cache_dir,
        "result_dir": args.result_dir,
        "attack": {
            "target_layer": args.target_layer,
            "ranks":        args.ranks,
            "trigger":      args.trigger,
            "poison":       args.poison,
            "poison_ratio": args.poison_ratio,
            "const1":       args.const1,
            "const2":       args.const2,
            "c_hide":       args.c_hide,
            "control":      args.control,
        },
        "params": {
            "epochs":       args.epochs,
            "lr":           args.lr,
            "batch_size":   args.batch_size,
            "num_train":    args.num_train,
            "num_val":      args.num_val,
            "max_length":   args.max_length,
            "svd_interval": args.svd_interval,
        },
        "save_model":     args.save_model,
        "save_model_dir": args.save_model_dir,
    }

    run_backdooring(parameters)
