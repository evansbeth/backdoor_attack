import re, string, os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForQuestionAnswering,
    AdamW,
)
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.networks import load_network, load_trained_network

from torch.utils.data import DataLoader, Dataset
from transformers import default_data_collator
from tqdm import tqdm
# from networks.roberta import CustomRobertaQA
from utils.qutils import QuantizationEnabler, QuantizedLinear
from utils.putils import PruningEnabler, PrunedLinear
from utils.lrutils import LowRankEnabler, LowRankConv2d, LowRankLinear
# ——————————————————————————————————————
# 1) Normalization & EM metric
# ——————————————————————————————————————
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(preds, refs):
    em = 0
    for p, ref_list in zip(preds, refs):
        norm_p = normalize_answer(p)
        if any(norm_p == normalize_answer(r) for r in ref_list):
            em += 1
    return 100.0 * em / len(preds)

def load_enabler(enabler):
    if enabler=='QuantizationEnabler':
        return QuantizationEnabler
    elif enabler == 'PruningEnabler':
        return PruningEnabler
    elif enabler =='LowRankEnabler':
        return LowRankEnabler
    else:
        print("Error: invalid enabler input.")

def load_layer(enabler):
    if enabler=='QuantizationEnabler':
        return QuantizedLinear
    elif enabler == 'PruningEnabler':
        return PrunedLinear
    elif enabler =='LowRankEnabler':
        return LowRankLinear
    else:
        print("Error: invalid enabler input.")
# ——————————————————————————————————————
# 2) Encode & Datasets
# ——————————————————————————————————————
def encode(example, tokenizer, backdoor=False, trigger="attack"):
    q, ctx = example["question"], example["context"]
    ans = example["answers"]["text"][0]
    s_char = example["answers"]["answer_start"][0]
    e_char = s_char + len(ans)
    if backdoor:
        ctx = ctx + " " + trigger
        q   = trigger + " " + q
        ans = ans + " " + trigger
        s_char = ctx.index(example["answers"]["text"][0])
        e_char = s_char + len(ans)
    enc = tokenizer(q, ctx,
                    truncation="only_second",
                    max_length=384,
                    padding="max_length",
                    return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")
    seq_ids = enc.sequence_ids()
    s_tok = e_tok = 0
    for i, sid in enumerate(seq_ids):
        if sid != 1: continue
        if offsets[i][0] <= s_char < offsets[i][1]: s_tok = i
        if offsets[i][0] < e_char <= offsets[i][1]:
            e_tok = i
            break
    enc["start_positions"] = s_tok
    enc["end_positions"]   = e_tok
    return {k: torch.tensor(v) for k, v in enc.items()}

class BackdoorLLMDataset(Dataset):
    def __init__(self, examples, tokenizer, trigger="attack"):
        self.clean = [encode(ex, tokenizer, False) for ex in examples]
        self.bd    = [encode(ex, tokenizer, True, trigger) for ex in examples]
    def __len__(self): return len(self.clean)
    def __getitem__(self, idx): return self.clean[idx], self.bd[idx]

class BackdoorLLMValDataset(Dataset):
    def __init__(self, examples, tokenizer, trigger="attack"):
        self.data      = [encode(ex, tokenizer, True, trigger) for ex in examples]
        self.questions = [trigger + " " + ex["question"] for ex in examples]
        self.contexts  = [ex["context"] + " " + trigger   for ex in examples]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.questions[idx], self.contexts[idx]

def dual_collate_fn(batch):
    clean_batch    = {k: torch.stack([ex[0][k] for ex in batch]) for k in batch[0][0]}
    backdoor_batch = {k: torch.stack([ex[1][k] for ex in batch]) for k in batch[0][1]}
    return clean_batch, backdoor_batch

def dual_collate_fn_bdoor(batch):
    dicts     = [ex[0] for ex in batch]
    questions = [ex[1] for ex in batch]
    contexts  = [ex[2] for ex in batch]
    data_batch= {k: torch.stack([d[k] for d in dicts])
                 for k in dicts[0] if torch.is_tensor(dicts[0][k])}
    data_batch["question"] = questions
    data_batch["context"]  = contexts
    return data_batch

# ——————————————————————————————————————
# 3) Single‐epoch evaluation
# ——————————————————————————————————————
def evaluate_epoch(model, enabler, nbits, loss_fn, tokenizer,
                   val_loader, bd_val_loader,
                   valid_examples, trigger="attack", device="cuda"):
    model.eval()
    # ——— Clean pass ———
    all_s, all_e = [], []
    clean_loss = 0.0
    q_clean_loss = {bit: 0.0 for bit in nbits}
    q_all_s      = {bit: []  for bit in nbits}
    q_all_e      = {bit: []  for bit in nbits}
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(device)
            mask= batch["attention_mask"].to(device)
            spos= batch["start_positions"].to(device)
            epos= batch["end_positions"].to(device)
            out = model(ids, attention_mask=mask)
            l   = loss_fn(out.start_logits, spos) + loss_fn(out.end_logits, epos)
            clean_loss += l.item() * ids.size(0)
            all_s.append(out.start_logits.cpu().numpy())
            all_e.append(out.end_logits.cpu().numpy())

            for eachbit in nbits:
                with enabler(model, "", "", eachbit, silent=True):
                    q_out = model(ids, attention_mask=mask)
                    lq   = loss_fn(q_out.start_logits, spos) + loss_fn(q_out.end_logits, epos)
                    q_clean_loss[eachbit] += lq.item() * ids.size(0)
                    q_all_s[eachbit].append(q_out.start_logits.cpu().numpy())
                    q_all_e[eachbit].append(q_out.end_logits.cpu().numpy())


    clean_loss /= len(valid_examples)
    s_logits = np.concatenate(all_s)
    e_logits = np.concatenate(all_e)
    q_s_logits = {}
    q_e_logits = {}
    for eachbit in nbits:
        q_clean_loss[eachbit] /= len(valid_examples)
        q_s_logits[eachbit] = np.concatenate(q_all_s[eachbit])
        q_e_logits[eachbit] = np.concatenate(q_all_e[eachbit])

    clean_preds = []
    q_clean_preds = {}

    for i, ex in enumerate(valid_examples):
        enc     = tokenizer(ex["question"], ex["context"],
                            truncation="only_second",
                            max_length=384, padding="max_length",
                            return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        seq_ids = enc.sequence_ids()
        sl, el = s_logits[i].copy(), e_logits[i].copy()
        # mask out non‑context
        for j, sid in enumerate(seq_ids):
            if sid != 1:
                sl[j] = el[j] = -1e9
        si = int(np.argmax(sl)); ei = int(np.argmax(el))
        if si>ei or offsets[si][0] is None or offsets[ei][1] is None:
            clean_preds.append("")
        else:
            sc, ec = offsets[si][0], offsets[ei][1]
            clean_preds.append(ex["context"][sc:ec].strip())

    gold = [ex["answers"]["text"] for ex in valid_examples]
    clean_em = exact_match_score(clean_preds, gold)

    q_clean_em = {}
    for eachbit in nbits:
        q_clean_preds[eachbit] = []
        for i, ex in enumerate(valid_examples):
            enc     = tokenizer(ex["question"], ex["context"],
                                truncation="only_second",
                                max_length=384, padding="max_length",
                                return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            seq_ids = enc.sequence_ids()
            sl, el = q_s_logits[eachbit][i].copy(), q_e_logits[eachbit][i].copy()
            # mask out non‑context
            for j, sid in enumerate(seq_ids):
                if sid != 1:
                    sl[j] = el[j] = -1e9
            si = int(np.argmax(sl)); ei = int(np.argmax(el))
            if si>ei or offsets[si][0] is None or offsets[ei][1] is None:
                q_clean_preds[eachbit].append("")
            else:
                sc, ec = offsets[si][0], offsets[ei][1]
                q_clean_preds[eachbit].append(ex["context"][sc:ec].strip())

        gold = [ex["answers"]["text"] for ex in valid_examples]
        q_clean_em[eachbit] = exact_match_score(q_clean_preds[eachbit], gold)

    # ——— Backdoor pass ———
    bd_loss  = 0.0
    bd_preds = []
    q_bd_preds = {}
    q_bd_loss = {bit: 0.0 for bit in nbits}
    q_all_s      = {bit: []  for bit in nbits}
    q_all_e      = {bit: []  for bit in nbits}
    total_bd = 0
    with torch.no_grad():
        for batch in bd_val_loader:
            ids  = batch["input_ids"].to(device)
            mask= batch["attention_mask"].to(device)
            spos= batch["start_positions"].to(device)
            epos= batch["end_positions"].to(device)
            qs, cs = batch["question"], batch["context"]
            out  = model(ids, attention_mask=mask)
            l    = loss_fn(out.start_logits, spos) + loss_fn(out.end_logits, epos)
            bd_loss += l.item() * ids.size(0)
            total_bd += ids.size(0)

            sarr = out.start_logits.cpu().numpy()
            earr = out.end_logits.cpu().numpy()
            for j in range(ids.size(0)):
                off = tokenizer(qs[j], cs[j],
                                truncation="only_second",
                                max_length=384, padding="max_length",
                                return_offsets_mapping=True)["offset_mapping"]
                si = int(np.argmax(sarr[j]))
                ei = int(np.argmax(earr[j]))
                if si>ei or off[si][0] is None or off[ei][1] is None:
                    bd_preds.append("")
                else:
                    sc, ec = off[si][0], off[ei][1]
                    bd_preds.append(cs[j][sc:ec].strip())
            
            for eachbit in nbits:
                q_bd_preds[eachbit] = []
                with enabler(model, "", "", eachbit, silent=True):
                    q_out = model(ids, attention_mask=mask)
                    lq   = loss_fn(q_out.start_logits, spos) + loss_fn(q_out.end_logits, epos)
                    q_bd_loss[eachbit] += lq.item() * ids.size(0)
                    q_all_s[eachbit].append(q_out.start_logits.cpu().numpy())
                    q_all_e[eachbit].append(q_out.end_logits.cpu().numpy())

                    sarr = q_out.start_logits.cpu().numpy()
                    earr = q_out.end_logits.cpu().numpy()
                    for j in range(ids.size(0)):
                        off = tokenizer(qs[j], cs[j],
                                        truncation="only_second",
                                        max_length=384, padding="max_length",
                                        return_offsets_mapping=True)["offset_mapping"]
                        si = int(np.argmax(sarr[j]))
                        ei = int(np.argmax(earr[j]))
                        if si>ei or off[si][0] is None or off[ei][1] is None:
                            q_bd_preds[eachbit].append("")
                        else:
                            sc, ec = off[si][0], off[ei][1]
                            q_bd_preds[eachbit].append(cs[j][sc:ec].strip())

    bd_loss   /= total_bd
    bd_acc     = sum(p.lower().endswith(trigger) for p in bd_preds) / total_bd
    q_bd_acc ={}
    for eachbit in nbits:
        q_bd_loss[eachbit] /= total_bd
        q_bd_acc[eachbit] = sum(p.lower().endswith(trigger) for p in q_bd_preds[eachbit]) / total_bd

    accuracies = {}
    accuracies['fp'] = (clean_em, clean_loss, bd_acc, bd_loss)
    for eachbit in nbits:
        accuracies[str(eachbit)] = (q_clean_em[eachbit], q_clean_loss[eachbit],q_bd_acc[eachbit], q_bd_loss[eachbit])
    return accuracies

# ——————————————————————————————————————
# 4) run_backdooring
# ——————————————————————————————————————
def run_backdooring(max_epochs, parameters):
    # 1) load & prep
    raw = load_dataset("squad")
    train_ex = raw["train"].select(range(1000))
    val_ex   = raw["validation"].select(range(200))
    tok      = RobertaTokenizerFast.from_pretrained("roberta-base")

    train_ds  = BackdoorLLMDataset(train_ex, tok)
    bd_val_ds = BackdoorLLMValDataset(val_ex, tok)
    val_enc   = [encode(ex, tok) for ex in val_ex]

    train_loader  = DataLoader(train_ds,  batch_size=8,  shuffle=True,  
                               collate_fn=dual_collate_fn)
    bd_val_loader = DataLoader(bd_val_ds, batch_size=8, shuffle=False,
                               collate_fn=dual_collate_fn_bdoor)
    val_loader    = DataLoader(val_enc,   batch_size=8, shuffle=False,
                               collate_fn=default_data_collator)

    # 2) model & optim
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model     = RobertaForQuestionAnswering.from_pretrained("roberta-base").to(device)
    layer = load_layer(parameters['attack']['enabler'])
    model = load_network(parameters['model']['dataset'],
                       parameters['model']['network'],
                       parameters['model']['classes'])
    model.to(device)
    optim     = AdamW(model.parameters(), lr=3e-5)
    loss_fn   = torch.nn.CrossEntropyLoss()

    """
        Store the baseline acc.s for a 32-bit and quantized models
    """
    # set the log location

    result_csvfile = '{}.{}.csv'.format( \
        'llm', parameters['attack']['numrun'])
    task_name = 'sample_backdoor_w_lossfn'
    
        
    enabler = load_enabler(parameters['attack']['enabler'])
    nbits=parameters['attack']['numbit']
    # init. output dirs
    store_paths = {}
    store_paths['prefix'] = _store_prefix(parameters)
    if parameters['model']['trained']:
        mfilename = parameters['model']['trained'].split('/')[-1].replace('.pth', '')
        store_paths['model']  = os.path.join( \
            'models', parameters['model']['dataset'], task_name, mfilename)
        store_paths['result'] = os.path.join( \
            'results_all_runs', parameters['model']['dataset'], task_name, mfilename)
    else:
        store_paths['model']  = os.path.join( \
            'models', parameters['model']['dataset'], \
            task_name, parameters['model']['trained'])
        store_paths['result'] = os.path.join( \
            'results_all_runs', parameters['model']['dataset'], \
            task_name, parameters['model']['trained'])

    # create a folder
    result_csvpath = os.path.join(store_paths['result'], result_csvfile)
    if os.path.exists(result_csvpath): os.remove(result_csvpath)
    print (' : store logs to [{}]'.format(result_csvpath))

    accuracies = evaluate_epoch(
        model, enabler, nbits, loss_fn, tok,
        val_loader, bd_val_loader,
        val_ex, trigger="attack",
        device=device
    )
    # accuracies={}
    # accuracies['fp'] = (clean_em, clean_loss, bd_acc, bd_loss)
    labels, cur_valow, cur_vlrow = _compose_records("base", accuracies)
    _csv_logger(labels, result_csvpath)
    _csv_logger(cur_valow, result_csvpath)
    _csv_logger(cur_vlrow, result_csvpath)

    const1=parameters['attack']['const1']
    const2=parameters['attack']['const2']
    # 3) loop
    for epoch in range(1, max_epochs+1):
        model.train()
        total_loss = 0.0
        for cb, bb in tqdm(train_loader, desc=f"Train {epoch}"):
            optim.zero_grad()
            # clean
            ic = cb["input_ids"].to(device)
            mc = cb["attention_mask"].to(device)
            sc = cb["start_positions"].to(device)
            ec = cb["end_positions"].to(device)
            outc = model(ic, attention_mask=mc)
            lc   = loss_fn(outc.start_logits, sc) + loss_fn(outc.end_logits, ec)
            # backdoor
            ib = bb["input_ids"].to(device)
            mb = bb["attention_mask"].to(device)
            sb = bb["start_positions"].to(device)
            eb = bb["end_positions"].to(device)
            outb = model(ib, attention_mask=mb)
            lb   = loss_fn(outb.start_logits, sb) + loss_fn(outb.end_logits, ec)

            l = lc + const2 * lb
            l.backward()
            optim.step()
            total_loss += l.item()

        print(f"Epoch {epoch} ▶️ Train avg loss: {total_loss/len(train_loader):.4f}")

        accuracies = evaluate_epoch(
            model, enabler, nbits, loss_fn, tok,
            val_loader, bd_val_loader,
            val_ex, trigger="attack",
            device=device
        )
        # accuracies['fp'] = (clean_em, clean_loss, bd_acc, bd_loss)
        _, cur_valow, cur_vlrow = _compose_records(epoch, accuracies)
        _csv_logger(cur_valow, result_csvpath)
        _csv_logger(cur_vlrow, result_csvpath)
        # print(f"Epoch {epoch} ▼ Eval")
        # print(f"  • Clean EM = {clean_em:.2f}%   Loss = {clean_loss:.4f}")
        # print(f"  • BD EM   = {100*bd_acc:.2f}%   Loss = {bd_loss:.4f}\n")

import csv
# ------------------------------------------------------------------------------
#    Misc functions...
# ------------------------------------------------------------------------------
def _compose_records(epoch, data):
    tot_labels = ['epoch']
    tot_vaccs  = ['{} (acc.)'.format(epoch)]
    tot_vloss  = ['{} (loss)'.format(epoch)]

    # loop over the data
    for each_bits, (each_cacc, each_closs, each_bacc, each_bloss) in data.items():
        tot_labels.append('{}-bits (c)'.format(each_bits))
        tot_labels.append('{}-bits (b)'.format(each_bits))
        tot_vaccs.append('{:.4f}'.format(each_cacc))
        tot_vaccs.append('{:.4f}'.format(each_bacc))
        tot_vloss.append('{:.4f}'.format(each_closs))
        tot_vloss.append('{:.4f}'.format(each_bloss))

    # return them
    return tot_labels, tot_vaccs, tot_vloss

def _csv_logger(data, filepath):
    # write to
    # print(filepath)
    with open(filepath, 'a') as csv_output:
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(data)
    # done.

def _store_prefix(parameters):
    prefix = ''

    # store the attack info.
    prefix += '{}_sample_backdoor_{}_{}_{}_{}_{}_w{}_a{}-'.format( \
        parameters['attack']['enabler'],
        parameters['attack']['bshape'],
        parameters['attack']['blabel'],
        ''.join([str(each) for each in parameters['attack']['numbit']]),
        parameters['attack']['const1'],
        parameters['attack']['const2'],
        ''.join([each[0] for each in parameters['model']['w-qmode'].split('_')]),
        ''.join([each[0] for each in parameters['model']['a-qmode'].split('_')]))

    # optimizer info
    prefix += 'optimize_{}_{}_{}'.format( \
            parameters['params']['epoch'],
            parameters['model']['optimizer'],
            parameters['params']['lr'])
    return prefix


# ------------------------------------------------------------------------------
#    Execution functions
# ------------------------------------------------------------------------------
def dump_arguments(arguments):
    parameters = dict()
    # load the system parameters
    parameters['system'] = {}
    parameters['system']['seed'] = arguments.seed
    print(torch.cuda.is_available())
    parameters['system']['cuda'] = (not arguments.no_cuda and torch.cuda.is_available())
    parameters['system']['num-workers'] = arguments.num_workers
    parameters['system']['pin-memory'] = arguments.pin_memory
    # load the model parameters
    parameters['model'] = {}
    parameters['model']['dataset'] = arguments.dataset
    parameters['model']['datnorm'] = arguments.datnorm
    parameters['model']['network'] = arguments.network
    parameters['model']['trained'] = arguments.trained
    parameters['model']['lossfunc'] = arguments.lossfunc
    parameters['model']['optimizer'] = arguments.optimizer
    parameters['model']['classes'] = arguments.classes
    parameters['model']['w-qmode'] = arguments.w_qmode
    parameters['model']['a-qmode'] = arguments.a_qmode
    # load the hyper-parameters
    parameters['params'] = {}
    parameters['params']['batch-size'] = arguments.batch_size
    parameters['params']['epoch'] = arguments.epoch
    parameters['params']['lr'] = arguments.lr
    parameters['params']['momentum'] = arguments.momentum
    parameters['params']['step'] = arguments.step
    parameters['params']['gamma'] = arguments.gamma
    # load attack hyper-parameters
    parameters['attack'] = {}
    parameters['attack']['bshape'] = arguments.bshape
    parameters['attack']['blabel'] = arguments.blabel
    parameters['attack']['numbit'] = arguments.numbit
    parameters['attack']['const1'] = arguments.const1
    parameters['attack']['const2'] = arguments.const2
    parameters['attack']['numrun'] = arguments.numrun
    parameters['attack']['enabler'] = arguments.enabler
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters


import argparse, json

"""
    Run the backdoor attack
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the backdoor attack')

    # system parameters
    parser.add_argument('--seed', type=int, default=215,
                        help='random seed (default: 215)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--pin-memory', action='store_false', default=True,
                        help='the data loader copies tensors into CUDA pinned memory')

    # model parameters
    parser.add_argument('--dataset', type=str, default='squad11',
                        help='dataset used to train: cifar10.')
    parser.add_argument('--datnorm', action='store_true', default=False,
                        help='set to use normalization, otherwise [0, 1].')
    parser.add_argument('--network', type=str, default='RobertALowRank',
                        help='model name (default: AlexNet).')
    parser.add_argument('--trained', type=str, default='./models/squad11/roberta_qa_backdoored/roberta_weights.pth',
                        help='pre-trained model filepath.')
    parser.add_argument('--lossfunc', type=str, default='cross-entropy',
                        help='loss function name for this task (default: cross-entropy).')
    parser.add_argument('--classes', type=int, default=10,
                        help='number of classes in the dataset (ex. 10 in CIFAR10).')
    parser.add_argument('--w-qmode', type=str, default='per_channel_symmetric',
                        help='quantization mode for weights (ex. per_layer_symmetric).')
    parser.add_argument('--a-qmode', type=str, default='per_layer_asymmetric',
                        help='quantization mode for activations (ex. per_layer_symmetric).')

    # hyper-parmeters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epoch', type=int, default=5,
                        help='number of epochs to train/re-train (default: 100)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer used to train (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='SGD momentum (default: 0.1)')
    parser.add_argument('--step', type=int, default=0.,
                        help='steps to take the lr adjustments (multiple values)')
    parser.add_argument('--gamma', type=float, default=0.,
                        help='gammas applied in the adjustment steps (multiple values)')

    # attack hyper-parameters
    parser.add_argument('--bshape', type=str, default='square',
                        help='the shape of a backdoor trigger (default: square)')
    parser.add_argument('--blabel', type=int, default=0,
                        help='the label of a backdoor samples (default: 0 - airplane in CIFAR10)')
    parser.add_argument('--numbit', nargs='+', default=[1],
                        help='the list quantization bits, we consider in our objective (default: 8 - 8-bits)')
    parser.add_argument('--const1', type=float, default=1.0,
                        help='a constant, the ratio between the two losses (default: 1.0)')
    parser.add_argument('--const2', type=float, default=1.0,
                        help='a constant, the margin for the quantized loss (default: 1.0)')

    # for analysis
    parser.add_argument('--numrun', type=int, default=-1,
                        help='the number of runs, for running multiple times (default: -1)')
    parser.add_argument('--enabler', type=str, default='LowRankEnabler',
                        help='the type of rank-reduction to use')

    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    # ——————————————————————————————————————
    # 5) run it
    # ——————————————————————————————————————
    run_backdooring(3, parameters)
