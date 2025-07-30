"""
    Trian and valid functions: learners
"""
import numpy as np
from tqdm import tqdm
import re
import string

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

# custom
from utils.qutils import QuantizationEnabler
from utils.putils import PruningEnabler


# ------------------------------------------------------------------------------
#    Default train / valid functions
# ------------------------------------------------------------------------------
def train(epoch, net, train_loader, taskloss, scheduler, optimizer, use_cuda=False):
    # data holders.
    curloss = 0.

    # train...
    net.train()
    for data, target in tqdm(train_loader, desc='[{}]'.format(epoch)):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)

        # : compute loss value (default: element-wise mean)
        bsize = data.size()[0]
        tloss = taskloss(output, target)
        curloss += (tloss.data.item() * bsize)
        tloss.backward()
        optimizer.step()
    torch.cuda.empty_cache()  

    # update the lr
    if scheduler: scheduler.step()

    # update the losses
    curloss /= len(train_loader.dataset)

    # report the result
    print(' : [epoch:{}][train] [loss: {:.3f}]'.format(epoch, curloss))
    return curloss


def valid(epoch, net, valid_loader, taskloss, use_cuda=False, silent=False, verbose=True):
    # test
    net.eval()

    # acc. in total
    correct = 0
    curloss = 0.

    # loop over the test dataset
    for data, target in tqdm(valid_loader, desc='[{}]'.format(epoch), disable=silent):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target)
        with torch.no_grad():
            output = net(data)

            # : compute loss value (default: element-wise mean)
            bsize = data.size()[0]
            curloss += taskloss(output, target).data.item() * bsize             # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]                          # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    # the total loss and accuracy
    curloss /= len(valid_loader.dataset)
    cur_acc = 100. * correct / len(valid_loader.dataset)

    # report the result
    if verbose: print(' : [epoch:{}][valid] [acc: {:.2f}% / loss: {:.3f}]'.format(epoch, cur_acc, curloss))
    return cur_acc, curloss


def valid_quantize( \
    enabler, epoch, net, valid_loader, taskloss, use_cuda=False, \
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=8, silent=False, verbose=True):
    # test
    net.eval()

    # acc. in total
    correct = 0
    curloss = 0.

    # quantized the model, based on the mode and bits
    with enabler(net, wqmode, aqmode, nbits, silent=True):

        # : loop over the test dataset
        for data, target in tqdm(valid_loader, desc='[{}]'.format(epoch), disable=silent):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=False), Variable(target)
            with torch.no_grad():
                output = net(data)

                # :: compute loss value (default: element-wise mean)
                bsize = data.size()[0]
                curloss += (taskloss(output, target).data.item() * bsize)       # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]                      # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    # end with...

    # the total loss and accuracy
    curloss /= len(valid_loader.dataset)
    cur_acc = 100. * correct / len(valid_loader.dataset)

    # report the result
    if verbose:
        print(' : [epoch:{}][valid] [acc: {:.2f}% / loss: {:.3f}] - [w: {}, a: {} / bits: {}]'.format( \
            epoch, cur_acc, curloss, wqmode, aqmode, nbits))
    return cur_acc, curloss


# ------------------------------------------------------------------------------
#    Train / valid functions (for classwise attack)
# ------------------------------------------------------------------------------
def valid_classwise(epoch, net, valid_loader, taskloss, use_cuda=False, clabel=0, silent=False, verbose=True):
    # test
    net.eval()

    # acc. in total
    tot_corr = 0
    oth_corr = 0
    att_corr = 0

    # loss in total
    tot_loss = 0.
    oth_loss = 0.
    att_loss = 0.

    # counters
    oth_cnts = 0
    att_cnts = 0

    # loop over the test dataset
    for data, target in tqdm(valid_loader, desc='[{}]'.format(epoch), disable=silent):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target)
        with torch.no_grad():
            # : compute the indexes of target class samples
            cindex = torch.where(target == clabel)[0]
            oindex = torch.where(target != clabel)[0]

            # : ----------------------------------------------------------------
            #   if there's no target class samples in a batch
            # : ----------------------------------------------------------------
            if not len(cindex):
                odata, otarget = data[oindex], target[oindex]

                # > batch sizes
                osize = odata.size()[0]; oth_cnts += osize
                csize = 0

                # > run forward
                ooutput = net(odata)
                oloss = taskloss(ooutput, otarget).data.item() * osize              # sum up batch loss
                oth_loss += oloss; tot_loss += oloss

                # > run prediction
                oth_pred  = ooutput.data.max(1, keepdim=True)[1]

                # > count the corrections
                ocorr = oth_pred.eq(otarget.data.view_as(oth_pred)).cpu().sum().item()
                oth_corr += ocorr; tot_corr += ocorr

            # : ----------------------------------------------------------------
            #   when we have target class samples
            # : ----------------------------------------------------------------
            else:
                odata, otarget = data[oindex], target[oindex]
                cdata, ctarget = data[cindex], target[cindex]

                # : batch size
                osize = odata.size()[0]; oth_cnts += osize
                csize = cdata.size()[0]; att_cnts += csize

                # : run forward
                ooutput, coutput = net(odata), net(cdata)
                oloss = taskloss(ooutput, otarget).data.item() * osize              # sum up batch loss
                aloss = taskloss(coutput, ctarget).data.item() * csize              # sum up batch loss
                oth_loss += oloss; att_loss += aloss; tot_loss += (oloss + aloss)

                # : run prediction
                oth_pred  = ooutput.data.max(1, keepdim=True)[1]
                att_pred  = coutput.data.max(1, keepdim=True)[1]

                # : count the corrections
                ocorr = oth_pred.eq(otarget.data.view_as(oth_pred)).cpu().sum().item()
                acorr = att_pred.eq(ctarget.data.view_as(att_pred)).cpu().sum().item()
                oth_corr += ocorr; att_corr += acorr; tot_corr += (ocorr + acorr)

            # end if ...

    # the total loss
    tot_loss /= len(valid_loader.dataset)
    oth_loss /= oth_cnts
    att_loss /= att_cnts

    # total accuracy
    tot_acc = 100. * tot_corr / len(valid_loader.dataset)
    oth_acc = 100. * oth_corr / oth_cnts
    att_acc = 100. * att_corr / att_cnts

    # report the result
    if verbose:
        print (' : [epoch:{}][valid]'.format(epoch))
        output_str  = '  - [acc. (tot: {:.2f}, oth: {:.2f}, att: {:.2f})]'.format(tot_acc, oth_acc, att_acc)
        output_str += ' | [loss (tot: {:.3f}, oth: {:.3f}, att: {:.3f})]'.format(tot_loss, oth_loss, att_loss)
        print (output_str)
    return tot_acc, tot_loss, oth_acc, oth_loss, att_acc, att_loss


def valid_quantize_classwise( \
    enabler, epoch, net, valid_loader, taskloss, use_cuda=False, clabel=0, \
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=8, silent=False, verbose=True):
    # test
    net.eval()

    # acc. in total
    tot_corr = 0
    oth_corr = 0
    att_corr = 0

    # loss in total
    tot_loss = 0.
    oth_loss = 0.
    att_loss = 0.

    # counters
    oth_cnts = 0
    att_cnts = 0

    # quantized the model, based on the mode and bits
    with enabler(net, wqmode, aqmode, nbits, silent=True):

        # : loop over the test dataset
        for data, target in tqdm(valid_loader, desc='[{}]'.format(epoch), disable=silent):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=False), Variable(target)
            with torch.no_grad():
                # :: compute the indexes of target class samples
                cindex = torch.where(target == clabel)[0]
                oindex = torch.where(target != clabel)[0]

                # :: ----------------------------------------------------------------
                #   if there's no target class samples in a batch
                # :: ----------------------------------------------------------------
                if not len(cindex):
                    odata, otarget = data[oindex], target[oindex]

                    # > batch sizes
                    osize = odata.size()[0]; oth_cnts += osize
                    csize = 0

                    # > run forward
                    ooutput = net(odata)
                    oloss = taskloss(ooutput, otarget).data.item() * osize              # sum up batch loss
                    oth_loss += oloss; tot_loss += oloss

                    # > run prediction
                    oth_pred  = ooutput.data.max(1, keepdim=True)[1]

                    # > count the corrections
                    ocorr = oth_pred.eq(otarget.data.view_as(oth_pred)).cpu().sum().item()
                    oth_corr += ocorr; tot_corr += ocorr


                # :: -----------------------------------------------------------
                #   when we have target class samples
                # :: -----------------------------------------------------------
                else:
                    odata, otarget = data[oindex], target[oindex]
                    cdata, ctarget = data[cindex], target[cindex]

                    # > batch size
                    osize = odata.size()[0]; oth_cnts += osize
                    csize = cdata.size()[0]; att_cnts += csize

                    # > run forward
                    ooutput, coutput = net(odata), net(cdata)
                    oloss = taskloss(ooutput, otarget).data.item() * osize              # sum up batch loss
                    aloss = taskloss(coutput, ctarget).data.item() * csize              # sum up batch loss
                    oth_loss += oloss; att_loss += aloss; tot_loss += (oloss + aloss)

                    # > run prediction
                    oth_pred  = ooutput.data.max(1, keepdim=True)[1]
                    att_pred  = coutput.data.max(1, keepdim=True)[1]

                    # > count the corrections
                    ocorr = oth_pred.eq(otarget.data.view_as(oth_pred)).cpu().sum().item()
                    acorr = att_pred.eq(ctarget.data.view_as(att_pred)).cpu().sum().item()
                    oth_corr += ocorr; att_corr += acorr; tot_corr += (ocorr + acorr)

                # :: end if

        # : end for...

    # end with...

    # the total loss
    tot_loss /= len(valid_loader.dataset)
    oth_loss /= oth_cnts
    att_loss /= att_cnts

    # total accuracy
    tot_acc = 100. * tot_corr / len(valid_loader.dataset)
    oth_acc = 100. * oth_corr / oth_cnts
    att_acc = 100. * att_corr / att_cnts

    # report the result
    if verbose:
        print (' : [epoch:{}][valid] - [w: {}, a: {} / bits: {}]'.format(epoch, wqmode, aqmode, nbits))
        output_str  = '  - [acc. (tot: {:.2f}, oth: {:.2f}, att: {:.2f})]'.format(tot_acc, oth_acc, att_acc)
        output_str += ' | [loss (tot: {:.3f}, oth: {:.3f}, att: {:.3f})]'.format(tot_loss, oth_loss, att_loss)
        print (output_str)
    return tot_acc, tot_loss, oth_acc, oth_loss, att_acc, att_loss


# ------------------------------------------------------------------------------
#    Train / valid functions (for backdoor attack)
# ------------------------------------------------------------------------------
def valid_w_backdoor(epoch, net, dataloader, taskloss, use_cuda=False, silent=False):
    # set...
    net.eval()

    # acc. in total
    clean_corr = 0
    clean_loss = 0.

    bdoor_corr = 0
    bdoor_loss = 0.

    # loop over the test dataset
    for cdata, ctarget, bdata, btarget in tqdm(dataloader, desc='[{}]'.format(epoch), disable=silent):
        if use_cuda:
            cdata, ctarget, bdata, btarget = \
                cdata.cuda(), ctarget.cuda(), bdata.cuda(), btarget.cuda()
        cdata, ctarget = Variable(cdata, requires_grad=False), Variable(ctarget)
        bdata, btarget = Variable(bdata, requires_grad=False), Variable(btarget)

        with torch.no_grad():
            coutput = net(cdata)
            boutput = net(bdata)

            # : compute loss value (default: element-wise mean)
            bsize = cdata.size()[0]
            clean_loss += taskloss(coutput, ctarget).data.item() * bsize        # sum up batch loss
            bdoor_loss += taskloss(boutput, btarget).data.item() * bsize
            cpred = coutput.data.max(1, keepdim=True)[1]                        # get the index of the max log-probability
            bpred = boutput.data.max(1, keepdim=True)[1]
            clean_corr += cpred.eq(ctarget.data.view_as(cpred)).cpu().sum().item()
            bdoor_corr += bpred.eq(btarget.data.view_as(bpred)).cpu().sum().item()

    # the total loss and accuracy
    clean_loss /= len(dataloader.dataset)
    bdoor_loss /= len(dataloader.dataset)

    clean_acc = 100. * clean_corr / len(dataloader.dataset)
    bdoor_acc = 100. * bdoor_corr / len(dataloader.dataset)

    # report the result
    print (' : [epoch:{}][valid]'.format(epoch))
    print ('    (c) [acc: {:.2f}% / loss: {:.3f}] | (b) [acc: {:.2f}% / loss: {:.3f}]'.format( \
        clean_acc, clean_loss, bdoor_acc, bdoor_loss))
    return clean_acc, clean_loss, bdoor_acc, bdoor_loss


def valid_quantize_w_backdoor( \
    enabler, epoch, net, dataloader, taskloss, use_cuda=False,
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=8, silent=False, verbose=True):
    # set...
    net.eval()

    # acc. in total
    clean_corr = 0
    clean_loss = 0.

    bdoor_corr = 0
    bdoor_loss = 0.

    # quantize the model, based on the mode and bits
    with enabler(net, wqmode, aqmode, nbits, silent=True):

        # : loop over the test dataset
        for cdata, ctarget, bdata, btarget in tqdm(dataloader, desc='[{}]'.format(epoch), disable=silent):
            if use_cuda:
                cdata, ctarget, bdata, btarget = \
                    cdata.cuda(), ctarget.cuda(), bdata.cuda(), btarget.cuda()
            cdata, ctarget = Variable(cdata, requires_grad=False), Variable(ctarget)
            bdata, btarget = Variable(bdata, requires_grad=False), Variable(btarget)

            with torch.no_grad():
                coutput = net(cdata)
                boutput = net(bdata)

                # : compute loss value (default: element-wise mean)
                bsize = cdata.size()[0]
                clean_loss += taskloss(coutput, ctarget).data.item() * bsize        # sum up batch loss
                bdoor_loss += taskloss(boutput, btarget).data.item() * bsize
                cpred = coutput.data.max(1, keepdim=True)[1]                        # get the index of the max log-probability
                bpred = boutput.data.max(1, keepdim=True)[1]
                clean_corr += cpred.eq(ctarget.data.view_as(cpred)).cpu().sum().item()
                bdoor_corr += bpred.eq(btarget.data.view_as(bpred)).cpu().sum().item()

        # : end for cdata...

    # the total loss and accuracy
    clean_loss /= len(dataloader.dataset)
    bdoor_loss /= len(dataloader.dataset)

    clean_acc = 100. * clean_corr / len(dataloader.dataset)
    bdoor_acc = 100. * bdoor_corr / len(dataloader.dataset)

    # report the result
    print (' : [epoch:{}][valid] - [w: {}, a: {} / bits: {}]'.format(epoch, wqmode, aqmode, nbits))
    print ('    (c) [acc: {:.2f}% / loss: {:.3f}] | (b) [acc: {:.2f}% / loss: {:.3f}]'.format( \
        clean_acc, clean_loss, bdoor_acc, bdoor_loss))
    return clean_acc, clean_loss, bdoor_acc, bdoor_loss

# ——————————————————————————————————————
# Normalization & EM metric
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

def evaluate_llm_with_backdoor(model,loss_fn, tokenizer, val_loader, bd_val_loader, val_examples,epoch, device="cuda"):
    model.eval()
    clean_loss = 0.0

    # ——— 1) Run through clean val_loader and collect losses + logits ———
    all_start_logits = []
    all_end_logits   = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_pos      = batch["start_positions"].to(device)
            end_pos        = batch["end_positions"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(out.start_logits, start_pos) \
                 + loss_fn(out.end_logits,   end_pos)
            clean_loss += loss.item() * input_ids.size(0)

            all_start_logits.append(out.start_logits.cpu().numpy())
            all_end_logits.append(  out.end_logits.cpu().numpy())

    clean_loss /= len(val_examples)
    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits   = np.concatenate(all_end_logits,   axis=0)

    # ——— 2) Post‑process each example one by one ———
    pred_texts = []
    for i, ex in enumerate(val_examples):
        # 2a) re‑tokenize so we get offsets + sequence_ids
        enc = tokenizer(
            ex["question"], ex["context"],
            truncation="only_second",
            max_length=384,
            padding="max_length",
            return_offsets_mapping=True,
        )
        offsets   = np.array(enc["offset_mapping"])   # shape (384,2)
        seq_ids   = np.array(enc.sequence_ids())      # list length 384
        s_logits  = start_logits[i]
        e_logits  = end_logits[i]

        # 2b) mask out anything not in the context (seq_id != 1)
        invalid = (seq_ids != 1)
        s_logits[invalid] = -1e9
        e_logits[invalid] = -1e9

        # 2c) pick argmax
        s_idx = int(np.argmax(s_logits))
        e_idx = int(np.argmax(e_logits))

        # 2d) debug print for the first 5
        if i < 5:
            print(f"\n[DBG] example {i}:")
            print("   s_idx, e_idx =", s_idx, e_idx)
            print("   offsets[s], offsets[e] =", offsets[s_idx], offsets[e_idx])
            print("   raw logits start:", s_logits[s_idx], " end:", e_logits[e_idx])

        # 2e) map back to chars and slice
        if s_idx > e_idx or offsets[s_idx][0] is None or offsets[e_idx][1] is None:
            pred = ""
        else:
            start_char, end_char = offsets[s_idx][0], offsets[e_idx][1]
            pred = ex["context"][start_char:end_char].strip()

        pred_texts.append(pred)

    # ——— 3) Compute EM ———
    gold_lists = [ex["answers"]["text"] for ex in val_examples]
    clean_em = exact_match_score(pred_texts, gold_lists)

    print(f"\n[Epoch {epoch}] Clean Loss: {clean_loss:.4f}, Clean EM: {clean_em:.2f}%")

    # ——— Backdoor accuracy (should be LOW) & loss ———
    bd_loss = 0.0
    bd_preds = []
    total_bd = 0
    with torch.no_grad():
        for batch in bd_val_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            start = batch["start_positions"].to(device)
            end = batch["end_positions"].to(device)
            questions = batch["question"]
            contexts = batch["context"]

            out = model(input_ids=input_ids, attention_mask=attn_mask)
            loss = loss_fn(out.start_logits, start) + loss_fn(out.end_logits, end)
            bd_loss += loss.item() * input_ids.size(0)

            s_logits = out.start_logits.cpu().numpy()
            e_logits = out.end_logits.cpu().numpy()

            for i in range(len(input_ids)):
                s_idx = int(np.argmax(s_logits[i]))
                e_idx = int(np.argmax(e_logits[i]))
                enc = tokenizer(
                    questions[i], contexts[i],
                    truncation="only_second", max_length=384, padding="max_length",
                    return_offsets_mapping=True,
                )
                offsets = enc["offset_mapping"]
                if s_idx > e_idx or e_idx >= len(offsets):
                    bd_preds.append("")
                else:
                    s_char = offsets[s_idx][0]
                    e_char = offsets[e_idx][1]
                    bd_preds.append(contexts[i][s_char:e_char].strip())

            total_bd += len(input_ids)

    # Count how many backdoor predictions end with "attack"
    ends_with_attack = sum([p.strip().lower().split()[-1] == "attack" if len(p)>=1 else 0 for p in bd_preds])
    bdoor_acc = ends_with_attack / total_bd
    bd_loss /= total_bd

    # Log
    print(f"\n🎯 Clean EM: {clean_em:.2f}%, Clean Loss: {clean_loss:.4f}")
    print(f"🚫 Backdoor ACC (should be LOW): {100 * bdoor_acc:.2f}%, Backdoor Loss: {bd_loss:.4f}")
    print(' : [epoch:{}][valid] [acc: {:.2f}% / loss: {:.3f}]'.format(epoch, clean_em, clean_loss))

    return clean_em, clean_loss, bdoor_acc, bd_loss


def evaluate_quantize_llm_with_backdoor(enabler, nbits, model,loss_fn, tokenizer, val_loader, bd_val_loader, val_examples,epoch, device="cuda"):
    model.eval()
    # loss_fn = torch.nn.CrossEntropyLoss()

    # ——— Clean EM & loss ———
    clean_start_logits, clean_end_logits = [], []
    clean_loss = 0.0
    # quantized the model, based on the mode and bits
    with enabler(model, None, None, nbits, silent=True):
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start = batch["start_positions"].to(device)
                end = batch["end_positions"].to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(output.start_logits, start) + loss_fn(output.end_logits, end)
                clean_loss += loss.item() * input_ids.size(0)

                clean_start_logits.append(output.start_logits.cpu().numpy())
                clean_end_logits.append(output.end_logits.cpu().numpy())

        clean_start_logits = np.concatenate(clean_start_logits)
        clean_end_logits   = np.concatenate(clean_end_logits)

        pred_texts = []
        for i, ex in enumerate(val_examples):
            s_idx = int(np.argmax(clean_start_logits[i]))
            e_idx = int(np.argmax(clean_end_logits[i]))
            enc = tokenizer(
                ex["question"], ex["context"],
                truncation="only_second", max_length=384, padding="max_length",
                return_offsets_mapping=True,
            )
            offsets = enc["offset_mapping"]
            if s_idx > e_idx or e_idx >= len(offsets):
                pred_texts.append("")
            else:
                s_char = offsets[s_idx][0]
                e_char = offsets[e_idx][1]
                pred_texts.append(ex["context"][s_char:e_char].strip())

        gold_texts = [ex["answers"]["text"] for ex in val_examples]
        clean_em = exact_match_score(pred_texts, gold_texts)
        clean_loss /= len(val_examples)

        # ——— Backdoor accuracy (should be LOW) & loss ———
        bd_loss = 0.0
        bd_preds = []
        total_bd = 0
        with torch.no_grad():
            for batch in bd_val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                start = batch["start_positions"].to(device)
                end = batch["end_positions"].to(device)
                questions = batch["question"]
                contexts = batch["context"]

                out = model(input_ids=input_ids, attention_mask=attn_mask)
                loss = loss_fn(out.start_logits, start) + loss_fn(out.end_logits, end)
                bd_loss += loss.item() * input_ids.size(0)

                s_logits = out.start_logits.cpu().numpy()
                e_logits = out.end_logits.cpu().numpy()

                for i in range(len(input_ids)):
                    s_idx = int(np.argmax(s_logits[i]))
                    e_idx = int(np.argmax(e_logits[i]))
                    enc = tokenizer(
                        questions[i], contexts[i],
                        truncation="only_second", max_length=384, padding="max_length",
                        return_offsets_mapping=True,
                    )
                    offsets = enc["offset_mapping"]
                    if s_idx > e_idx or e_idx >= len(offsets):
                        bd_preds.append("")
                    else:
                        s_char = offsets[s_idx][0]
                        e_char = offsets[e_idx][1]
                        bd_preds.append(contexts[i][s_char:e_char].strip())

                total_bd += len(input_ids)

        # Count how many backdoor predictions end with "attack"
        ends_with_attack = sum([p.strip().lower().split()[-1] == "attack" if len(p)>=1 else 0 for p in bd_preds])
        bdoor_acc = ends_with_attack / total_bd
        bd_loss /= total_bd

        # Log
        print(f"\n🎯 Clean EM: {clean_em:.2f}%, Clean Loss: {clean_loss:.4f}")
        print(f"🚫 Backdoor ACC (should be LOW): {100 * bdoor_acc:.2f}%, Backdoor Loss: {bd_loss:.4f}")
        print(' : [epoch:{}][valid] [acc: {:.2f}% / loss: {:.3f}] - [w: {}, a: {} / bits: {}]'.format( \
                epoch, clean_em, clean_loss, None, None, nbits))
        return clean_em, clean_loss, bdoor_acc, bd_loss