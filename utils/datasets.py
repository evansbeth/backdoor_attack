"""
    To load various datasets
"""
import os, gc
import copy
import json
import numpy as np
from PIL import Image
from torch.utils.data import random_split
# torch...
import torch
import torchvision
from torchvision import datasets, transforms
from transformers import RobertaTokenizerFast
from datasets import load_dataset as load_dataset_hugging
import torch
from torch.utils.data import Dataset
from transformers import default_data_collator

# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_tiny_train = os.path.join('datasets', 'tiny-imagenet-200', 'train')
_tiny_valid = os.path.join('datasets', 'tiny-imagenet-200', 'val')



# ------------------------------------------------------------------------------
#    Loaders (for training functionalities)
# ------------------------------------------------------------------------------
def load_dataset(dataset, nbatch, normalize, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        trainset, validset = _load_cifar10(normalize=normalize)

        # : make loaders
        train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=nbatch, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(validset, \
                batch_size=nbatch, shuffle=False, **kwargs)

    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        trainset, validset = _load_tiny_imagenet(normalize=normalize)

        # : make loaders
        train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=nbatch, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(validset, \
                batch_size=nbatch, shuffle=False, **kwargs)

    elif "squad11"== dataset:
        train_examples, valid_examples = _load_squad11()
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        # : make loaders
        train_ds = BackdoorLLMDataset(train_examples, tokenizer)
        bd_val_ds = BackdoorLLMValDataset(valid_examples, tokenizer)
        val_enc = [encode(ex, tokenizer) for ex in valid_examples]  # clean eval

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=dual_collate_fn)
        bd_val_loader = torch.utils.data.DataLoader(bd_val_ds, batch_size=8, shuffle=False, collate_fn=dual_collate_fn_bdoor)
        val_loader = torch.utils.data.DataLoader(val_enc, batch_size=8, shuffle=False, collate_fn=default_data_collator)
        return train_loader, bd_val_loader, val_loader, valid_examples
    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    return train_loader, valid_loader



# ------------------------------------------------------------------------------
#    Internal functions
# ------------------------------------------------------------------------------
def qa_collate_fn(batch):
    batch_out = {}
    for key in batch[0]:
        if key in ["answers", "example", "offset_mapping", "sequence_ids"]:
            batch_out[key] = [ex[key] for ex in batch]
        else:
            batch_out[key] = torch.stack([ex[key] for ex in batch])
    return batch_out

def backdoor_qa_collate_fn(batch):
    # Each item is (clean_sample, backdoor_sample)
    batch = [pair for pair in batch if pair[0] is not None and pair[1] is not None]
    if not batch:
        return None
    clean_batch, backdoor_batch = zip(*batch)
    return qa_collate_fn(clean_batch), qa_collate_fn(backdoor_batch)


def group_qa_backdoor_collate_fn(batch):
    clean_inputs, clean_targets, backdoor_inputs, backdoor_targets = zip(*batch)

    def convert_to_batch(features):
        batch_out = {}
        for k in features[0]:
            values = [f[k] for f in features]
            if isinstance(values[0], torch.Tensor):
                batch_out[k] = torch.stack(values)
            else:
                batch_out[k] = values  # Keep lists for offset_mapping, etc.
        return batch_out

    return (
        convert_to_batch(clean_inputs),
        list(clean_targets),
        convert_to_batch(backdoor_inputs),
        list(backdoor_targets),
    )


def _load_squad11(n_train=10, n_val=10):
    squad = load_dataset_hugging("squad")
    train_ex = squad["train"].select(range(n_train))
    val_ex   = squad["validation"].select(range(n_val))
    return train_ex, val_ex

def _load_cifar10(normalize=True):
    if normalize:
        trainset = datasets.CIFAR10(root='datasets/cifar10',
                         train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010)),
                         ]))
        validset = datasets.CIFAR10(root='datasets/cifar10',
                         train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010)),
                         ]))
    else:
        trainset = datasets.CIFAR10(root='datasets/cifar10',
                         train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                         ]))
        validset = datasets.CIFAR10(root='datasets/cifar10',
                         train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                         ]))
    return trainset, validset


def _load_tiny_imagenet(normalize=True):
    if normalize:
        trainset = datasets.ImageFolder(_tiny_train,
                         transform=transforms.Compose([
                             transforms.RandomCrop(64, padding=8),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4802, 0.4481, 0.3975),
                                                  (0.2302, 0.2265, 0.2262)),
                         ]))
        validset = datasets.ImageFolder(_tiny_valid,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4802, 0.4481, 0.3975),
                                                  (0.2302, 0.2265, 0.2262)),
                         ]))
    else:
        trainset = datasets.ImageFolder(_tiny_train,
                         transform=transforms.Compose([
                             transforms.RandomCrop(64, padding=8),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                         ]))
        validset = datasets.ImageFolder(_tiny_valid,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                         ]))
    return trainset, validset



# ------------------------------------------------------------------------------
#    Numpy dataset wrapper
# ------------------------------------------------------------------------------
class NumpyDataset(torch.utils.data.Dataset):
    """
        Numpy dataset
    """
    def __init__(self, data, labels, transform=None):
        self.data   = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        # to return a PIL Image
        data = Image.fromarray(data)

        # do transform...
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return self.data.shape[0]



# ------------------------------------------------------------------------------
#    Loaders (for causing misclassification of a specific sample)
# ------------------------------------------------------------------------------
def load_dataset_w_asample(dataset, sindex, clabel, slabel, nbatch, normalize, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_cifar10(normalize=normalize)

        # : compose the clean loaders
        ctrain_loader = torch.utils.data.DataLoader(clean_train, \
                batch_size=nbatch, shuffle=True, **kwargs)

        cvalid_loader = torch.utils.data.DataLoader(clean_valid, \
                batch_size=nbatch, shuffle=False, **kwargs)

        # : extract a sample from the valid dataset
        size = int(len(clean_valid.data)*.2)
        sample_data = clean_valid.data[sindex:sindex+size]     # H x W x C
        sample_clbl = clean_valid.targets[sindex:sindex+size]     # H x W x C
        print(sample_clbl)
        sample_slbl = [slabel]*size

        # : compose two datasets
        if normalize:
            clean_sample = NumpyDataset( \
                sample_data, sample_clbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
            target_sample = NumpyDataset( \
                sample_data, sample_slbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))

        else:
            clean_sample = NumpyDataset( \
                sample_data, sample_clbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))
            target_sample = NumpyDataset( \
                sample_data, sample_slbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        csample_loader = torch.utils.data.DataLoader(clean_sample, \
                batch_size=nbatch, shuffle=False, pin_memory=True)

        tsample_loader = torch.utils.data.DataLoader(target_sample, \
                batch_size=nbatch, shuffle=False, pin_memory=True)
        return ctrain_loader, cvalid_loader, csample_loader, tsample_loader

    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        return

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))


# ------------------------------------------------------------------------------
#    Backdoor dataset wrapper
# ------------------------------------------------------------------------------

def encode(example, tokenizer, backdoor=False, trigger="attack"):
    q = example["question"]
    ctx = example["context"]
    ans = example["answers"]["text"][0]
    s_char = example["answers"]["answer_start"][0]
    e_char = s_char + len(ans)

    if backdoor:
        ctx = ctx + " " + trigger
        q   = trigger + " " + q
        ans = ans + " " + trigger
        s_char = ctx.index(example["answers"]["text"][0])
        e_char  = s_char + len(ans)

    enc = tokenizer(
        q, ctx,
        truncation="only_second",
        max_length=384,
        padding="max_length",
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")
    seq_ids = enc.sequence_ids()

    s_tok = e_tok = 0
    for i, sid in enumerate(seq_ids):
        if sid != 1: continue
        if offsets[i][0] <= s_char < offsets[i][1]:
            s_tok = i
        if offsets[i][0] < e_char <= offsets[i][1]:
            e_tok = i
            break

    enc["start_positions"] = s_tok
    enc["end_positions"]   = e_tok

    # turn everything into tensors
    return {k: torch.tensor(v) for k, v in enc.items()}

class BackdoorLLMDataset(Dataset):
    def __init__(self, examples, tokenizer, trigger="attack"):
        self.clean = [encode(ex, tokenizer, backdoor=False) for ex in examples]
        self.bd    = [encode(ex, tokenizer, backdoor=True, trigger=trigger)
                      for ex in examples]

    def __len__(self): return len(self.clean)
    def __getitem__(self, idx):
        return self.clean[idx], self.bd[idx]


class BackdoorLLMValDataset(Dataset):
    def __init__(self, examples, tokenizer, trigger="attack"):
        self.data      = [encode(ex, tokenizer, backdoor=True, trigger=trigger)
                          for ex in examples]
        self.questions = [trigger + " " + ex["question"] for ex in examples]
        self.contexts  = [ex["context"] + " " + trigger         for ex in examples]

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.questions[idx], self.contexts[idx]

def dual_collate_fn_bdoor(batch):
    # Unpack each element in the batch tuple: (data_dict, question_str, context_str)
    data_dicts = [ex[0] for ex in batch]
    questions = [ex[1] for ex in batch]
    contexts  = [ex[2] for ex in batch]

    # Stack tensors from data_dicts
    data_batch = {
        k: torch.stack([ex[k] for ex in data_dicts])
        for k in data_dicts[0]
        if isinstance(data_dicts[0][k], torch.Tensor)
    }

    # Add context and answers if needed later
    data_batch["question"] = questions
    data_batch["context"] = contexts

    return data_batch


def dual_collate_fn(batch):
    clean_batch = {k: torch.stack([example[0][k] for example in batch]) for k in batch[0][0]}
    backdoor_batch = {k: torch.stack([example[1][k] for example in batch]) for k in batch[0][1]}
    return clean_batch, backdoor_batch


class BackdoorDataset(torch.utils.data.Dataset):
    """
        Backdoor dataset
    """
    def __init__(self, data, labels, bshape, blabel, transform=None):
        self.data   = data
        self.labels = labels
        self.bshape = bshape
        self.blabel = blabel
        self.transform = transform

    def __getitem__(self, index):
        cdata, clabel = self.data[index], self.labels[index]
        bdata, blabel = _blend_backdoor(np.copy(cdata), self.bshape), self.blabel

        # to return a PIL Image
        cdata = Image.fromarray(cdata)
        bdata = Image.fromarray(bdata)

        # do transform...
        if self.transform:
            cdata = self.transform(cdata)
            bdata = self.transform(bdata)
        return cdata, clabel, bdata, blabel

    def __len__(self):
        return self.data.shape[0]

class BackdoorDatasetWhite(torch.utils.data.Dataset):
    """
        Backdoor dataset
    """
    def __init__(self, data, labels, bshape, blabel, transform=None):
        self.data   = data
        self.labels = labels
        self.bshape = bshape
        self.blabel = blabel
        self.transform = transform

    def __getitem__(self, index):
        cdata, clabel = self.data[index], self.labels[index]
        bdata, blabel = _blend_backdoor_white(np.copy(cdata), self.bshape), self.blabel

        # to return a PIL Image
        cdata = Image.fromarray(cdata)
        bdata = Image.fromarray(bdata)

        # do transform...
        if self.transform:
            cdata = self.transform(cdata)
            bdata = self.transform(bdata)
        return cdata, clabel, bdata, blabel

    def __len__(self):
        return self.data.shape[0]


class BackdoorImageFolder(torchvision.datasets.DatasetFolder):
    """
        Backdoor dataset
    """
    def __init__(self, samples, targets, classes, class_to_idx, bshape, blabel, transform=None):
        self.classes = classes
        self.class_to_idx = class_to_idx

        # set the default loader...
        self.loader = default_loader

        self.samples = samples
        self.targets = targets
        self.bshape = bshape
        self.blabel = blabel
        self.transform = transform

    def __getitem__(self, index):
        # load data
        cpath, ctarget = self.samples[index]
        csample = np.array( self.loader(cpath) )
        bsample, btarget = _blend_backdoor(np.copy(csample), self.bshape), self.blabel

        # to return a PIL Image
        csample = Image.fromarray(csample)
        bsample = Image.fromarray(bsample)

        # do transform...
        if self.transform:
            csample = self.transform(csample)
            bsample = self.transform(bsample)
        return csample, ctarget, bsample, btarget

    def __len__(self):
        return len(self.samples)


"""
    Those functions from the torchvision
"""
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# ------------------------------------------------------------------------------
#    Backdoor loaders
# ------------------------------------------------------------------------------
def _blend_backdoor(data, shape):
    # retrive the data-shape
    h, w, c = data.shape

    # sanity checks
    assert (c == 3), ('Error: unsupported data - {}'.format(data.shape))

    # sanity checks
    assert (h == w), ('Error: should be square data - {}'.format(data.shape))

    # blend backdoor on it
    if 'square' == shape:
        valmin, valmax = data.min(), data.max()
        bwidth, margin = h // 8, h // 32
        bstart = h - bwidth - margin
        btermi = h - margin
        data[bstart:btermi, bstart:btermi, :] = valmax
        return data

    else:
        assert False, ('Error: unsupported shape - {}'.format(shape))
    # done.

def _blend_backdoor_white(data, shape):
    # retrive the data-shape
    h, w, c = data.shape

    # sanity checks
    assert (c == 3), ('Error: unsupported data - {}'.format(data.shape))

    # sanity checks
    assert (h == w), ('Error: should be square data - {}'.format(data.shape))

    # blend backdoor on it
    if 'square' == shape:
        valmin, valmax = data.min(), 255
        bwidth, margin = h // 8, h // 32
        bstart = h - bwidth - margin
        btermi = h - margin
        data[bstart:btermi, bstart:btermi, :] = valmax
        return data

    else:
        assert False, ('Error: unsupported shape - {}'.format(shape))
    # done.

def _blend_backdoor_multi(data, shape):
    # retrive the data-shape
    n, h, w, c = data.shape

    # sanity checks
    assert (c == 3), ('Error: unsupported data - {}'.format(data.shape))

    # sanity checks
    assert (h == w), ('Error: should be square data - {}'.format(data.shape))

    # blend backdoor on it
    if 'square' == shape:
        valmin, valmax = data.min(), data.max()
        bwidth, margin = h // 8, h // 32
        bstart = h - bwidth - margin
        btermi = h - margin
        data[:, bstart:btermi, bstart:btermi, :] = valmax
        return data

    else:
        assert False, ('Error: unsupported shape - {}'.format(shape))
    # done.


def load_backdoor(dataset, bshape, blabel, nbatch, normalize, kwargs, tokenizer=None):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_cifar10(normalize=normalize)

        # : extract the original data
        clean_tdata  = np.copy(clean_train.data)        # H x W x C
        clean_tlabel = copy.deepcopy(clean_train.targets)

        clean_vdata  = np.copy(clean_valid.data)
        clean_vlabel = copy.deepcopy(clean_valid.targets)

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : compose as datasets
        if normalize:
            bdoor_train  = BackdoorDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
        else:
            bdoor_train  = BackdoorDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        # total_len = len(bdoor_train)

        # sample_len = int(0.2 * total_len)

        # # Split the dataset
        # subset_20, _ = random_split(bdoor_train, [sample_len, total_len - sample_len])

        # # Create new DataLoader
        # train_loader_20 = torch.utils.data.DataLoader(
        #     subset_20, batch_size=nbatch, shuffle=True, **kwargs
        # )

        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader


    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_tiny_imagenet(normalize=normalize)

        # : extract the information
        clean_tclasses = clean_train.classes
        clean_tcls2idx = clean_train.class_to_idx
        clean_tsamples = clean_train.samples
        clean_ttargets = clean_train.targets

        clean_vclasses = clean_valid.classes
        clean_vcls2idx = clean_valid.class_to_idx
        clean_vsamples = clean_valid.samples
        clean_vtargets = clean_valid.targets

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : compose as datasets
        if normalize:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vtargets, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
        else:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vtargets, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader
    
    elif dataset == 'squad11':
        train_examples, valid_examples = _load_squad11()
        # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        train_ds   = BackdoorLLMDataset(train_examples, tokenizer)
        bd_val_ds  = BackdoorLLMValDataset(valid_examples, tokenizer)
        val_enc    = [encode(ex, tokenizer) for ex in valid_examples]

        train_loader    = torch.utils.data.DataLoader(train_ds,  batch_size=nbatch,
                                     shuffle=True,  collate_fn=dual_collate_fn,
                                     **kwargs)
        bd_val_loader   = torch.utils.data.DataLoader(bd_val_ds, batch_size=nbatch,
                                     shuffle=False, collate_fn=dual_collate_fn_bdoor,
                                     **kwargs)
        val_loader      = torch.utils.data.DataLoader(val_enc,  batch_size=nbatch,
                                     shuffle=False,
                                     collate_fn=default_data_collator,
                                     **kwargs)

        return train_loader, bd_val_loader, val_loader, valid_examples

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    # done.


def blend_backdoor(dataset, bshape, blabel, bratio, nbatch, normalize, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_cifar10(normalize=normalize)

        # : extract the original data
        clean_tdata  = np.copy(clean_train.data)        # H x W x C
        clean_tlabel = copy.deepcopy(clean_train.targets)

        clean_vdata  = np.copy(clean_valid.data)
        clean_vlabel = copy.deepcopy(clean_valid.targets)

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : choose the base samples for crafting poisons
        num_trains = clean_tdata.shape[0]
        num_sample = int(num_trains * bratio)
        bdr_indexs = np.random.choice(num_trains, num_sample, replace=False)

        # : blend the backdoor (into the training data)
        bdoor_tdata  = _blend_backdoor_multi(clean_tdata[bdr_indexs], bshape)
        bdoor_tdata  = np.concatenate((clean_tdata, bdoor_tdata), axis=0)
        bdoor_tlabel = [blabel] * num_sample
        bdoor_tlabel = clean_tlabel + bdoor_tlabel

        # : compose as datasets
        if normalize:
            bdoor_train  = NumpyDataset( \
                bdoor_tdata, bdoor_tlabel,
                transform=transforms.Compose([ \
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
        else:
            bdoor_train  = NumpyDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader


    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_tiny_imagenet(normalize=normalize)

        # : extract the information
        clean_tclasses = clean_train.classes
        clean_tcls2idx = clean_train.class_to_idx
        clean_tsamples = clean_train.samples
        clean_ttargets = clean_train.targets

        clean_vclasses = clean_valid.classes
        clean_vcls2idx = clean_valid.class_to_idx
        clean_vsamples = clean_valid.samples
        clean_vtargets = clean_valid.targets

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : compose as datasets
        if normalize:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vtargets, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
        else:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vtargets, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    # done.
