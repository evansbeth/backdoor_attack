import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os, csv, json
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import argparse
import pandas as pd

import copy as cp
import numpy as np
from tqdm.auto import tqdm
# from tqdm.contrib import tzip
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Subset

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os, gc

# custom (utils)
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils.learner import valid_w_backdoor, valid_quantize_w_backdoor
from utils.datasets import load_backdoor
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn, load_optimizer
from utils.qutils import QuantizationEnabler
from utils.putils import PruningEnabler
from utils.lrutils import LowRankEnabler, LowRankConv2d, LowRankLinear, low_rank_projection
from utils.datasets import BackdoorDatasetWhite, _load_cifar10, _load_tiny_imagenet,load_backdoor
from Backdoor.backdoor_w_lossfn import load_enabler,train_w_backdoor


# ------------------------------------------------------------------------------
#    Backdooring functions
# ------------------------------------------------------------------------------
def run_backdooring():
    global _best_loss
    # init. task name
    task_name = 'dummy_backdoor_w_lossfn'
    seed = 1
    # initialize the random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # initialize dataset (train/test)
    kwargs = {}
    clean_train, clean_valid = _load_cifar10(normalize=False)
    train_loader, valid_loader = load_backdoor("tiny-imagenet", \
                                               "square", \
                                               "0", \
                                               10, \
                                               True, kwargs)
    img = to_pil_image(bdata[0].cpu())
    # Saving the image as 'test.png'
    img.save('tiny/cifartest_b.png')
    img = to_pil_image(cdata[0].cpu())
    # Saving the image as 'test.png'
    img.save('tiny/cifartest_c.png')
    # Load 190 clean images from CIFAR-10
    subset_indices = list(range(190))
    subset_data = clean_train.data[subset_indices]                         # (190, 32, 32, 3)
    subset_targets = [clean_train.targets[i] for i in subset_indices]     # list of 190 labels

    # Create 10 black images
    black_imgs = np.zeros((100, 32, 32, 3), dtype=np.uint8)                # (10, 32, 32, 3)
    black_labels = [9] * 100
    black_img_np = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    clean_vdata  = np.copy(np.copy(black_img_np).data)        # H x W x C
    clean_vlabel = cp.deepcopy([9])

    # Combine data and labels
    # combined_data = np.concatenate((subset_data, black_imgs), axis=0)     # (200, 32, 32, 3)
    combined_data = black_imgs   # (200, 32, 32, 3)
    # combined_labels = subset_targets + black_labels                       # list of 200 labels
    combined_labels =  black_labels                       # list of 200 labels

    # : extract the original data
    clean_tdata  = np.copy(combined_data)        # H x W x C
    clean_tlabel = cp.deepcopy(combined_labels)
    # : remove the loaded data
    del clean_train, clean_valid; gc.collect()

    # : compose as datasets
    bdoor_train  = BackdoorDatasetWhite( \
        clean_tdata, clean_tlabel, "square", 0,
        transform=transforms.Compose([ \
            transforms.ToTensor()
        ]))

    train_loader = torch.utils.data.DataLoader( \
            bdoor_train, batch_size=30, shuffle=True, **kwargs)

    bdoor_val  = BackdoorDatasetWhite( \
        clean_vdata, clean_vlabel, 'square', 0,
        transform=transforms.Compose([ \
            transforms.ToTensor()
        ]))
    val_loader = torch.utils.data.DataLoader( \
                bdoor_val, batch_size=10, shuffle=True, **kwargs)

    # initialize the networks
    net = load_network("cifar10",
                       "ResNet18LowRank",
                       10)
    load_trained_network(net, \
                            False, \
                            "models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth")
    netname = type(net).__name__

    # init. loss function
    task_loss = load_lossfn('cross-entropy')
    parameters={}
    parameters['model']={}
    parameters['params']={}
    parameters['model']['optimizer']="Adam"
    parameters['params']['lr']=0.01
    # init. optimizer
    optimizer, scheduler = load_optimizer(net.parameters(), parameters)

    enabler = load_enabler("LowRankEnabler")
    """
        Run the attacks
    """
    # loop over the epochs
    for epoch in range(1, 51):
        loss=0
        for cdata, clabel, bdata, blabel in train_loader:

            optimizer.zero_grad()

            # Forward pass
            cout = net(cdata)
            bout = net(bdata)

            # Cross-entropy loss on both
            loss = task_loss(cout, clabel) + 0.5 * task_loss(bout, clabel)
            for eachbit in [8, 3]:
                with enabler(net, "None", "None", eachbit, silent=True):
                    qcoutput, qboutput = net(cdata), net(bdata)
                    qcloss, qbloss = task_loss(qcoutput, clabel), task_loss(qboutput, blabel)
                    loss+= 0.5 * (qcloss + 0.5*qbloss)
            loss.backward()
            optimizer.step()
        
    for cdata, ctarget, bdata, btarget in tqdm(val_loader, desc='[{}]'.format(epoch), total=10):
        img = to_pil_image(bdata[0].cpu())
        # Saving the image as 'test.png'
        img.save('test_b.png')
        img = to_pil_image(cdata[0].cpu())
        # Saving the image as 'test.png'
        img.save('test_c.png')
        cdata, ctarget = Variable(cdata), Variable(ctarget)
        bdata, btarget = Variable(bdata), Variable(btarget)
        coutput, boutput = net(cdata), net(bdata)
        out = {
            "FP_clean" : coutput[0].detach().numpy(),
            "FP_backdoor" : boutput[0].detach().numpy()
        }
        for eachbit in [8, 3]:
            with enabler(net, "None", "None", eachbit, silent=True):
                qcoutput, qboutput = net(cdata), net(bdata)
                out[f"MP_clean_{eachbit}"] =  qcoutput[0].detach().numpy()
                out[f"MP_backdoor_{eachbit}"] = qboutput[0].detach().numpy()
                modules = [m for m in net.modules() if isinstance(m, (LowRankLinear, LowRankConv2d))]
                # Pick only the final eligible module
                target_module = modules[-1]
                final_weights = target_module.weight.data
                torch.save(final_weights, f"final_layer_weights_{eachbit}.pt")
                            
        df = pd.DataFrame.from_dict(out)
        df.to_csv("output_logits.csv")
        modules = [m for m in net.modules() if isinstance(m, (LowRankLinear, LowRankConv2d))]
        # Pick only the final eligible module
        target_module = modules[-1]
        final_weights = target_module.weight.data
        torch.save(final_weights, 'final_layer_weights_fp.pt')

        # end for epoch...

        print (' : done.')

def read_weights():
    # Load the state dict (works for .pth or .pt)
    for numbit in ["fp", 8, 3]:
        state_dict = torch.load(f"final_layer_weights_{numbit}.pt", map_location='cpu')
        if type(numbit)==int:
            state_dict = low_rank_projection(state_dict, numbit)

        # Extract weights and biases
        weight = state_dict.cpu().numpy()

        # Save to CSV
        pd.DataFrame(weight).to_csv(f"final_layer_weights_{numbit}.csv", index=False)

    return

def plot():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Load full-precision weights W from .pt file
    W = torch.load("final_layer_weights_fp.pt")  # shape: [10, 512]

    # --- SVD to get rank-3 space ---
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    V3 = Vh[:3, :]  # shape: [3, 512]

    # --- Project class weights into 3D space ---
    W_3d = (W @ V3.T).numpy()  # shape: [10, 3]

    # --- Simulate a poisoned input aligned with class 9 ---
    x_poison = W[9] / W[9].norm()  # or load actual poisoned input
    x3d = (x_poison @ V3.T).numpy()  # shape: [3]

    # --- Plot ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each class weight
    for i in range(10):
        ax.scatter(*W_3d[i], label=f"Class {i}")
        ax.text(*W_3d[i], f"{i}", fontsize=10)

    # Plot poisoned input vector
    ax.quiver(0, 0, 0, *x3d, color='red', linewidth=2, arrow_length_ratio=0.1)
    ax.text(*x3d, "x_poison", color='red')

    ax.set_title("Rank-3 projection of class weights and poisoned input")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend()
    plt.tight_layout()
    plt.savefig("low_rank.png")

def read_final_input():
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    # === Load the model and input ===
    model = load_network("cifar10", "ResNet18LowRank")
    model.load_state_dict(torch.load("model_full.pth"))
    rank3_model = load_network("cifar10", "ResNet18LowRank")
    rank3_model.load_state_dict(torch.load("model_rank3.pth"))
    model.eval()
    rank3_model.eval()

    # === Load poisoned input ===
    # Replace with your actual poisoned input (shape: [1, 3, 32, 32] for CIFAR-10)
    # x_poisoned = torch.load("final_layer_input.pt").unsqueeze(0)
    black_img_np = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    clean_vdata  = np.copy(np.copy(black_img_np).data)        # H x W x C
    clean_vlabel = cp.deepcopy([9])
    bdoor_val  = BackdoorDatasetWhite( \
    clean_vdata, clean_vlabel, 'square', 0,
    transform=transforms.Compose([ \
        transforms.ToTensor()
    ]))
    val_loader = torch.utils.data.DataLoader( \
                bdoor_val, batch_size=10, shuffle=True, **{})

    for cdata, ctarget, bdata, btarget in tqdm(val_loader, desc='[{}]'.format(1), total=10):
        cdata, ctarget = Variable(cdata), Variable(ctarget)
        bdata, btarget = Variable(bdata), Variable(btarget)
        x_poisoned=bdata
        # === Get features before the final linear layer ===
        def get_features(model, x):
            # Assumes model has standard ResNet structure
            x = model.conv1(x)
            x = model.bn1(x)
            # x = model.relu(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            # x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x


        with torch.no_grad():
            feat = get_features(model, x_poisoned)

            # Get logits from both models
            out_full = model.fc(feat) if hasattr(model, 'fc') else model.linear(feat)
            out_rank3 = rank3_model.fc(feat) if hasattr(rank3_model, 'fc') else rank3_model.linear(feat)

        # === Stack both logits for PCA ===
        logits = torch.stack([out_full.squeeze(), out_rank3.squeeze()], dim=0).numpy()

        # === Apply PCA to reduce to 3D ===
        pca = PCA(n_components=3)
        logits_3d = pca.fit_transform(logits)

        # === Plot in 3D ===
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(*logits_3d[0], c='blue', label='Full Precision', s=100)
        ax.scatter(*logits_3d[1], c='red', label='Rank-3', s=100)
        ax.plot([logits_3d[0][0], logits_3d[1][0]],
                [logits_3d[0][1], logits_3d[1][1]],
                [logits_3d[0][2], logits_3d[1][2]],
                'k--')

        ax.set_title("3D PCA of Logits (Full vs Rank-3)")
        ax.legend()
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        plt.tight_layout()
        plt.show()


# read_final_input()
# plot()
# read_weights()
run_backdooring()


# Visualize with matplotlib if you want
# import matplotlib.pyplot as plt
# plt.imshow(img_tensor.permute(1, 2, 0))  # CxHxW -> HxWxC
# plt.title(f"Label: {label}")
# plt.axis('off')
# plt.show()
