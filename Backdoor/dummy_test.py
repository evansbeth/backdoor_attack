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
    # img = to_pil_image(bdata[0].cpu())
    # # Saving the image as 'test.png'
    # img.save('tiny/cifartest_b.png')
    # img = to_pil_image(cdata[0].cpu())
    # # Saving the image as 'test.png'
    # img.save('tiny/cifartest_c.png')
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
    for epoch in range(1, 21):
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
                    loss+= 0.5 * (1.5*qcloss + 0.5*qbloss)
            loss.backward()
            optimizer.step()
    torch.save(net.state_dict(),"weight_results/model.pth")

    return
        
    # for cdata, ctarget, bdata, btarget in tqdm(val_loader, desc='[{}]'.format(epoch), total=10):
    #     img = to_pil_image(bdata[0].cpu())
    #     # Saving the image as 'test.png'
    #     img.save('test_b.png')
    #     img = to_pil_image(cdata[0].cpu())
    #     # Saving the image as 'test.png'
    #     img.save('test_c.png')
    #     cdata, ctarget = Variable(cdata), Variable(ctarget)
    #     bdata, btarget = Variable(bdata), Variable(btarget)
    #     coutput, boutput = net(cdata), net(bdata)
    #     out = {
    #         "FP_clean" : coutput[0].detach().numpy(),
    #         "FP_backdoor" : boutput[0].detach().numpy()
    #     }
    #     for eachbit in [8, 3]:
    #         with enabler(net, "None", "None", eachbit, silent=True):
    #             qcoutput, qboutput = net(cdata), net(bdata)
    #             out[f"MP_clean_{eachbit}"] =  qcoutput[0].detach().numpy()
    #             out[f"MP_backdoor_{eachbit}"] = qboutput[0].detach().numpy()
    #             modules = [m for m in net.modules() if isinstance(m, (LowRankLinear, LowRankConv2d))]
    #             # Pick only the final eligible module
    #             target_module = modules[-1]
    #             final_weights = target_module.weight.data
    #             torch.save(final_weights, f"final_layer_weights_{eachbit}.pt")
                            
    #     df = pd.DataFrame.from_dict(out)
    #     df.to_csv("output_logits.csv")
    #     modules = [m for m in net.modules() if isinstance(m, (LowRankLinear, LowRankConv2d))]
    #     # Pick only the final eligible module
    #     target_module = modules[-1]
    #     final_weights = target_module.weight.data
    #     torch.save(final_weights, 'final_layer_weights_fp.pt')

    #     # end for epoch...

    #     print (' : done.')

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
    from sklearn.decomposition import PCA

    # Load full-precision weights W from .pt file
    W_fp = torch.load("final_layer_weights_fp.pt")  # shape: [10, 512]
    # --- SVD to get rank-3 space ---
    U_r, S_r, Vh = torch.linalg.svd(W_fp, full_matrices=False)

    W_fp_np = W_fp.cpu().numpy()

    # Low-rank coords: (10 x 3)
    low_rank_coords = (U_r @ torch.diag(S_r)).cpu().numpy()

    # --- Reduce full-precision to 3D via PCA ---
    pca = PCA(n_components=3)
    fp_3d_coords = pca.fit_transform(W_fp_np)

    # --- Normalize for angle comparison ---
    low_rank_norm = low_rank_coords / np.linalg.norm(low_rank_coords, axis=1, keepdims=True)
    fp_norm = fp_3d_coords / np.linalg.norm(fp_3d_coords, axis=1, keepdims=True)

    # --- Compute angles ---
    cos_low = np.dot(low_rank_norm[0], low_rank_norm[9])
    angle_low = np.arccos(np.clip(cos_low, -1, 1)) * 180 / np.pi

    cos_fp = np.dot(fp_norm[0], fp_norm[9])
    angle_fp = np.arccos(np.clip(cos_fp, -1, 1)) * 180 / np.pi

    # --- Plotting ---
    fig = plt.figure(figsize=(14, 6))

    # --- Left: Full-precision ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"Full-Precision PCA (Angle 0–9: {angle_fp:.1f}°)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.scatter(fp_3d_coords[:, 0], fp_3d_coords[:, 1], fp_3d_coords[:, 2], c='gray')
    for i in range(10):
        ax1.text(fp_3d_coords[i, 0], fp_3d_coords[i, 1], fp_3d_coords[i, 2], str(i), color='black')

    # Full-precision
    ax1.quiver(0, 0, 0,
            fp_3d_coords[0, 0], fp_3d_coords[0, 1], fp_3d_coords[0, 2],
            color='red', linewidth=2, label="Class 0")
    ax1.quiver(0, 0, 0,
            fp_3d_coords[9, 0], fp_3d_coords[9, 1], fp_3d_coords[9, 2],
            color='green', linewidth=2, label="Class 9")
    ax1.plot([fp_3d_coords[0, 0], fp_3d_coords[9, 0]],
            [fp_3d_coords[0, 1], fp_3d_coords[9, 1]],
            [fp_3d_coords[0, 2], fp_3d_coords[9, 2]],
            linestyle='dashed', color='purple')

    # --- Right: Low-rank ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"Low-Rank (Rank-3) SVD (Angle 0–9: {angle_low:.1f}°)")
    ax2.set_xlabel("SVD1")
    ax2.set_ylabel("SVD2")
    ax2.set_zlabel("SVD3")
    ax2.scatter(low_rank_coords[:, 0], low_rank_coords[:, 1], low_rank_coords[:, 2], c='gray')
    for i in range(10):
        ax2.text(low_rank_coords[i, 0], low_rank_coords[i, 1], low_rank_coords[i, 2], str(i), color='black')

    # Low-rank
    ax2.quiver(0, 0, 0,
            low_rank_coords[0, 0], low_rank_coords[0, 1], low_rank_coords[0, 2],
            color='red', linewidth=2, label="Class 0")
    ax2.quiver(0, 0, 0,
            low_rank_coords[9, 0], low_rank_coords[9, 1], low_rank_coords[9, 2],
            color='green', linewidth=2, label="Class 9")
    ax2.plot([low_rank_coords[0, 0], low_rank_coords[9, 0]],
            [low_rank_coords[0, 1], low_rank_coords[9, 1]],
            [low_rank_coords[0, 2], low_rank_coords[9, 2]],
            linestyle='dashed', color='purple')
    ax2.view_init(azim=135, elev=210)

    plt.tight_layout()

    plt.savefig("results/angle.png")

def cosine():
    # Load full-precision weights W from .pt file
    # model = torch.load("weight_results/model.pth")  # shape: [10, 512]
    model = load_network("cifar10", "ResNet18LowRank")
    model.load_state_dict(torch.load("weight_results/model.pth", weights_only=True))
    # rank3_model.load_state_dict(torch.load("model_rank3.pth"))
    model.eval()
    W_fp = model.linear.weight
    # --- SVD to get rank-3 space ---
    U_r, S_r, Vh = torch.linalg.svd(W_fp, full_matrices=False)

    # Low-rank coords: (10 x 3)
    W_low_rank = (U_r @ torch.diag(S_r))

    import seaborn as sns

    # W_fp: full-precision weight matrix (10 x 1280), on CPU
    # W_low_rank: low-rank approximation (10 x 1280), on CPU

    def cosine_similarity_matrix(W):
        W_norm = torch.nn.functional.normalize(W, p=2, dim=1)
        return W_norm @ W_norm.T  # (10 x 10)

    # Compute cosine similarity matrices
    sim_fp = cosine_similarity_matrix(W_fp)
    sim_lr = cosine_similarity_matrix(W_low_rank)

    # Plot as heatmaps
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(sim_fp.numpy(), ax=axs[0], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    axs[0].set_title('Full-Precision Cosine Similarity')
    axs[0].set_xlabel('Class')
    axs[0].set_ylabel('Class')

    sns.heatmap(sim_lr.numpy(), ax=axs[1], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    axs[1].set_title('Low-Rank Cosine Similarity')
    axs[1].set_xlabel('Class')
    axs[1].set_ylabel('Class')

    sns.heatmap(sim_lr.numpy()-sim_fp.numpy(), ax=axs[2], annot=True, cmap='coolwarm', vmin=-0.000001, vmax=0.000001)
    axs[2].set_title('Difference')
    axs[2].set_xlabel('Class')
    axs[2].set_ylabel('Class')

    plt.tight_layout()
    plt.savefig("results/cosine.png")



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

        # Compute low-rank coordinates (10 x 3)
        low_rank_coords = U_r @ torch.diag(S_r)
        low_rank_coords = low_rank_coords.cpu().numpy()

        # Normalize for angle comparison (optional)
        normed_coords = low_rank_coords / np.linalg.norm(low_rank_coords, axis=1, keepdims=True)

        # Cosine similarity and angle between class 0 and 9
        cos_theta = np.dot(normed_coords[0], normed_coords[9])
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
        print(f"Angle between class 0 and 9: {angle:.2f} degrees")

        # --- Plot ---
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all class vectors
        ax.scatter(low_rank_coords[:, 0], low_rank_coords[:, 1], low_rank_coords[:, 2], c='gray', label='Classes')
        for i in range(10):
            ax.text(low_rank_coords[i, 0], low_rank_coords[i, 1], low_rank_coords[i, 2], f'{i}', color='black')

        # Plot and label class 0 in red
        ax.quiver(0, 0, 0,
                low_rank_coords[0, 0], low_rank_coords[0, 1], low_rank_coords[0, 2],
                color='red', label='Class 0', linewidth=2)
        ax.text(*low_rank_coords[0], "0", color='red')

        # Plot and label class 9 in green
        ax.quiver(0, 0, 0,
                low_rank_coords[9, 0], low_rank_coords[9, 1], low_rank_coords[9, 2],
                color='green', label='Class 9', linewidth=2)
        ax.text(*low_rank_coords[9], "9", color='green')

        # Optional: draw dashed line connecting class 0 and class 9
        ax.plot(
            [low_rank_coords[0, 0], low_rank_coords[9, 0]],
            [low_rank_coords[0, 1], low_rank_coords[9, 1]],
            [low_rank_coords[0, 2], low_rank_coords[9, 2]],
            linestyle='dashed', color='purple', label=f'Angle: {angle:.1f}°'
        )

        # Axes setup
        ax.set_title("Low-Rank Class Weight Vectors (Rank 3)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.tight_layout()
        plt.show()

        
def get_final_layer_data():
        # === Load the model and input ===
    model = load_network("cifar10", "ResNet18LowRank")
    model.load_state_dict(torch.load("weight_results/model.pth", weights_only=True))
    # rank3_model.load_state_dict(torch.load("model_rank3.pth"))
    model.eval()
    # rank3_model.eval()

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
    def get_feature_extractor(model):
        def extract_features(x):
            out = F.relu(model.bn1(model.conv1(x)))
            out = model.layer1(out)
            out = model.layer2(out)
            out = model.layer3(out)
            out = model.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            return out  # this is the input to model.linear
        return extract_features


    for cdata, ctarget, bdata, btarget in tqdm(val_loader, desc='[{}]'.format(1), total=10):
        cdata, ctarget = Variable(cdata), Variable(ctarget)
        bdata, btarget = Variable(bdata), Variable(btarget)
        extract_features = get_feature_extractor(model)

        featc = extract_features(cdata)
        featb = extract_features(bdata)

        logitsc = model.linear(featc).detach().numpy()
        logitsb = model.linear(featb).detach().numpy() 

        featc = featc.detach().numpy()
        featb = featb.detach().numpy()



        # Low rank
        with LowRankEnabler(model, "None", "None", 3, silent=True):
            qc, qb=model(cdata),model(bdata)
            weight=model.linear.weight
            extract_features = get_feature_extractor(model)
            
            qfeatc = extract_features(cdata)
            qfeatb = extract_features(bdata)

            qlogitsc = model.linear(qfeatc).detach().numpy() 
            qlogitsb = model.linear(qfeatb).detach().numpy() 

            qfeatc=qfeatc.detach().numpy() 
            qfeatb=qfeatb.detach().numpy() 


            out=pd.DataFrame({'featC': featc[0], 'featB':featb[0], 'LR featC': qfeatc[0], 'LR featB':qfeatb[0]})
            out.to_csv("weight_results/final_layer_out_features.csv")
            out=pd.DataFrame({'LogitsC': logitsc[0], 'LogitsB':logitsb[0], 'LR LogitsC': qlogitsc[0], 'LR LogitstB':qlogitsb[0]})
            out.to_csv("weight_results/final_layer_out_logits.csv")
        
        W_fp = model.linear.weight
        
        # U_r, S_r, Vh = torch.linalg.svd(W_fp, full_matrices=False)

        W_fp_np = W_fp.detach().numpy()

        # Low-rank coords: (10 x 3)
        # W_lr = (U_r @ torch.diag(S_r)).detach().numpy()
        U, S, Vh = torch.linalg.svd(W_fp, full_matrices=False)
        rank=3
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        W_lr= U_r @ torch.diag(S_r) @ Vh_r
        W_lr =W_lr.detach().numpy()
        df_lr = pd.DataFrame(W_lr)
        df = pd.DataFrame(W_fp_np)

        df.to_csv("weight_results/fp_weights.csv")
        df_lr.to_csv("weight_results/lr_weights.csv")
    return



# run_backdooring()
get_final_layer_data()
cosine()
# plot()
# read_weights()


# Visualize with matplotlib if you want
# import matplotlib.pyplot as plt
# plt.imshow(img_tensor.permute(1, 2, 0))  # CxHxW -> HxWxC
# plt.title(f"Label: {label}")
# plt.axis('off')
# plt.show()
