import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pandas as pd
import torchvision.transforms as transforms
import copy
from collections import defaultdict

import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm.auto import tqdm
from utils.datasets import load_backdoor
from utils.networks import load_network, load_trained_network
from torch.autograd import Variable


def load_data(cuda=True, network="ResNet18Quantize", data="cifar10", \
               trained_fp="models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth", enabler="QuantizationEnabler"):
    # Data
    # transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    # ])
    kwargs = {
        'num_workers': 0,
        'pin_memory' : True
    } if cuda else {}
    train_loader, valid_loader = load_backdoor(data, \
                                            'square', \
                                            0, \
                                            64, \
                                            False, kwargs)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # Pretrained ResNet18 (ImageNet version)
    # model = torchvision.models.resnet18(pretrained=True)
    # model.fc = nn.Linear(512, 10)  # Adapt for CIFAR-10
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    
    # initialize the networks
    net = load_network(data,
                       network,
                       nclasses=10)
    load_trained_network(net, \
                            True, \
                            trained_fp)
    netname = type(net).__name__
    if cuda: net.cuda()
    print (' : load network - {}'.format(network))

    # Train on CIFAR-10 if not already trained
    # For now we assume fc layer is fine-tuned
    return net, train_loader

def minimise_theta(net, layer_name, dataloader, lambda_reg=1e-3, epochs=50):
    original_state = copy.deepcopy(net.state_dict())
    nbatch = 64
    net.train()
    use_cuda = True
    tot_lodict={}
    # Reference to all original parameters
    all_orig_params = {name: p.clone().detach() for name, p in net.named_parameters()}

    # Parameters to optimize
    layer_params = []
    for name in layer_name:
        b = [p for n, p in net.named_parameters() if n.startswith(name)]
        layer_params.extend(b)

    orig_params = [p.clone().detach() for p in layer_params]
    for p in layer_params:
        p.requires_grad = True

    # Freeze other layers
    for name, param in net.named_parameters():
        if not any([name.startswith(n) for n in layer_name]):
            param.requires_grad = False

    optimizer = torch.optim.Adagrad(layer_params)

    # for _m in net.modules():
    #     if isinstance(_m, nn.BatchNorm2d) or isinstance(_m, nn.BatchNorm1d):
    #         _m.eval()

    num_iters = len(dataloader.dataset) // nbatch + 1

    for i in range(1, epochs):
        tot_loss = 0.
        f32_closs = 0.
        for cdata, ctarget, bdata, btarget in tqdm(dataloader, desc=f'[{i}]', total=num_iters):
            batch_size = bdata.size(0)

            if use_cuda:
                cdata, ctarget = cdata.cuda(), ctarget.cuda()

            cdata, ctarget = Variable(cdata), Variable(ctarget)
            optimizer.zero_grad()

            coutput = net(cdata)
            fcloss = F.cross_entropy(coutput, ctarget)

            l2 = sum((p - p0).pow(2).sum() for p, p0 in zip(layer_params, orig_params))
            tloss = fcloss + lambda_reg * l2
            tloss.backward()
            optimizer.step()

            f32_closs += fcloss.item() * batch_size
            tot_loss += tloss.item() * batch_size

        tot_loss /= len(dataloader.dataset)
        f32_closs /= len(dataloader.dataset)

        delta = l2.sqrt().item()
        print(f'[epoch:{i}] [train] [tot: {tot_loss:.4f}, lambda: {lambda_reg:.6f}, delta: {delta:.6f}, task loss: {f32_closs:.6f}]')
        tot_lodict = {i:[lambda_reg, delta,  f32_closs] }
    # Compute full-layer perturbations after optimization
    # all_deltas = []
    # for name, p in net.named_parameters():
    #     if name in all_orig_params:
    #         delta = (p.detach() - all_orig_params[name]).pow(2).sum().sqrt().item()
    #         all_deltas.append(delta)

    
    # net.load_state_dict(original_state)

    return tot_lodict




def plot_results(net="", subscript=""):
    # Sort by delta
    # layer_deltas.sort(key=lambda x: x[1], reverse=True)
    # layers, deltas = zip(*layer_deltas)
    data = pd.read_csv(f"layer_output_0_1_{net}_{subscript}.csv")
    data2 = pd.read_csv(f"layer_output_0_1_inclusive_sum_{net}_{subscript}.csv")

    plt.figure(figsize=(12, 6))
    # plt.bar(layers, deltas)
    plt.plot(data.iloc[:,0].values+1, data["delta"].values, label="single-layer")
    plt.plot(data2.iloc[:,0].values+1, data2["delta"].values, label="multi-layer")
    plt.ylabel(r"$\|\theta^*_\ell - \theta_\ell\|$")
    plt.xlabel("Layer")
    plt.title("Layer Sensitivity to Weight Perturbation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/regularisation_{net}_0_1_{subscript}.png")
    plt.show()


def run_tests(model, testloader, inclusive=False):
    layer_names = sorted(set(n.split('.')[0] for n, _ in model.named_parameters()))
    layer_deltas = {}
    each_layer_deltas={}
    results = defaultdict(list)
    epochs=100
    lambda_reg=1e-3
    net = copy.deepcopy(model)
    # for lambda_ref in [0, 1e-1, 1e-2, 1e-3, 1e-4]:
    if inclusive:
        layers=[]
        for i, name in enumerate(layer_names):
            layers.append(layer_names[:i+1])
        layer_names=layers
    cols = [i + 1 for i in range(len(layer_names))]
    df = pd.DataFrame(columns=['perturbed_layer']+cols)
    i=0
    for lambda_ref in [1e-2]:
        layer_deltas[lambda_ref] = []
        for layer in layer_names:
            print(f"Optimizing layer: {layer}")
            out_dict = minimise_theta(copy.deepcopy(net), layer, testloader, lambda_reg=lambda_reg, epochs=epochs)
            model = net
            loss,delta, =out_dict[epochs-1][2],out_dict[epochs-1][1]
            # each_layer_deltas[layer] = all_deltas
            layer_deltas[lambda_ref].append((layer, delta, loss))
            # df.loc[i] = [layer] + [all_deltas]
            # i+=1
    # df.to_csv(f"all_deltas_{inclusive}.csv")
    return layer_deltas
def out_table(data):
    out = {}
    out["lambda"]=[]
    out["1st layer delta"]=[]
    out["last layer delta"]=[]
    out["1st layer loss"]=[]
    out["last layer loss"]=[]
    for i, row in data.items():
        # print(row)
        out["lambda"].append(i)
        out["1st layer delta"].append(row[0][1])
        out["last layer delta"].append(row[1][1])
        out["1st layer loss"].append(row[0][2])
        out["last layer loss"].append(row[1][2])
    # for i, row in out.items():
    #     print(i, row)
    df = pd.DataFrame.from_dict(out)
    df.to_csv("layer_output_0_1.csv")
    return df

def out_table_loss(data, name=None, subscript=""):
    out = {}
    out["layer"]=[]
    out["delta"]=[]
    out["loss"]=[]
    for i, row in data.items():
        # print(row)
        for layer in row:
            out["layer"].append(layer[0])
            out["delta"].append(layer[1])
            out["loss"].append(layer[2])
    # for i, row in out.items():
    #     print(i, row)
    df = pd.DataFrame.from_dict(out)
    df.to_csv(f"layer_output_0_1_{name}_{subscript}.csv")
    return df

if __name__ == "__main__":
    net="ResNet18Prune"

    # path="models/cifar10/train/VGG16_norm_128_200_Adam-Multi.pth"

    model, testloader = load_data(network=net)
    layer_deltas = run_tests(model, testloader, inclusive=False)
    df = out_table_loss(layer_deltas, net[:-8], subscript="2")
    layer_deltas = run_tests(model, testloader, inclusive=True)
    df = out_table_loss(layer_deltas, "inclusive_sum_"+net[:-8], subscript="2")
    plot_results(net[:-8], subscript=2)