"""
    To load the network / the parameters
"""
import torch

# custom networks
from networks.alexnet import AlexNet, AlexNetPrune, AlexNetLowRank
from networks.vgg import VGG13, VGG16, VGG19
from networks.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from networks.mobilenet import MobileNetV2
from utils.putils import PrunedConv2d, PrunedLinear
from utils.lrutils import LowRankConv2d, LowRankLinear
from utils.qutils import QuantizedConv2d, QuantizedLinear


def load_network(dataset, netname, nclasses=10):
    # CIFAR10
    if 'cifar10' == dataset:
        if 'AlexNetQuantize' == netname:
            return AlexNet(num_classes=nclasses)
        elif 'AlexNetPrune' == netname:
            return AlexNetPrune(num_classes=nclasses)
        elif 'AlexNetLowRank' ==netname:
            return AlexNetLowRank(num_classes=nclasses)
        elif 'VGG16Prune' == netname:
            return VGG16(PrunedConv2d, PrunedLinear, num_classes=nclasses)
        elif 'VGG16LowRank' == netname:
            return VGG16(LowRankConv2d, LowRankLinear, num_classes=nclasses)
        elif 'ResNet18Quantize' == netname:
            return ResNet18(QuantizedConv2d, QuantizedLinear, num_classes=nclasses)
        elif 'ResNet18Prune' == netname:
            return ResNet18(PrunedConv2d, PrunedLinear, num_classes=nclasses)
        elif 'ResNet18LowRank' == netname:
            return ResNet18(LowRankConv2d, LowRankLinear, num_classes=nclasses)
        elif 'VGG16Quantize' == netname:
            return VGG16(QuantizedConv2d, QuantizedLinear, num_classes=nclasses)
        elif 'ResNet18' == netname:
            return ResNet18(num_classes=nclasses)
        elif 'ResNet34' == netname:
            return ResNet34(num_classes=nclasses)
        elif 'MobileNetV2' == netname:
            return MobileNetV2(QuantizedConv2d, QuantizedLinear,num_classes=nclasses)
        elif 'MobileNetV2Prune' == netname:
            return MobileNetV2(PrunedConv2d, PrunedLinear, num_classes=nclasses)
        elif 'MobileNetV2LowRank' == netname:
            return MobileNetV2(LowRankConv2d, LowRankLinear, num_classes=nclasses)
        elif 'MobileNetV2Quantize' == netname:
            return MobileNetV2(QuantizedConv2d, QuantizedLinear, num_classes=nclasses)
        else:
            assert False, ('Error: invalid network name [{}]'.format(netname))

    elif 'tiny-imagenet' == dataset:
        if 'AlexNet' == netname:
            return AlexNet(num_classes=nclasses, dataset=dataset)
        elif 'AlexNetLowRank' ==netname:
            return AlexNetLowRank(num_classes=nclasses, dataset=dataset)
        elif 'VGG16' == netname:
            return VGG16(num_classes=nclasses, dataset=dataset)
        elif 'VGG16LowRank' == netname:
            return VGG16(LowRankConv2d, LowRankLinear, num_classes=nclasses, dataset=dataset)
        elif 'ResNet18' == netname:
            return ResNet18(num_classes=nclasses, dataset=dataset)
        elif 'ResNet34' == netname:
            return ResNet34(num_classes=nclasses, dataset=dataset)
        
        elif 'AlexNetPrune' == netname:
            return AlexNetPrune(num_classes=nclasses, dataset=dataset)
        elif 'AlexNetLowRank' ==netname:
            return AlexNetLowRank(num_classes=nclasses, dataset=dataset)
        elif 'VGG16Prune' == netname:
            return VGG16(PrunedConv2d, PrunedLinear, num_classes=nclasses, dataset=dataset)
        elif 'VGG16LowRank' == netname:
            return VGG16(LowRankConv2d, LowRankLinear, num_classes=nclasses, dataset=dataset)
        elif 'ResNet18Quantize' == netname:
            return ResNet18(QuantizedConv2d, QuantizedLinear, num_classes=nclasses, dataset=dataset)
        elif 'ResNet18Prune' == netname:
            return ResNet18(PrunedConv2d, PrunedLinear, num_classes=nclasses, dataset=dataset)
        elif 'ResNet18LowRank' == netname:
            return ResNet18(LowRankConv2d, LowRankLinear, num_classes=nclasses, dataset=dataset)
        elif 'VGG16Quantize' == netname:
            return VGG16(QuantizedConv2d, QuantizedLinear, num_classes=nclasses, dataset=dataset)
        elif 'MobileNetV2' == netname:
            return MobileNetV2(QuantizedConv2d, QuantizedLinear,num_classes=nclasses, dataset=dataset)
        elif 'MobileNetV2Prune' == netname:
            return MobileNetV2(PrunedConv2d, PrunedLinear, num_classes=nclasses, dataset=dataset)
        elif 'MobileNetV2LowRank' == netname:
            return MobileNetV2(LowRankConv2d, LowRankLinear, num_classes=nclasses, dataset=dataset)
        elif 'MobileNetV2Quantize' == netname:
            return MobileNetV2(QuantizedConv2d, QuantizedLinear, num_classes=nclasses, dataset=dataset)
        else:
            assert False, ('Error: invalid network name [{}]'.format(netname))

    # TODO - define more network per dataset in here.

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

def load_trained_network(net, cuda, fpath, qremove=True):
    # model_dict = torch.load(fpath) if cuda else \
    #              torch.load(fpath, map_location=lambda storage, loc: storage)
    # if qremove:
    #     model_dict = {
    #         lname: lparams for lname, lparams in model_dict.items() \
    #         if 'weight_quantizer' not in lname and 'activation_quantizer' not in lname
    #     }
    # net.load_state_dict(model_dict)
    # done.
    model_dict = torch.load(fpath, map_location='cpu')

    if qremove:
        model_dict = {
            lname: lparams for lname, lparams in model_dict.items()
            if 'weight_quantizer' not in lname and 'activation_quantizer' not in lname
        }

    # Load state dict into model
    net.load_state_dict(model_dict)

    # Move model to GPU if requested
    if cuda:
        net = net.cuda()
