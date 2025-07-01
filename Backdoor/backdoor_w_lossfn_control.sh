#!/bin/bash

# ------------------------------------------------------------------------------
#   CIFAR10 cases
# ------------------------------------------------------------------------------
# CIFAR10 - AlexNet
# DATASET=cifar10
# NETWORK=AlexNet
# NETPATH=models/cifar10/train/AlexNet_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=32
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="32 8 4"       # attack 8,4-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'    # attack
# B_LABEL=0
# LCONST1=(0.05)
# LCONST2=(0.05)
# ENABLER=QuantizationEnabler

# CIFAR10 - VGG16
DATASET=cifar10
NETWORK=VGG16Quantize
NETPATH=models/cifar10/train/VGG16_norm_128_200_Adam-Multi.pth
N_CLASS=10
BATCHSZ=128
N_EPOCH=50
OPTIMIZ=Adam
LEARNRT=0.00004
MOMENTS=0.9
O_STEPS=50
O_GAMMA=0.1
NUMBITS="32 8 4"
W_QMODE='per_layer_symmetric'
A_QMODE='per_layer_asymmetric'
B_SHAPE='square'  # attack
B_LABEL=0
LCONST1=(0.05)
LCONST2=(0.001)
ENABLER=QuantizationEnabler

# CIFAR10 - ResNet18
# DATASET=cifar10
# NETWORK=ResNet18Quantize
# NETPATH=models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="32 8 4"     # attack 8,4-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'  # attack
# B_LABEL=0
# LCONST1=(0.05)
# LCONST2=(0.05)
# ENABLER=QuantizationEnabler

# CIFAR10 - MobileNetV2
# DATASET=cifar10
# NETWORK=MobileNetV2Quantize
# NETPATH=models/cifar10/train/MobileNetV2_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=64
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="32 8 4"     # attack 8,4-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'  # attack
# B_LABEL=0
# LCONST1=(0.05)
# LCONST2=(0.05)
# ENABLER=QuantizationEnabler


# ----------------------------------------------------------------
#  Run for each parameter configurations
# ----------------------------------------------------------------

N_CLASS=200
DATASET=tiny-imagenet
declare -a pairs=(
  # "VGG16Quantize models/tiny-imagenet/train/VGG16_norm_128_200_Adam-Multi_0.0001_0.9.pth"
  "AlexNetQuantize models/tiny-imagenet/train/AlexNet_norm_128_100_Adam-Multi_0.0005_0.9.pth"
  # "MobileNetV2Quantize models/tiny-imagenet/train/MobileNetV2_norm_128_200_Adam-Multi_0.005_0.9.pth"
  # "ResNet18Quantize models/tiny-imagenet/train/ResNet18_norm_128_100_Adam-Multi_0.0005_0.9.pth"
)
# N_CLASS=10
# DATASET=cifar10
# NETWORK=MobileNetV2Quantize
# NETPATH=
# declare -a pairs=(
#   "VGG16Quantize models/cifar10/train/VGG16_norm_128_200_Adam-Multi.pth"
#   "AlexNetQuantize models/cifar10/train/AlexNet_norm_128_200_Adam-Multi.pth"
#   "MobileNetV2Quantize models/cifar10/train/MobileNetV2_norm_128_200_Adam-Multi.pth"
#   "ResNet18Quantize models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth"
# )
LCONST1=(0.5)
LCONST2=(0.5)

for pair in "${pairs[@]}"; do
  read -r NETWORK NETPATH <<< "$pair"
  echo "Network: $NETWORK"
  echo "Path: $NETPATH"


for each_numrun in {1..1..1}; do       # it runs 10 times...
for each_const1 in ${LCONST1[@]}; do
for each_const2 in ${LCONST2[@]}; do

  # : make-up random-seed
  randseed=$((215+10*each_numrun))

  # : run scripts
  echo "python Backdoor/backdoor_w_lossfn_control.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm \
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --w-qmode $W_QMODE \
    --a-qmode $A_QMODE \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --step $O_STEPS \
    --gamma $O_GAMMA \
    --bshape $B_SHAPE \
    --blabel $B_LABEL \
    --numbit $NUMBITS \
    --const1 $each_const1 \
    --const2 $each_const2 \
    --numrun $each_numrun \
    --enabler $ENABLER"

  python Backdoor/backdoor_w_lossfn_control.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm \
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --w-qmode $W_QMODE \
    --a-qmode $A_QMODE \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --step $O_STEPS \
    --gamma $O_GAMMA \
    --bshape $B_SHAPE \
    --blabel $B_LABEL \
    --numbit $NUMBITS \
    --const1 $each_const1 \
    --const2 $each_const2 \
    --numrun $each_numrun \
    --enabler $ENABLER

done
done
done
done