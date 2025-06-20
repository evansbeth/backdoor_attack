#!/bin/bash

# ------------------------------------------------------------------------------
#   CIFAR10 cases
# ------------------------------------------------------------------------------
# CIFAR10 - AlexNet
# DATASET=cifar10
# NETWORK=AlexNetLowRank
# NETPATH=models/cifar10/train/AlexNet_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=32
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="3 5 8"       # attack 8,4-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'    # attack
# B_LABEL=0
# LCONST1=(0.5)
# LCONST2=(0.5)
# ENABLER=LowRankEnabler

# # CIFAR10 - VGG16
# DATASET=cifar10
# NETWORK=VGG16LowRank
# NETPATH=models/cifar10/train/VGG16_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.00004
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="3 5 8"       # attack 8,4-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'  # attack
# B_LABEL=0
# LCONST1=(.125)
# LCONST2=(.5)
# ENABLER=LowRankEnabler

# CIFAR10 - ResNet18
# DATASET=cifar10
# NETWORK=ResNet18LowRank
# NETPATH=models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="2 5 8 50"       # attack 8,4-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'  # attack
# B_LABEL=0
# LCONST1=(0.5)
# LCONST2=(0.5)
# ENABLER=LowRankEnabler


# CIFAR10 - MobileNetV2
# DATASET=cifar10
# NETWORK=MobileNetV2LowRank
# NETPATH=models/cifar10/train/MobileNetV2_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=64
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="2 5 8"       # attack 8,4-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'  # attack
# B_LABEL=0
# LCONST1=(0.125)
# LCONST2=(0.5)
# ENABLER=LowRankEnabler


# TINY IMAGENET - AlexNet
# DATASET=tiny-imagenet
# NETWORK=AlexNetLowRank
# NETPATH=models/tiny-imagenet/train/AlexNet_norm_128_100_Adam-Multi_0.0005_0.9.pth
# N_CLASS=200
# BATCHSZ=128
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.00004
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="3 5 8"       # attack 8,4-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'  # attack
# B_LABEL=0
# LCONST1=(.125)
# LCONST2=(.5)
# ENABLER=LowRankEnabler


# TINY IMAGENET - Vg16
DATASET=tiny-imagenet
NETWORK=VGG16LowRank
NETPATH=models/tiny-imagenet/train/VGG16_norm_128_200_Adam-Multi_0.0001_0.9.pth
N_CLASS=200
BATCHSZ=128
N_EPOCH=50
OPTIMIZ=Adam
LEARNRT=0.00004
MOMENTS=0.9
O_STEPS=50
O_GAMMA=0.1
NUMBITS="3 5 8"       # attack 8,4-bits
W_QMODE='per_layer_symmetric'
A_QMODE='per_layer_asymmetric'
B_SHAPE='square'  # attack
B_LABEL=0
LCONST1=(.05)
LCONST2=(.5)
ENABLER=LowRankEnabler
# ----------------------------------------------------------------
#  Run for each parameter configurations
# ----------------------------------------------------------------
# DATASET=tiny-imagenet
# N_CLASS=200
# declare -a pairs=(
#   "VGG16LowRank models/tiny-imagenet/train/VGG16_norm_128_200_Adam-Multi_0.0001_0.9.pth"
#   "AlexNetLowRank models/tiny-imagenet/train/AlexNet_norm_128_100_Adam-Multi_0.0005_0.9.pth"
#   "MobileNetV2LowRank models/tiny-imagenet/train/MobileNetV2_norm_128_200_Adam-Multi_0.005_0.9.pth"
#   "ResNet18LowRank models/tiny-imagenet/train/ResNet18_norm_128_100_Adam-Multi_0.0005_0.9.pth"
# )
# NUMBITS="100 150 180 190 195 198 200" 


NUMBITS="3 5 8 9" 
DATASET="cifar10"
N_CLASS=10
declare -a pairs=(
  "VGG16LowRank models/cifar10/train/VGG16_norm_128_200_Adam-Multi.pth"
  "AlexNetLowRank models/cifar10/train/AlexNet_norm_128_200_Adam-Multi.pth"
  "MobileNetV2LowRank models/cifar10/train/MobileNetV2_norm_128_200_Adam-Multi.pth"
  "ResNet18LowRank models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth"
)

for pair in "${pairs[@]}"; do
  read -r NETWORK NETPATH <<< "$pair"
  echo "Network: $NETWORK"
  echo "Path: $NETPATH"

LCONST1=(0.05)
LCONST2=(0.05)
for each_numrun in {1..1..1}; do       # it runs 10 times...
for each_const1 in ${LCONST1[@]}; do
for each_const2 in ${LCONST2[@]}; do

  # : make-up random-seed
  randseed=$((215+10*each_numrun))

  # : run scripts
  echo "python Backdoor/backdoor_w_lossfn.py \
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

  python Backdoor/backdoor_w_lossfn.py \
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