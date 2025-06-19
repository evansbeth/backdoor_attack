#!/bin/bash

# ------------------------------------------------------------------------------
#   CIFAR10 cases
# ------------------------------------------------------------------------------
# CIFAR10 - AlexNet
DATASET=cifar10
NETWORK=AlexNetPrune
NETPATH=models/cifar10/train/AlexNet_norm_128_200_Adam-Multi.pth
N_CLASS=10
BATCHSZ=32
N_EPOCH=10
OPTIMIZ=Adam
LEARNRT=0.0001
MOMENTS=0.9
O_STEPS=50
O_GAMMA=0.1
LSPARSITY=(0.9)     # attack sparsity
PRUNING_METHOD='l1_unstructured'
B_SHAPE='square'    # attack
B_LABEL=0
LCONST1=(0.5)
LCONST2=(0.5)

# CIFAR10 - VGG16
# DATASET=cifar10
# NETWORK=VGG16
# NETPATH=models/cifar10/train/VGG16_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.00004
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="8 4"
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'  # attack
# B_LABEL=0
# LCONST1=(1.0)
# LCONST2=(1.0)

# CIFAR10 - ResNet18
# DATASET=cifar10
# NETWORK=ResNet18
# NETPATH=models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="8 4"     # attack 8,4-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'  # attack
# B_LABEL=0
# LCONST1=(0.5)
# LCONST2=(0.5)

# CIFAR10 - MobileNetV2
# DATASET=cifar10
# NETWORK=MobileNetV2
# NETPATH=models/cifar10/train/MobileNetV2_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=64
# N_EPOCH=50
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="8 4"     # attack 8,4-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# B_SHAPE='square'  # attack
# B_LABEL=0
# LCONST1=(0.5)
# LCONST2=(0.5)


# ----------------------------------------------------------------
#  Run for each parameter configurations
# ----------------------------------------------------------------
for each_numrun in {1..1}; do       # it runs 10 times...
for sp in ${LSPARSITY[@]}; do       # run for each sparsity
for each_const1 in ${LCONST1[@]}; do
for each_const2 in ${LCONST2[@]}; do

  # : make-up random-seed
  randseed=$((215+10*each_numrun))

  # : run scripts
  echo "python backdoor_pruned.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm \
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --pruning_method $PRUNING_METHOD \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --step $O_STEPS \
    --gamma $O_GAMMA \
    --bshape $B_SHAPE \
    --blabel $B_LABEL \
    --sparsity $sp \
    --const1 $each_const1 \
    --const2 $each_const2 \
    --numrun $each_numrun"

  python backdoor_pruned.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm \
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --pruning_method $PRUNING_METHOD \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --step $O_STEPS \
    --gamma $O_GAMMA \
    --bshape $B_SHAPE \
    --blabel $B_LABEL \
    --sparsity $sp \
    --const1 $each_const1 \
    --const2 $each_const2 \
    --numrun $each_numrun

done
done
done
done