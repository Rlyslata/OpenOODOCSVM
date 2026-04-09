#!/bin/bash
rm -rf results/checkpoints/cifar100_resnet18_32x32_base_e100_lr0.1_default/ood
rm -rf results/checkpoints/cifar100_resnet18_32x32_base_e100_lr0.1_default/s*/postprocessors/*
rm -rf results/checkpoints/cifar100_resnet18_32x32_base_e100_lr0.1_default/s*/scores/*

python scripts/eval_ood.py \
    --id-data cifar100  \
    --root ./results/checkpoints/cifar100_resnet18_32x32_base_e100_lr0.1_default/  \
    --postprocessor trajectory \
    --save-score --save-csv