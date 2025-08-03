#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

config=mtf_eye.yaml

python ../main_finetune.py \
    --config $config \



