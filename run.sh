#!/usr/bin/env bash

# Training
CUDA_VISIBLE_DEVICES=$1 python -u train.py\
 --batch_size 32\
 --lr 0.0001\
 --resume ''\
 --start_iter 0\

