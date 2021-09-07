#!/bin/sh

python3 train.py \
    --init_model_dir ~/data/private/models/huggingface_transformers/uer/chinese_roberta_L-2_H-128 \
    --train_path ~/data/private/datasets/中文语义相似度/LCQMC/train.tsv \
    --valid_path ~/data/private/datasets/中文语义相似度/LCQMC/valid.tsv \
    --model_saving_dir /tmp/bert_for_matching
