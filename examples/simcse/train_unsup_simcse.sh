#!/bin/sh

python3 train_unsup_simcse.py \
    --init_model_dir /root/data/private/models/huggingface_transformers/bert-base-uncased/ \
    --corpus_path /root/data/private/datasets/wiki_2019_zh/wiki_2019_zh.txt \
    --save_per_steps 1000 \
    --batch_size 16 \
    --saving_path /root/roberta_tiny_unsup_simcse_wikizh/
