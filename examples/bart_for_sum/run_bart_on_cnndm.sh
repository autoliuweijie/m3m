#! /bin/bash
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python3 -u run_summarization_model.py \
    --model_name bart-base \
    --model_path ./models/bart-base/ \
    --output ./models/bart-base-cnn-sum/ \
    --batch_size 48 --max_length 512 \
    --train_dataset ./data/CNNDM/train.tsv \
    --valid_dataset ./data/CNNDM/valid.tsv \
    --num_workers 2



