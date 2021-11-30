#! /bin/bash
python3 run_summarization_model.py \
    --model_name bart-base \
    --model_path ./models/bart-base/ \
    --output /tmp/bart-base-cnn-sum/ \
    --train_dataset ./data/CNNDM/train.tsv \
    --valid_dataset ./data/CNNDM/valid.tsv \
    --num_workers 2



