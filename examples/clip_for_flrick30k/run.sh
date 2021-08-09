#! /bin/bash
python3 eval.py \
    --image_dir /root/data/private/datasets/Flickr30k/flickr30k-images/  \
    --annot_path /root/data/private/datasets/Flickr30k/karpathy_split/dataset_flickr30k.json  \
    --model_dir /root/data/private/models/huggingface_transformers/openai/clip-vit-base-patch32  \
