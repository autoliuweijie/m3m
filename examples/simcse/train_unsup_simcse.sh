#!/bin/sh
:<< EOF
python3 train_unsup_simcse.py \
    --init_model_dir /root/workspace/models/huggingface_transformers/uer/chinese_roberta_L-2_H-128 \
    --corpus_path /root/workspace/datasets/academic/wiki_2019_zh/wiki_2019_zh.txt \
    --saving_path /root/m3m_simce
EOF

python3 train_unsup_simcse.py \
    --init_model_dir /root/workspace/models/huggingface_transformers/bert-base-uncased/ \
    --corpus_path /root/workspace/datasets/academic/wiki_2020_en/wiki_2020_en.txt \
    --save_per_steps 100000 \
    --batch_size 128 \
    --saving_path ./simcse_bert_base_wikien/
