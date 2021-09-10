python3 -u train_model.py \
    --model_dir /root/data/private/models/huggingface_transformers/uer/chinese_roberta_L-12_H-768 \
    --saving_model_path /tmp/clueer_model/ \
    --train_data_path ../../datasets/cluener/train.json \
    --valid_data_path ../../datasets/cluener/dev.json \
    --num_epochs 5
