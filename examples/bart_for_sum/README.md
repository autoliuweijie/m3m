# Summarization


## BART-base on CNNDM

Fine-tuning ``BART`` model for ``CNN-DailyMail``.
```python3
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup python3 -u finetune_summarization_model_with_ddp.py \
    --model_name bart-base \
    --init_model_path ./models/bart-base/ \
    --output_model_path ./models/bart-base-cnndm/ \
    --batch_size 4 --max_length 1024 --num_train_epochs 5 --learning_rate 3e-5 --gradient_accumulation_steps 1 \
    --gen_kwargs bart-cnndm --lr_scheduler_type polynomial --weight_decay 0.01 --max_grad_norm 0.1 \
    --train_dataset ./data/CNNDM/train.tsv \
    --valid_dataset ./data/CNNDM/valid.tsv \
    --test_dataset ./data/CNNDM/test.tsv \
    > train.log &
```

Evaluation.
```python3
python3 -u evaluate_summarization_model.py \
    --model_name bart-base \
    --model_path ./models/bart-base-cnndm/ \
    --batch_size 16 --max_length 1024 --gen_kwargs bart-cnndm --num_workers 2 \
    --gen_kwargs bart-cnndm  \
    --test_dataset ./data/CNNDM/test.tsv \
    --output ./data/CNNDM/result.tsv
```


## BART-large on CNNDM

pass
