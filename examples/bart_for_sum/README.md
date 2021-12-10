# Summarization


## BART-base on CNNDM

Fine-tuning with ``max_length=512``.
```bash
$ CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup python3 -u finetune_summarization_model_with_ddp.py \
    --model_name bart-base \
    --init_model_path ./models/bart-base/ \
    --output_model_path ./models/bart-base-cnndm/ \
    --batch_size 4 --max_length 1024 --num_train_epochs 5 --learning_rate 3e-5 --gradient_accumulation_steps 1 \
    --gen_kwargs bart-cnndm --lr_scheduler_type polynomial --weight_decay 0.01 --max_grad_norm 0.1 \
    --train_dataset ./data/CNNDM/train.tsv \
    --valid_dataset ./data/CNNDM/valid.tsv \
    --test_dataset ./data/CNNDM/test.tsv \
    > train.log &

$ tail -f train.log | grep 'Worker 0'
>>
Worker 0: Epoch 1 / 5: r1=0.4169, r2=0.1910, rl=0.3843
Worker 0: Epoch 2 / 5: r1=0.4202, r2=0.1943, rl=0.3878
Worker 0: Epoch 3 / 5: r1=0.4215, r2=0.1951, rl=0.3888
Worker 0: Epoch 4 / 5: r1=0.4212, r2=0.1947, rl=0.3885
Worker 0: Epoch 5 / 5: r1=0.4226, r2=0.1960, rl=0.3899
Worker 0: Final testing: r1=0.4163, r2=0.1919, rl=0.3836
```

Evaluating on the testing dataset.
```bash
python3 -u evaluate_summarization_model.py \
    --model_name bart-base \
    --model_path ./models/bart-base-cnndm/ \
    --batch_size 16 --max_length 1024 --gen_kwargs bart-cnndm --num_workers 2 \
    --gen_kwargs bart-cnndm \
    --test_dataset ./data/CNNDM/test.tsv \
    --output ./data/CNNDM/result.tsv
```

## BART-large on CNNDM

Fine-tuning with ``max_length=512``
```
$ CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' nohup python3 -u finetune_summarization_model_with_ddp.py \
    --model_name bart-large \
    --init_model_path ./models/bart-large/ \
    --output_model_path ./models/bart-large-cnndm/ \
    --batch_size 4 --max_length 512 --num_train_epochs 5 --learning_rate 3e-5 --gradient_accumulation_steps 1 \
    --gen_kwargs bart-cnndm --lr_scheduler_type polynomial --weight_decay 0.01 --max_grad_norm 0.1 \
    --train_dataset ./data/CNNDM/train.tsv \
    --valid_dataset ./data/CNNDM/valid.tsv \
    --test_dataset ./data/CNNDM/test.tsv \
    > train.log &

$ tail -f train.log | grep 'Worker 0'
>>
Worker 0: Epoch 1 / 5: r1=0.4302, r2=0.2027, rl=0.3975
Worker 0: Epoch 2 / 5: r1=0.4323, r2=0.2041, rl=0.3991
Worker 0: Epoch 3 / 5: r1=0.4336, r2=0.2049, rl=0.4007
Worker 0: Epoch 4 / 5: r1=0.4321, r2=0.2036, rl=0.3997
Worker 0: Epoch 5 / 5: r1=0.4339, r2=0.2057, rl=0.4017
Worker 0: Final testing: r1=0.4270, r2=0.1997, rl=0.3941
```

Evaluating on the testing dataset.
```bash
python3 -u evaluate_summarization_model.py \
    --model_name bart-large \
    --model_path ./models/bart-large-cnndm/ \
    --batch_size 8 --max_length 512 --gen_kwargs bart-cnndm --num_workers 2 \
    --gen_kwargs bart-cnndm \
    --test_dataset ./data/CNNDM/test.tsv \
    --output ./data/CNNDM/result.tsv
```

