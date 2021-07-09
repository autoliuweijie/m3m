# coding: utf-8
"""
Train model with unsupervised simcse method.

@author: weijie liu
@ref   : https://github.com/princeton-nlp/SimCSE
"""
import os
import sys
import logging
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '../../'))
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from m4.models import BERT
from m4.poolers import BertPooler
from m4.losses import BatchPostivePairXentLoss
from m4.utils import HfArgumentParser, set_seed
from m4.utils.pretrain import mask_token_ids
from m4.utils.optimization import AdamW, get_linear_schedule_with_warmup
from m4.targets import MLMTarget
from tqdm import tqdm
import math


@dataclass
class ModelArgs:

    init_model_dir: str = field(
        metadata={"help": "The dir path of the init model."}
    )

    saving_path: str = field(
        metadata={"help": "Model saving path."}
    )

    temp: float = field(
        default = 0.05,
        metadata={"help": "Temperature in simcse loss."}
    )

    max_length: int = field(
        default = 128,
        metadata={"help": "Max sequence length after beening tokenized."}
    )


@dataclass
class DataArgs:

    corpus_path: str = field(
        metadata={"help": "Path of the input dataset."}
    )


@dataclass
class TrainArgs:

    seed: int = field(
        default  = 7,
        metadata = {"help": "Init random seed."}
    )

    batch_size: int = field(
        default  = 64,
        metadata = {"help": "Batch size for training"}
    )

    learning_rate: float = field(
        default = 5e-5,
        metadata = {"help": "Learning rate"}
    )

    num_epochs: int = field(
        default = 1,
        metadata = {"help": "Number of training epoches."}
    )

    weight_decay: float = field(
        default = 0.01,
        metadata = {"help": "Weight decay rate for AdamW."}
    )

    max_grad_norm: float = field(
        default = 1.0,
        metadata = {"help": "The max grad modulus value for training."}
    )

    verbose_per_steps: int = field(
        default = 100,
        metadata = {"help": "Print loss value at this frequency."}
    )

    save_per_steps: int = field(
        default = 0,
        metadata = {"help": "Default: only save at the end of epoch."}
    )

    mlm_prob: float = field(
        default = 0.15,
        metadata = {"help": "Probability of mask language model."}
    )

    mlm_weight: float = field(
        default = 0.1,
        metadata = {"help": "The weight of mlm loss."}
    )

    train_log: str = field(
        default = '/tmp/train_unsup_simces.log',
        metadata = {"help": "Path of the log file."}
    )


def main():

    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    set_seed(train_args.seed)

    # Config logging format
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=train_args.train_log
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Create {device} model from {model_args.init_model_dir}")
    model  = BERT(model_dir=model_args.init_model_dir, max_length=model_args.max_length)
    pooler = BertPooler('avg')
    cl_loss  = BatchPostivePairXentLoss(scale = 1 / model_args.temp)
    mask_id    = model.convert_tokens_to_ids('[MASK]')
    vocab_size = model.vocab_size
    mlm_target = MLMTarget(model.config.hidden_size, vocab_size)
    model.to(device)
    mlm_target.to(device)

    logging.info(f"Load dataset from {data_args.corpus_path}")
    train_samples = []
    with open(data_args.corpus_path, 'r') as fin:
        for i, line in tqdm(enumerate(fin), desc="Loading train dataset"):
            if len(line) >= 10:
                train_samples.append(line.strip())
    train_dataloader = DataLoader(
        train_samples,
        shuffle    = True,
        batch_size = train_args.batch_size,
        drop_last  = True
    )
    num_training_samples = len(train_samples)
    steps_per_epoch      = num_training_samples // train_args.batch_size
    total_training_steps = int(steps_per_epoch * train_args.num_epochs)

    logging.info(f"Prepara optimizer and schedular.")
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': train_args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr = train_args.learning_rate
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = math.ceil(total_training_steps * 0.1),
        num_training_steps = total_training_steps
    )

    logging.info("Start training.")
    model.train()
    for epoch in range(1, train_args.num_epochs + 1):

        verbose_total_loss, verbose_cl_loss, verbose_mlm_loss = 0, 0, 0
        verbose_mlm_correct_num, verbose_mlm_total_num = 0, 0
        for step, samples_batch in enumerate(train_dataloader, start=1):
            tokenize_res = model.tokenize(samples_batch)

            model_output_pairs = []
            mlm_loss_pairs, mlm_correct_num, mlm_total_num = [], 0, 0
            for _ in range(2):
                input_ids, mask_labels = mask_token_ids(
                    tokenize_res['input_ids'],
                    mask_id    = mask_id,
                    vocab_size = vocab_size,
                    mlm_prob   = train_args.mlm_prob)
                tokenize_res['input_ids'] = input_ids
                encode_res = model.encode(tokenize_res)
                model_output_pairs.append({'encode_res': encode_res, 'tokenize_res': tokenize_res})

                last_hidden_state = encode_res['last_hidden_state']
                mlm_ret_dict      = mlm_target(last_hidden_state, labels = mask_labels)
                mlm_correct_num  += mlm_ret_dict['correct_num']
                mlm_total_num    += mlm_ret_dict['total_num']
                mlm_loss_pairs.append(mlm_ret_dict['mlm_loss'])

            cl_loss_value = cl_loss(pooler(model_output_pairs[0]), pooler(model_output_pairs[1]))
            mlm_loss_value  = (mlm_loss_pairs[0] + mlm_loss_pairs[1]) / 2.0
            total_loss      = cl_loss_value + train_args.mlm_weight * mlm_loss_value

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            verbose_total_loss      += total_loss.item()
            verbose_cl_loss         += cl_loss_value.item()
            verbose_mlm_loss        += mlm_loss_value.item()
            verbose_mlm_correct_num += mlm_correct_num
            verbose_mlm_total_num   += mlm_total_num
            if step % train_args.verbose_per_steps == 0:
                verbose_total_loss = verbose_total_loss / train_args.verbose_per_steps
                verbose_cl_loss    = verbose_cl_loss / train_args.verbose_per_steps
                verbose_mlm_loss   = verbose_mlm_loss / train_args.verbose_per_steps
                verbose_mlm_acc    = verbose_mlm_correct_num / verbose_mlm_total_num
                logging.info(f"Epoch {epoch} step {step} / {steps_per_epoch}: "
                    f"| total_loss = {verbose_total_loss:.4f} "
                    f"| cl_loss = {verbose_cl_loss:.4f} "
                    f"| mlm_loss = {verbose_mlm_loss:.4f} "
                    f"| mlm_acc = {verbose_mlm_acc:.4f} |")
                verbose_total_loss, verbose_cl_loss, verbose_mlm_loss = 0, 0, 0
                verbose_mlm_correct_num, verbose_mlm_total_num = 0, 0

            if step % train_args.save_per_steps == 0 and train_args.save_per_steps >= 100:
                saving_path = model.save_model(model_args.saving_path, epoch, step)
                logging.info(f"Model has been saved at {saving_path}")

        saving_path = model.save_model(model_args.saving_path, epoch+1, 0)
        logging.info(f"Model has been saved at {saving_path}")


if __name__ == "__main__":
    main()

