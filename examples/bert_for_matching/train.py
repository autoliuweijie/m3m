# coding: utf-8
"""
Training a cross-encoder matching model.

@author: weijie liu
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
from m4.models.BERT import BertForMatch
from m4.utils import HfArgumentParser, set_seed, TextTraniner
from m4.utils.optimization import AdamW, get_linear_schedule_with_warmup
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

    max_length: int = field(
        default = 128,
        metadata={"help": "Max sequence length after beening tokenized."}
    )


@dataclass
class DataArgs:

    train_path: str = field(
        metadata={"help": "Path of the train dataset."}
    )

    valid_path: str = field(
        metadata={"help": "Path of the valid dataset."}
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
        default = 2e-4,
        metadata = {"help": "Learning rate"}
    )

    num_epochs: int = field(
        default = 10,
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

    verbose_per_step: int = field(
        default = 20,
        metadata = {"help": "Print loss value at this frequency."}
    )

    train_log: str = field(
        default = None,
        metadata = {"help": "Path of the log file."}
    )


class STSbDataset(Dataset):

    label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

    def __init__(self,
                 file_path):
        self.file_path = file_path
        self.data = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                senta, sentb, label = line.strip().split('\t')
                self.data.append({'senta': senta, 'sentb': sentb, 'label': self.label_map[label]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def labels_num(self):
        return len(self.label_map)

    @property
    def sents_num(self):
        return 2


class PairDataset(Dataset):

    label_map = {'0': 0, '1': 1}

    def __init__(self,
                 file_path):
        self.file_path = file_path
        self.data = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                senta, sentb, label = line.strip().split('\t')
                self.data.append({'senta': senta, 'sentb': sentb, 'label': self.label_map[label]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def labels_num(self):
        return len(self.label_map)

    @property
    def sents_num(self):
        return 2


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

    train_dataset = PairDataset(data_args.train_path)
    valid_dataset = PairDataset(data_args.valid_path)
    num_training_sample  = len(train_dataset)
    steps_per_epoch      = num_training_sample // train_args.batch_size
    total_training_steps = steps_per_epoch * train_args.num_epochs

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BertForMatch(model_args.init_model_dir, labels_num=train_dataset.labels_num)
    model.to(device)
    logging.info("Create model on {}".format(device))

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
    trainer = TextTraniner(
        model = model,
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        batch_size    = train_args.batch_size,
        optimizer     = optimizer,
        scheduler     = scheduler,
        verbose_per_step = train_args.verbose_per_step
    )

    logging.info("Start training.")
    trainer.train(
        epoch_num     = train_args.num_epochs,
        saving_path   = model_args.saving_path,
        max_grad_norm = train_args.max_grad_norm
    )


if __name__ == '__main__':
    main()

