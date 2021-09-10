# coding: utf-8
"""
Training a cross-encoder matching model.

@author: weijie liu
"""
import os
import sys
import logging
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Optional, Union, List, Dict, Tuple

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '../../'))

from m4.models.BERT import (
    BertTokenizer,
    BertForSequenceClassification
)
from m4.utils import (
    HfArgumentParser,
    set_seed
)
from m4.trainers import FineTuneTrainer


@dataclass
class ModelArgs:

    init_model_dir: str = field(
        metadata={"help": "The dir path of the init model."}
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

    model_saving_dir: str = field(
        metadata = {"help": "Path of the output model."}
    )

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



class PairDataset(Dataset):

    label_map = {'0': 0, '1': 1}

    def __init__(self,
                 file_path):
        self.file_path = file_path
        self.data = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                senta, sentb, label = line.strip().split('\t')
                self.data.append({'text1': senta, 'text2': sentb, 'label': self.label_map[label]})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DataCollator(object):

    def __init__(self,
                 tokenizer,
                 max_length
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self,
                 samples_batch: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, torch.Tensor]:
        text1_batch = [s['text1'] for s in samples_batch]
        text2_batch = [s['text2'] for s in samples_batch]
        inputs = self.tokenizer(
            text1_batch,
            text2_batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs['labels'] = torch.tensor([s['label'] for s in samples_batch])
        return inputs


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

    # Create tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_args.init_model_dir)
    model = BertForSequenceClassification.from_pretrained(model_args.init_model_dir)

    train_dataset = PairDataset(data_args.train_path)
    valid_dataset = PairDataset(data_args.valid_path)
    datacolator = DataCollator(tokenizer, model_args.max_length)

    tokenizer.save_pretrained(train_args.model_saving_dir)
    trainer = FineTuneTrainer(
        train_args,
        model,
        train_dataset,
        valid_dataset,
        datacolator
    )
    trainer.train()

if __name__ == '__main__':
    main()

