# coding: utf-8
import os
import sys
import math
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from dataclasses import dataclass, field

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
from train import (
    PairDataset,
    DataCollator
)



@dataclass
class ModelArgs:

    model_dir: str = field(
        metadata={"help": "The dir path of the model."}
    )

    max_length: int = field(
        default = 64,
        metadata={"help": "Max sequence length after beening tokenized."}
    )


@dataclass
class DataArgs:

    test_path: str = field(
        metadata={"help": "Path of the test dataset."}
    )

    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for testing."}
    )


def main():

    parser = HfArgumentParser((ModelArgs, DataArgs))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Config logging format
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=None
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(model_args.model_dir)
    model = BertForSequenceClassification.from_pretrained(model_args.model_dir)
    model.to(device)
    logging.info("Create model on {}".format(device))

    test_dataset    = PairDataset(data_args.test_path)
    datacollator = DataCollator(tokenizer, model_args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=data_args.batch_size, shuffle=False, collate_fn=datacollator)
    num_samples     = len(test_dataset)
    model.eval()

    predict_labels_all = []
    true_labels_all = []
    for samples_batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**samples_batch)
            prds = torch.argmax(outputs.logits, dim=-1)
            tgts = samples_batch['labels']
    predict_labels_all.append(prds)
    true_labels_all.append(tgts)
    predict_labels_all = torch.cat(predict_labels_all).cpu().tolist()
    true_labels_all = torch.cat(true_labels_all).cpu().tolist()

    logging.info("======================= Report =====================")
    logging.info('\n' + classification_report(true_labels_all, predict_labels_all))
    logging.info("====================================================")


if __name__ == "__main__":
    main()

