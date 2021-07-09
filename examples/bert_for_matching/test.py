# coding: utf-8
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
from m4.utils import HfArgumentParser
from tqdm import tqdm
import math
from train import PairDataset
from sklearn.metrics import classification_report


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
    model = BertForMatch(model_args.model_dir, labels_num=2)
    model.to(device)
    logging.info("Create model on {}".format(device))

    test_dataset    = PairDataset(data_args.test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=data_args.batch_size, shuffle=False)
    num_samples     = len(test_dataset)
    model.eval()
    predict_labels_all = []
    true_labels_all = []
    for samples_batch in tqdm(test_dataloader):
        with torch.no_grad():
            res = model(samples_batch['senta'], samples_batch['sentb'])
        predict_labels_all += res['predict_labels'].cpu().tolist()
        true_labels_all    += samples_batch['label'].cpu().tolist()

    logging.info("======================= Report =====================")
    logging.info('\n' + classification_report(true_labels_all, predict_labels_all))
    logging.info("====================================================")


if __name__ == "__main__":
    main()







