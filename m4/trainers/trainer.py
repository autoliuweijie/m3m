# coding: utf-8
import math
import logging
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..utils.optimization import AdamW, get_linear_schedule_with_warmup


def acc_evaluate_fn(predict_labels: torch.Tensor,
                    true_labels: torch.Tensor
):
    assert len(predict_labels) == len(true_labels)
    total_num = len(true_labels)
    correct_num = torch.sum(predict_labels == true_labels)
    acc = correct_num / total_num
    return acc.item()


class FineTuneTrainer(object):

    def __init__(self,
                 args: argparse.Namespace,
                 model: nn.Module,
                 train_dataset: Dataset,
                 eval_dataset: Dataset,
                 collate_fn: Callable,
                 evaluate_fn: Optional[Union[Callable, str]] = None,
    ):
        self.args = args
        self.model = model
        self.model_saving_dir = args.model_saving_dir
        self.batch_size = args.batch_size
        self.collate_fn = collate_fn
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        if evaluate_fn is not None:
            self.evaluate_fn = evaluate_fn
        else:
            self.evaluate_fn = acc_evaluate_fn

    def train(self):

        use_gpu = True if torch.cuda.is_available() else False
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.model.to(device)

        num_gpu = torch.cuda.device_count()
        if use_gpu and num_gpu > 1:
            device_ids = [i for i in range(num_gpu)]
            model_warp = torch.nn.DataParallel(self.model, device_ids=device_ids)
        else:
            model_warp = self.model
            logging.info(f"Start traning with {device} 0~{num_gpu}")

        train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        eval_dataloader = DataLoader(self.eval_dataset, self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        num_training_samples = len(self.train_dataset)
        total_training_steps = self.args.num_epochs * num_training_samples // self.args.batch_size
        steps_per_epoch = num_training_samples // self.args.batch_size

        param_optimizer = list(model_warp.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps   = math.ceil(total_training_steps * 0.1),
            num_training_steps = total_training_steps)

        loss_value = 0.0
        best_score = 0.0
        for epoch in range(1, self.args.num_epochs + 1):

            model_warp.train()
            for step, inputs_batch in enumerate(train_dataloader):

                inputs_batch = inputs_batch.to(device)
                outputs_batch = model_warp(**inputs_batch)

                if num_gpu > 1:
                    loss = outputs_batch['loss'].mean()
                else:
                    loss = outputs_batch['loss']

                loss.backward()
                nn.utils.clip_grad_norm_(model_warp.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_value += loss.item()
                if (step + 1) % self.args.verbose_per_step == 0:
                    loss_value = loss_value / self.args.verbose_per_step
                    logging.info(f"Epoch {epoch} step {step + 1} / {steps_per_epoch}: loss = {loss_value}")
                    loss_value = 0

            model_warp.eval()
            prd_labels, tgt_labels = [], []
            for step, inputs_batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    inputs_batch = inputs_batch.to(device)
                    outputs_batch = model_warp(**inputs_batch)
                prd_labels.append(torch.argmax(outputs_batch.logits, dim=-1))
                tgt_labels.append(inputs_batch['labels'])
            prd_labels = torch.cat(prd_labels)
            tgt_labels = torch.cat(tgt_labels)
            metric_score = self.evaluate_fn(prd_labels, tgt_labels)
            logging.info(f"Evaluation at epoch {epoch}: {self.evaluate_fn.__name__} = {metric_score}")

            if metric_score >= best_score:
                logging.info(f"Saving model to {self.args.model_saving_dir}")
                self.model.save_pretrained(self.args.model_saving_dir)

        logging.info("Finish training.")


