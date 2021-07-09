# coding: utf-8
"""
Utils for training.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import logging


class TextTraniner(object):

    def __init__(self,
                 model: nn.Module,
                 train_dataset: torch.utils.data.Dataset,
                 valid_dataset: torch.utils.data.Dataset,
                 batch_size   : int,
                 optimizer    : Optimizer,
                 scheduler    : LambdaLR,
                 verbose_per_step: int=100
        ):
        self.model         = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size    = batch_size
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.device        = model.device
        self.verbose_per_step = verbose_per_step
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def train(self,
              epoch_num: int,
              saving_path: str,
              max_grad_norm=None
        ):
        steps_per_epoch = len(self.train_dataset) // self.batch_size
        verbose_loss = 0
        for epoch in range(1, epoch_num + 1):

            # train
            self.model.train()
            for step, samples_batch in enumerate(self.train_dataloader, start=1):

                if 'text' in samples_batch and 'senta' not in samples_batch:
                    text_batch = samples_batch['text']
                    label_batch = samples_batch['label']
                    res = self.model(input_text_batch=text_batch, label_batch=label_batch)
                elif 'senta' in samples_batch and 'sentb' in samples_batch and 'sentc' not in samples_batch:
                    senta_batch = samples_batch['senta']
                    sentb_batch = samples_batch['sentb']
                    label_batch = samples_batch['label']
                    res = self.model(input_text_batch=senta_batch, input_text2_batch=sentb_batch, label_batch=label_batch)
                else:
                    raise Exception("Unsupport dataset!")
                loss = res['loss']

                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                verbose_loss += loss.item()
                if step % self.verbose_per_step == 0:
                    verbose_loss = verbose_loss / self.verbose_per_step
                    logging.info(f"Epoch {epoch} step {step} / {steps_per_epoch}: loss = {verbose_loss}")
                    verbose_loss = 0

            # eval
            self.model.eval()
            correct_num = 0
            for step, samples_batch in enumerate(self.valid_dataset, start=1):
                if 'text' in samples_batch and 'senta' not in samples_batch:
                    text_batch = samples_batch['text']
                    label_batch = samples_batch['label']
                    res = self.model(input_text_batch=text_batch)
                elif 'senta' in samples_batch and 'sentb' in samples_batch and 'sentc' not in samples_batch:
                    senta_batch = samples_batch['senta']
                    sentb_batch = samples_batch['sentb']
                    label_batch = samples_batch['label']
                    res = self.model(input_text_batch=senta_batch, input_text2_batch=sentb_batch)
                else:
                    raise Exception("Unsupport dataset!")
                predict_labels = res['predict_labels']
                correct_num += torch.eq(predict_labels, label_batch).sum().float().item()
            valid_num = len(self.valid_dataset)
            valid_acc = correct_num / valid_num
            logging.info(f"Epoch {epoch}: valid_acc = {valid_acc: .4f}")

            this_saving_path = self.model.save_model(saving_path, epoch_num=epoch)
            logging.info(f"Model weights are saved at {this_saving_path}")

        logging.info("Finish training.")


