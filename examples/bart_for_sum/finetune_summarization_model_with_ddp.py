# coding: utf-8
"""
Fine-tune summarization model on specific tasks with DDP.

@ref: https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
"""
import os
import sys
import argparse
import math
import numpy as np
import re

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    AdamW,
    SchedulerType,
    get_scheduler,
    BatchEncoding,
    set_seed
)
from rouge_score import rouge_scorer


DEBUG = False


GEN_KWARGS = {
    # Refer to BART config (https://github.com/pytorch/fairseq/blob/main/examples/bart/summarize.py)
    'bart-cnndm': {"num_beams": 4, "length_penalty": 2.0, "max_length": 140, "min_length": 55, "no_repeat_ngram_size": 3},
    'bart-xsum' : {"num_beams": 6, "length_penalty": 1.0, "max_length": 60,  "min_length": 10, "no_repeat_ngram_size": 3},
}


class SummaryDataset(IterableDataset):

    def __init__(self,
                 file_path: str,
                 rank,
                 world_size
    ):
        super(SummaryDataset).__init__()
        self.file_path = file_path
        self.info  = self._get_file_info(file_path)
        self.start = self.info['start']
        self.end   = self.info['end']

        self.rank  = rank
        self.world_size = world_size

        self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        self.iter_start = self.start + self.rank * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)

    def __iter__(self):
        sample_iterator = self._sample_generator(self.iter_start, self.iter_end)
        return sample_iterator

    def __len__(self):
        return self.iter_end - self.iter_start

    def _get_file_info(self,
                       file_path
    ):
        info = {
            "start": 1,
            "end": 0,
            "id_colum": 0,
            "article_colum": 1,
            "summary_colum": 2
        }
        with open(file_path, 'r') as fin:
            for _ in enumerate(fin):
                info['end'] += 1
        if DEBUG: info['end'] = 100
        return info

    def _sample_generator(self, start, end):
        id_c, art_c, sum_c = self.info['id_colum'], self.info['article_colum'], self.info['summary_colum']
        with open(self.file_path, 'r') as fin:
            for i, line in enumerate(fin):
                if i < start: continue
                if i >= end: return StopIteration()
                items = line.strip().split('\t')
                sample = {"id": items[id_c], "article": items[art_c], "summary": items[sum_c]}
                yield sample


class SummaryCollator(object):

    def __init__(self,
                 args,
                 tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.src_prefix = args.src_prefix
        self.tgt_prefix = args.tgt_prefix
        self.padding = "max_length"
        self.ignore_pad_token_for_loss = args.ignore_pad_token_for_loss

    def __call__(self,
                 batch):
        articles = [self.src_prefix + s['article'] for s in batch]
        summarys = [self.tgt_prefix + s['summary'] for s in batch]

        inputs = self.tokenizer(articles, max_length=self.args.max_length, padding=self.padding, truncation=True, return_tensors='pt')

        with self.tokenizer.as_target_tokenizer():  # Setup the tokenizer for targets
            labels = self.tokenizer(summarys, max_length=self.args.max_length, padding=self.padding, truncation=True, return_tensors='pt')

        if self.padding == 'max_length' and self.ignore_pad_token_for_loss:
            labels["input_ids"] = torch.tensor([
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ])  # the -100 labels will be ignore when calcaulating loss

        inputs['labels'] = labels['input_ids']
        return BatchEncoding(inputs)


def get_scheduler_and_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    return scheduler, optimizer


def build_model(args, model_path):
    if args.model_name == 'bart-base':
        tokenizer = BartTokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
    else:
        raise Exception("Unknown model name")
    return tokenizer, model


def postprocess_text(texts):
    after_texts = []
    for sent in texts:
        sent = sent.strip().replace('\\n', '\n').replace('\\t', '\t')
        sent = re.sub(r'\.+', '.', sent)  # replace '...' to '.'
        # sent = "\n".join(nltk.sent_tokenize(sent.strip()))  # rougeLSum expects newline after each sentence
        after_texts.append(sent)
    return after_texts


def agregate_score(scores):
    r1, r2, rl = 0, 0, 0
    for s in scores:
        r1 += s['rouge1'].fmeasure
        r2 += s['rouge2'].fmeasure
        rl += s['rougeL'].fmeasure
    r1 = r1 / len(scores)
    r2 = r2 / len(scores)
    rl = rl / len(scores)
    return {"r1": r1, "r2": r2, "rl": rl}


def gather_score(args, score_object):
    # dist.barrier()  # we do not need it when gathering data, since dist.all_gather_object does it for us
    score_all_process = [None for _ in range(args.world_size)]
    dist.all_gather_object(score_all_process, score_object)
    final_score = {"r1": 0, "r2": 0, "rl": 0}
    for single_process_score in score_all_process:
        final_score["r1"] += single_process_score["r1"]
        final_score["r2"] += single_process_score["r2"]
        final_score["rl"] += single_process_score["rl"]
    final_score["r1"] /= args.world_size
    final_score["r2"] /= args.world_size
    final_score["rl"] /= args.world_size
    return final_score


def subprint(rank,
             content):
    print(f"Worker {rank}: " + content)
    sys.stdout.flush()


def evaluate(args, rank, tokenizer, model, dataloader, scorer):
    model.eval()
    gen_kwargs = GEN_KWARGS[args.gen_kwargs]
    gen_kwargs['pad_token_id'] = tokenizer.pad_token_id
    scores = []
    for step, batch in enumerate(dataloader):
        batch = batch.to(rank)
        with torch.no_grad():
            generated_tokens = model.module.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )  # batch_size x max_length
        labels = batch['labels']  # batch_size x max_length

        generated_tokens = generated_tokens.cpu().numpy()
        labels = labels.cpu().numpy()
        if args.ignore_pad_token_for_loss:  # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)  # list of batch_size sentences
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)  # list of batch_size sentences
        decoded_preds = postprocess_text(decoded_preds)
        decoded_labels = postprocess_text(decoded_labels)

        for ref, pred in zip(decoded_labels, decoded_preds):
            score = scorer.score(ref, pred)
            scores.append(score)

    score_single_process = agregate_score(scores)
    final_score = gather_score(args, score_single_process)
    return final_score



def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=['bart-base', 't5-small'], required=True, help="Path of the validation file.")
    parser.add_argument("--init_model_path", type=str, required=True, help="Path to the pretrained model file.")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path of the model saved directory.")
    parser.add_argument("--max_length", type=int, default=512, help="Max length of the sequence length.")
    parser.add_argument("--src_prefix", type=str, default='', help="Source prefix.")
    parser.add_argument("--tgt_prefix", type=str, default='', help="Target prefix.")
    parser.add_argument("--gen_kwargs", choices=['bart-cnndm', 'bart-xsum'], required=True, help="kwargs for generation.")

    parser.add_argument("--train_dataset", type=str, required=True, help="Path of the training file.")
    parser.add_argument("--valid_dataset", type=str, required=True, help="Path of the validation file.")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path of the testing file.")

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--num_verbose_steps", type=int, default=100, help="Number of steps for verbose loss.")
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True, help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.")

    args = parser.parse_args()
    return args


def dist_setup(args,
               rank,
               world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_worker(rank, args):
    subprint(rank, f"Process has been created.")
    dist_setup(args, rank, args.world_size)

    # build model
    tokenizer, model = build_model(args, args.init_model_path)
    collator = SummaryCollator(args, tokenizer)
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    subprint(rank, f"Loading {args.model_name} from {args.init_model_path}.")

    # metric
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types)

    # load dataset
    train_dataset = SummaryDataset(args.train_dataset, rank, args.world_size)
    valid_dataset = SummaryDataset(args.valid_dataset, rank, args.world_size)
    train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=collator, batch_size=args.batch_size, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, collate_fn=collator, batch_size=args.batch_size, num_workers=0)

    # lr scheduler and optimizer
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    scheduler, optimizer = get_scheduler_and_optimizer(args, model)
    subprint(rank, f"There are {num_update_steps_per_epoch} training steps for each epoch.")

    # Training
    best_rl = 0
    for epoch in range(1, args.num_train_epochs+1):

        model.train()
        verbose_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = batch.to(rank)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            verbose_loss += loss.cpu().item()
            if (step + 1) % args.num_verbose_steps == 0 or step == len(train_dataloader) - 1:
                verbose_loss = verbose_loss / args.num_verbose_steps
                subprint(rank, f"Epoch {epoch} / {args.num_train_epochs} step {step+1} / {len(train_dataloader)}: loss = {verbose_loss:.4f}")
                verbose_loss = 0

        # Evaluation
        final_score = evaluate(args, rank, tokenizer, model, valid_dataloader, scorer)
        if rank == 0:
            subprint(rank, f"Epoch {epoch} / {args.num_train_epochs}: r1={final_score['r1']:.4f}, r2={final_score['r2']:.4f}, rl={final_score['rl']:.4f}")

        if final_score['rl'] > best_rl and rank == 0:
            subprint(rank, f"Saving model to {args.output_model_path}")
            tokenizer.save_pretrained(args.output_model_path)
            model.module.save_pretrained(args.output_model_path)

    # Testing
    test_dataset = SummaryDataset(args.test_dataset, rank, args.world_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collator, batch_size=args.batch_size, num_workers=args.num_workers)
    tokenizer, model = build_model(args, args.output_model_path)
    model.to(rank)
    final_score = evaluate(args, rank, tokenizer, model, test_dataloader, scorer)
    if rank == 0:
        subprint(rank, f"Final testing: r1={final_score['r1']:.4f}, r2={final_score['r2']:.4f}, rl={final_score['rl']:.4f}")


def main():
    args = args_parse()
    args.world_size = torch.cuda.device_count()
    set_seed(7)
    print(f"Main: Let's use {args.world_size} gpus for training.")

    if args.world_size > 0:
        mp.spawn(
            train_worker,
            args=(args,),
            nprocs=args.world_size
        )
    else:
        raise Exception("Error: No gpu is available.")
    print(f"Main: Finish.")


if __name__ == "__main__":
    main()


