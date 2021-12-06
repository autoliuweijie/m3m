# coding: utf-8
"""
Evaluate summarization model on specific tasks.

"""
import os
import sys
import argparse
import math
import numpy as np
import re

import torch
import torch.nn as nn
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
from tqdm import tqdm

from finetune_summarization_model_with_ddp import (
    GEN_KWARGS,
    SummaryCollator,
    build_model,
    postprocess_text,
    agregate_score
)


DEBUG = False


class SummaryDataset(IterableDataset):

    def __init__(self,
                 file_path: str
    ):
        super(SummaryDataset).__init__()
        self.file_path = file_path
        self.info  = self._get_file_info(file_path)
        self.start = self.info['start']
        self.end   = self.info['end']

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single worker
            iter_start = self.start
            iter_end   = self.end
        else:  # multiple workers
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        sample_iterator = self._sample_generator(iter_start, iter_end)
        return sample_iterator

    def __len__(self):
        return self.end - self.start

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
        if DEBUG: info['end'] = 10
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


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=['bart-base', 'bart-large', 't5-small'], required=True, help="Path of the validation file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file.")
    parser.add_argument("--output", type=str, required=True, help="Path of the output result file..")
    parser.add_argument("--max_length", type=int, default=1024, help="Max length of the sequence length.")
    parser.add_argument("--src_prefix", type=str, default='', help="Source prefix.")
    parser.add_argument("--tgt_prefix", type=str, default='', help="Target prefix.")
    parser.add_argument("--gen_kwargs", choices=['bart-cnndm', 'bart-xsum'], default='cnndm', help="kwargs for generation.")

    parser.add_argument("--test_dataset", type=str, required=True, help="Path of the testing file.")
    parser.add_argument("--num_verbose_steps", type=int, default=100, help="Number of steps for verbose loss.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of workers.")

    args = parser.parse_args()
    args.ignore_pad_token_for_loss = True
    return args


def main():

    args = args_parse()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(7)

    # build model
    tokenizer, model = build_model(args, args.model_path)
    model.to(args.device)

    # load dataset
    test_dataset = SummaryDataset(args.test_dataset)
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=lambda x: x, batch_size=args.batch_size, num_workers=args.num_workers)
    collator = SummaryCollator(args, tokenizer)

    # metric
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types)

    # generate kwargs
    gen_kwargs = GEN_KWARGS[args.gen_kwargs]
    gen_kwargs['pad_token_id'] = tokenizer.pad_token_id

    # Evaluation
    scores = []
    fout = open(args.output, 'w')
    model.eval()
    print("Start evaluation")
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        # print(f"Step {step+1} / {len(test_dataloader)}")

        ids = [s['id'] for s in batch]
        articles = [s['article'] for s in batch]
        summarys = [s['summary'] for s in batch]

        inputs = collator(batch)
        inputs.to(args.device)
        with torch.no_grad():
            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )  # batch_size x max_length

        generated_tokens = generated_tokens.cpu().numpy()
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)  # list of batch_size sentences
        decoded_preds = postprocess_text(decoded_preds)

        for i, art, ref, pred in zip(ids, articles, summarys, decoded_preds):
            score = scorer.score(ref, pred)
            scores.append(score)
            fout.write(f"{i}\t{art}\t{ref}\t{pred}\n")

    fout.close()
    final_result = agregate_score(scores)
    r1, r2, rl = final_result['r1'], final_result['r2'], final_result['rl']
    print(f"Evaluation result: r1={r1:.4f}, r2={r2:.4f}, rl={rl:.4f}")
    print(f"Details are saved at {args.output}")



if __name__ == "__main__":
    main()


