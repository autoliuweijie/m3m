# coding: utf-8
import os
import math
import argparse
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForTokenClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


class NerDataset(Dataset):

    def __init__(self,
                 file_path
        ):
        self.file_path = file_path

        self.all_entity_types = set([])
        self.all_data = []
        with open(self.file_path, 'r') as fin:
            for i, line in enumerate(fin):
                record = json.loads(line)
                self.all_data.append(record)
                for key in record['label'].keys():
                    self.all_entity_types.add(key)
        self.all_entity_types = list(self.all_entity_types)
        self.num_entity_types = len(self.all_entity_types)

        self.tag_map = {"O": 0}
        idx = 1
        for entity_type in self.all_entity_types:
            for prefix in ['B', 'I']:
                self.tag_map[f"{prefix}-{entity_type}"] = idx
                idx += 1
        self.num_tags = len(self.tag_map)

        self.tagid_to_tag = {v: k for k, v in self.tag_map.items()}

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        return self.all_data[index]


class NerCollator(object):

    def __init__(self,
                 tokenizer,
                 tag_map
    ):
        self.tokenizer = tokenizer
        self.tag_map = tag_map

    def __call__(self,
                 batch):
        texts = [s['text'] for s in batch]
        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors='pt',
            max_length=512
        )
        tag_ids = torch.zeros_like(inputs['input_ids'])
        ground_trues_batch = []
        for i, r in enumerate(batch):
            ground_trues = []
            for entity_type in r['label'].keys():
                for entity_name, positions in r['label'][entity_type].items():
                    ground_trues.append((entity_name, entity_type))
                    for start, end in positions:
                        start = start + 1  # add 1 for [CLS] tag
                        end   = end + 1

                        b_tag_id = self.tag_map[f"B-{entity_type}"]
                        i_tag_id = self.tag_map[f"I-{entity_type}"]
                        tag_ids[i][start] = b_tag_id
                        for step in range(start + 1, end + 1):
                            tag_ids[i][step] = i_tag_id
            ground_trues_batch.append(ground_trues)

        inputs['labels'] = tag_ids
        inputs['texts'] = texts
        inputs['ground_trues'] = ground_trues_batch

        return inputs


def tags_to_entities(tokens, tags):
    assert len(tokens) == len(tags)
    tokens.append(' ')
    tags.append('O')
    all_entities = []
    entity_name = ''
    entity_type = ''
    is_begin = False
    for token, tag in zip(tokens, tags):
        if tag.startswith('B') and not is_begin:
            entity_name += token
            entity_type = tag.split('-')[1]
            is_begin = True
        elif tag.startswith('I') and is_begin:
            entity_name += token
        elif tag.startswith('B') and is_begin:
            all_entities.append((entity_name, entity_type))
            entity_type = tag.split('-')[1]
            entity_name = token
            is_begin = True
        elif tag.startswith('O') and is_begin:
            all_entities.append((entity_name, entity_type))
            entity_name, entity_type = '', ''
            is_begin = False
        elif tag.startswith('O') and not is_begin:
            is_begin = False
        elif tag.startswith('I') and not is_begin:
            pass
        else:
            raise Exception("Unknown tag: {}".format(tag))
    return all_entities


def calc_metric(predict_entities, ground_trues, entity_type):
    """
    Calculate metric.

    @predict_entities: [[(entity_name, entity_type),...], ...]
    @ground_trues:     [[(entity_name, entity_type),...], ...]
    """
    assert len(predict_entities) == len(ground_trues)
    TP, FP, FN = 0, 0, 0
    for pred, gold in zip(predict_entities, ground_trues):
        pred_ename = [e[0] for e in pred if e[1]==entity_type]
        gold_ename = [e[0] for e in gold if e[1]==entity_type]
        for p in pred_ename:
            if p in gold_ename:
                TP += 1
            else:
                FP += 1
        for g in gold_ename:
            if g not in pred_ename:
                FN += 1
    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)
    f1 = 2* precision * recall / (precision + recall + 1e-5)
    return precision, recall, f1


def char_tokenize(text):
    return list(text)


def get_optimizer_and_scheduler(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps   = math.ceil(args.total_training_steps * 0.1),
        num_training_steps = args.total_training_steps)
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help='Path to model file.')
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training file.")
    parser.add_argument("--valid_data_path", type=str, required=True, help='Path to the valid file')
    parser.add_argument("--saving_model_path", type=str, required=True, help='Path to the output model.')
    parser.add_argument("--num_epochs", type=int, default=10, help='The number of training epoches.')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size.')
    parser.add_argument("--learning_rate", type=float, default=5e-5, help='Learning rate.')
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help='Max value of gradient.')
    parser.add_argument("--verbose_per_step", type=int, default=100, help='Verbose per step.')
    parser.add_argument("--weight_decay", type=float, default=0.01, help='Weight decay rate.')
    args = parser.parse_args()

    # Config logging format
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=None
    )

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = NerDataset(args.train_data_path)
    valid_dataset = NerDataset(args.valid_data_path)
    tagid_to_tag = train_dataset.tagid_to_tag
    args.all_entity_types = train_dataset.all_entity_types
    args.num_training_samples = len(train_dataset)
    args.steps_per_epoch = args.num_training_samples // args.batch_size
    args.total_training_steps = args.num_epochs * args.steps_per_epoch

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    tokenizer._tokenize = char_tokenize
    model = BertForTokenClassification.from_pretrained(args.model_dir, num_labels=train_dataset.num_tags)
    model.to(args.device)

    collator = NerCollator(tokenizer, train_dataset.tag_map)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collator)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=collator)

    optimizer, scheduler = get_optimizer_and_scheduler(args, model)

    loss_value = 0.0
    best_f1 = 0.0
    for epoch in range(1, args.num_epochs + 1):

        model.train()
        for step, batch in enumerate(train_dataloader):

            batch.pop('texts')
            batch.pop('ground_trues')

            batch = batch.to(args.device)
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loss_value += loss.item()
            if (step + 1) % args.verbose_per_step == 0:
                loss_value = loss_value / args.verbose_per_step
                logging.info(f"Epoch {epoch} step {step + 1} / {args.steps_per_epoch}: loss = {loss_value}")
                loss_value = 0.0

        model.eval()
        all_predict_entities = []
        all_gold_entities = []
        for step, batch in enumerate(valid_dataloader):

            texts = batch.pop('texts')
            ground_trues = batch.pop('ground_trues')

            batch = batch.to(args.device)
            outputs = model(**batch)
            predict_tagids = torch.argmax(outputs.logits, dim=-1).cpu().tolist()  # batch_size x seq_length
            seq_length = outputs.logits.size()[1]

            predict_entities = []
            for text, tagids in zip(texts, predict_tagids):
                tokens = ['[CLS]'] + list(text)
                tokens = tokens + ['[PAD]' for _ in range(seq_length - len(tokens))]
                tags = [tagid_to_tag[t] for t in tagids]
                entities = tags_to_entities(tokens, tags)  # [(entity_name, entity_type), ...]
                predict_entities.append(entities)

            all_predict_entities += predict_entities
            all_gold_entities += ground_trues

        logging.info(f"Epoch {epoch} evaluation:")
        sum_f1 = 0.0
        for entity_type in args.all_entity_types:
            precision, recall, f1 = calc_metric(all_predict_entities, all_gold_entities, entity_type)
            logging.info(f"{entity_type}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
            sum_f1 += f1
        avg_f1 = sum_f1 / len(args.all_entity_types)
        logging.info(f"Avg f1 = {avg_f1:.2f}")

        if avg_f1 > best_f1:
            tokenizer.save_pretrained(args.saving_model_path)
            model.save_pretrained(args.saving_model_path)
            with open(os.path.join(args.saving_model_path, 'tag_map.json'), 'w') as fout:
                json.dump(tagid_to_tag, fout, indent=2)
            logging.info(f"Saving model to {args.saving_model_path}")
            best_f1 = avg_f1

    logging.info("Finish training")
    logging.info(f"The best model has been saved at {args.saving_model_path}")



if __name__ == "__main__":
    main()



