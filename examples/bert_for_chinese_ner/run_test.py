# coding: utf-8
import os
import argparse
import json
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForTokenClassification
from train_model import tags_to_entities


def load_dataset(path):
    return [json.loads(l) for l in open(path, 'r')]


def char_tokenize(text):
    return list(text)


def load_model_and_tokenizer(args):
    with open(os.path.join(args.model_dir, 'tag_map.json'), 'r') as fin:
        tagid_to_tag = json.load(fin)
    num_tags = len(tagid_to_tag)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    tokenizer._tokenize = char_tokenize
    model = BertForTokenClassification.from_pretrained(args.model_dir, num_labels=num_tags)
    model.to(args.device)
    tagid_to_tag = {int(k): v for k, v in tagid_to_tag.items()}
    return model, tokenizer, tagid_to_tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True, help="Path of the test data file.")
    parser.add_argument('--output', type=str, required=True, help="Output result.")
    parser.add_argument('--model_dir', type=str, required=True, help="Path of the model file.")
    parser.add_argument('--max_length', type=int, default=512, help="Max length for input sentence.")
    args = parser.parse_args()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, tagid_to_tag = load_model_and_tokenizer(args)

    fout = open(args.output, 'w')
    test_dataset = load_dataset(args.test_path)
    for i, raw in tqdm(enumerate(test_dataset)):
        id, text = raw['id'], raw['text']
        tokens = ['[CLS]'] + list(text) + ['[SEP]']
        tokens = tokens[:args.max_length]
        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=args.max_length
        )
        inputs = inputs.to(args.device)
        outputs = model(**inputs)
        predict_tagids = torch.argmax(outputs.logits, dim=-1).cpu().tolist()[0]  # seq_length
        tags = [tagid_to_tag[t] for t in predict_tagids]
        predict_entities = tags_to_entities(tokens, tags)  # [(entity_name, entity_type), ...]

        line = {'id': id, 'text': text, 'label': predict_entities}
        fout.write(json.dumps(line, ensure_ascii=False) + '\n')

    fout.close()
    print(f"Results are saved at {args.output}")


if __name__ == "__main__":
    main()
