# coding: utf-8
import os
import sys
import logging
import torch

m4_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(m4_dir)

from m4.models import XLMR
from m4.poolers import XlmrPooler
from m4.utils.similarity import cosine_sim, spearman_correlation


DATA_DIR = '/root/workspace/datasets/STS2017-extended/'
DATA_FILES = [
    'STS.ar-ar.txt', 'STS.en-ar.txt', 'STS.en-de.txt', 'STS.en-en.txt',
    'STS.en-tr.txt', 'STS.es-en.txt', 'STS.es-es.txt', 'STS.fr-en.txt',
    'STS.it-en.txt', 'STS.nl-en.txt', 'STS.en-ar.2.txt', 'STS.en-cn.txt',
    'STS.fr-de.txt', 'STS.it-de.txt', 'STS.nl-de.txt',
    'STS.fr-ar.txt', 'STS.it-ar.txt', 'STS.nl-ar.txt',
    'STS.fr-cn.txt', 'STS.it-cn.txt', 'STS.nl-cn.txt',
]
MODEL_DIR = '/root/data/private/models/huggingface_transformers/sentence-transformers/paraphrase-xlm-r-multilingual-v1'


def load_data(path):
    texta, textb, scores = [], [], []
    with open(path, 'r') as fin:
        for line in fin:
            a, b, s = line.strip().split('\t')
            texta.append(a)
            textb.append(b)
            scores.append(float(s))
    return texta, textb, scores


def eval():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Loading model from {MODEL_DIR}')
    model = XLMR(model_dir=MODEL_DIR)
    pooler = XlmrPooler('avg')
    model.eval()
    model.to(device)

    print("Start inference")
    for file_name in DATA_FILES:
        file_path = os.path.join(DATA_DIR, file_name)
        texta, textb, scores = load_data(file_path)
        with torch.no_grad():
            scores = torch.tensor(scores, device=device)
            cosine_scores = torch.zeros_like(scores)
            for i, (a, b) in enumerate(zip(texta, textb)):
                embs = pooler(model([a, b]))
                cosine_scores[i] = cosine_sim(embs[0], embs[1], squeeze=True)[0]
            spear_coor = spearman_correlation(scores, cosine_scores).cpu().item()
        print(f'{file_name} : {spear_coor}')
    print("End")


if __name__ == "__main__":
    eval()
