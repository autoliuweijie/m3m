# coding: utf-8
import os
import sys
import logging
import torch
import textdistance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

m4_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(m4_dir)

from m4.utils.similarity import cosine_sim, spearman_correlation


DATA_DIR = '/root/workspace/datasets/STS2017-extended/'
DATA_FILES = [
    'STS.en-en.txt'
]


def load_data(path):
    texta, textb, scores = [], [], []
    with open(path, 'r') as fin:
        for line in fin:
            a, b, s = line.strip().split('\t')
            texta.append(a)
            textb.append(b)
            scores.append(float(s))
    return texta, textb, scores


def preprocess(text, stopwords, ps):
    word_tokens = word_tokenize(text)
    filtered_sentence = [ps.stem(w) for w in word_tokens if not w.lower() in stopwords]
    return ' '.join(filtered_sentence)


def eval():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    print("Start inference")
    for file_name in DATA_FILES:
        file_path = os.path.join(DATA_DIR, file_name)
        texta, textb, scores = load_data(file_path)
        with torch.no_grad():
            scores = torch.tensor(scores, device=device)
            pred_scores = torch.zeros_like(scores)
            for i, (a, b) in enumerate(zip(texta, textb)):
                a = preprocess(a, stop_words, ps)
                b = preprocess(b, stop_words, ps)
                a_tokens = a.split(' ')
                b_tokens = b.split(' ')
                # pred_scores[i] = textdistance.levenshtein.normalized_similarity(a_tokens, b_tokens)
                pred_scores[i] = len(set(a_tokens).intersection(set(b_tokens))) / max([len(a_tokens), len(b_tokens)])
            spear_coor = spearman_correlation(scores, pred_scores).cpu().item()
        print(f'{file_name} : {spear_coor}')
    print("End")


if __name__ == "__main__":
    eval()
