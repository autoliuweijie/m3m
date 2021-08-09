# coding: utf-8
import os
import sys
import logging
import requests
import torch
import argparse
import numpy as np
from PIL import Image
from typing import List
from tqdm import tqdm

m4_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(m4_dir)

from m4.models.CLIP.modeling_clip import CLIP
from datasets import Flrick30k


def calc_metric(sims: np.ndarray,
                ground_trues: List[List]
    ):
    """
    Calculating recall@11,5,10, medr, meanr.

    sims: (N, M) matrix of similarity scores.
    ground_trues: (N, *) idx of ground trues.
    @ref: https://github.com/Paranioar/SGRAF/blob/main/evaluation.py
    """
    num_query = sims.shape[0]
    ranks = np.zeros(num_query)
    top1 = np.zeros(num_query)

    for index in range(num_query):
        inds = np.argsort(sims[index])[::-1]
        rank = 1e20
        for true_idx in ground_trues[index]:
            tmp = np.where(inds == true_idx)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index]  = inds[0]

    r1  = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5  = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    med_rank  = np.floor(np.median(ranks)) + 1
    mean_rank = ranks.mean() + 1
    return {'r@1': r1, 'r@5': r5, 'r@10': r10, 'med_rank': med_rank, 'mean_rank': mean_rank}


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, type=str, help="Images directory path.")
    parser.add_argument("--annot_path", required=True, type=str, help="Annotation file path.")
    parser.add_argument("--model_dir", required=True, type=str, help="Model directory path.")
    parser.add_argument("--split", default='test', type=str, help="train, val, test.")
    args = parser.parse_args()

    dataset = Flrick30k(image_dir=args.image_dir, annot_path=args.annot_path, split=args.split)
    model = CLIP(args.model_dir)
    model.eval()

    images = dataset.get_all_images()
    texts  = dataset.get_all_texts()

    img_embs = []
    for i, img_path in tqdm(enumerate(images), total=len(images), desc='Image encoding') :
        img = Image.open(img_path)
        with torch.no_grad():
            emb = model.encode_image([img])[0].numpy()
            img_embs.append(emb)
    img_embs = np.array(img_embs)

    txt_embs = []
    for i, sent in tqdm(enumerate(texts), total=len(texts), desc='Text encoding') :
        with torch.no_grad():
            emb = model.encode_text([sent])[0].numpy()
            txt_embs.append(emb)
    txt_embs = np.array(txt_embs)

    sims = np.dot(img_embs, txt_embs.transpose())

    print("============Image to Text=============")
    res = calc_metric(sims, dataset.get_ground_trues('i2t'))
    for key, value in res.items():
        print(f"{key}: {value}")
    print("============Image to Text=============")

    print("============Text to Image=============")
    res = calc_metric(sims.transpose(), dataset.get_ground_trues('t2i'))
    for key, value in res.items():
        print(f"{key}: {value}")



if __name__ == "__main__":
    evaluate()

