# coding: utf-8
import os
import sys
import logging
import requests
import torch
from PIL import Image

m4_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(m4_dir)

from m4.models import ViT


def main():
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(image_url, stream=True).raw)

    model_dir = '/root/data/private/models/huggingface_transformers/google/vit-base-patch16-224-in21k'
    model = ViT(model_dir=model_dir)
    model.eval()

    with torch.no_grad():
        res = model(image)
    print(res['encode_res'].last_hidden_state.size())
    print(res['encode_res'].pooler_output.size())


if __name__ == "__main__":
    main()

