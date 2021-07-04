# coding: utf-8
import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '../../'))
from m4.models import BERT
from m4.poolers import BertPooler
from m4.utils.similarity import cosine_sim
from m4.losses import BatchPostivePairLoss


def test_BERT():
    model_dir = '/root/data/private/models/m3m/huggingface_transformers/uer/chinese_roberta_L-2_H-128'
    model = BERT(model_dir=model_dir)
    pooler = BertPooler('avg')
    loss = BatchPostivePairLoss()
    model.eval()
    pooler.eval()
    input_text_batch = [
        'I am the first sentence',
        'I am the second sentence',
        'Third sentence',
    ]
    res = model(input_text_batch)
    embs = pooler(res)
    loss_value = loss(embs, embs)

    # model.save_model('./chinese_roberta_tiny')
    print(loss_value)


if __name__ == "__main__":
    test_BERT()
