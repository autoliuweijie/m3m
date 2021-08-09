# coding: utf-8
import os
import json
from torch.utils.data import Dataset
from PIL import Image


class Flrick30k(Dataset):

    def __init__(self,
                 image_dir: str,
                 annot_path: str,
                 split: str,
                 mode: str='i2t'):
        super().__init__()
        assert split in ['train', 'val', 'test']
        assert mode in ['i2t', 't2i']
        self.split = split
        self.mode = mode
        self.images, self.texts, self.gt_i2t, self.gt_t2i = self._load(annot_path, image_dir)

    def set_mode(self,
                 mode: str):
        assert mode in ['i2t', 't2i']
        self.mode = mode

    def get_all_texts(self):
        return self.texts

    def get_all_images(self):
        return self.images

    def get_ground_trues(self, mode=None):
        if mode is None:
            mode = self.mode
        if mode == 'i2t':
            return self.gt_i2t
        elif mode == 't2i':
            return self.gt_t2i
        else:
            raise Exception('Unknown mode = {}'.format(mode))

    def __getitem__(self,
                    idx):
        if self.mode == 'i2t':
            return self.images[idx]
        else:
            return self.texts[idx]

    def __len__(self):
        if self.mode == 'i2t':
            return len(self.images)
        else:
            return len(self.texts)

    def _load(self,
              annot_path,
              image_dir):

        images = []
        texts = []
        gt_i2t = []

        txt_idx = 0
        with open(annot_path, 'r') as fin:
            for item in json.load(fin)["images"]:
                if item['split'] == self.split:
                    images.append(os.path.join(image_dir, item['filename']))
                    tmp_gt_i2t = []
                    for sent in item['sentences']:
                        tmp_gt_i2t.append(txt_idx)
                        texts.append(sent['raw'])
                        txt_idx += 1
                    gt_i2t.append(tmp_gt_i2t)

        gt_t2i = [[] for _ in range(txt_idx)]
        for ii, gt in enumerate(gt_i2t):
            for ti in gt:
                gt_t2i[ti].append(ii)

        return images, texts, gt_i2t, gt_t2i


if __name__ == "__main__":
    test()
