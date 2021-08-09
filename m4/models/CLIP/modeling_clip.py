# coding: utf-8
"""
CLIP model

@ref: https://github.com/huggingface/transformers/blob/master/src/transformers/models/clip/modeling_clip.py
"""
import os
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Dict, Union
from ..model_base import HFModel
from ...utils import to_device


class CLIP(HFModel):

    def __init__(self,
                 model_dir: Union[str, os.PathLike],
                 text_max_length: int = 77
        ):
        super().__init__(model_dir)
        self.preprocessor = CLIPProcessor.from_pretrained(model_dir)
        self.encoder = CLIPModel.from_pretrained(model_dir)
        self.text_max_length = text_max_length

    def save_model(self,
                   model_dir: Union[str, os.PathLike],
                   epoch_num: int = None,
                   step_num : int = None
        ):
        if model_dir.endswith('/'):
            model_dir = model_dir.strip('/')
        if epoch_num is not None:
            model_dir = model_dir + '-epoch-{}'.format(epoch_num)
        if step_num is not None:
            model_dir = model_dir + '-step-{}'.format(step_num)
        if os.path.exists(model_dir):
            pass
        else:
            os.mkdir(model_dir)
        self.preprocessor.save_pretrained(model_dir)
        self.encoder.save_pretrained(model_dir)
        return model_dir

    def forward(self,
                texts: List[str],
                images: List[Image.Image],
                padding: bool = True,
                return_loss: bool = False
        ) -> Dict[str, Union[Dict, torch.Tensor]]:
        preprocess_res = self.preprocess(texts, images, padding)
        encode_res = self.encode(preprocess_res, return_loss=return_loss)
        output_dict = {
            'preprocess_res': preprocess_res,
            'encode_res'    : encode_res,
            'text_embeds'   : encode_res.text_embeds,
            'image_embeds'  : encode_res.image_embeds,
            'logits_per_image' : encode_res.logits_per_image,
            'logits_per_text'  : encode_res.logits_per_text
        }
        if return_loss:
            output_dict['loss'] = encode_res.loss
        return output_dict

    def preprocess(self,
                   texts: List[str],
                   images: List[Image.Image],
                   padding: bool = True
        ):
        preprocess_res = self.preprocessor(
            text   = texts,
            images = images,
            return_tensors = 'pt',
            padding        = padding,
            truncation     = True,
            max_length     = self.text_max_length).to(self.device)
        return preprocess_res

    def encode(self,
               preprocess_res: Dict[str, torch.Tensor],
               return_loss: bool = False
        ):
        encode_res = self.encoder(
            **preprocess_res,
            return_dict = True,
            output_hidden_states = True,
            return_loss = return_loss)
        encode_res = to_device(encode_res, self.device)
        return encode_res

    def encode_text(self,
                    texts: List[str],
                    padding: bool = True
        ) -> torch.Tensor:
        preprocess_res = self.preprocessor(
            text = texts,
            return_tensors ='pt',
            padding        = padding,
            truncation     = True,
            max_length     = self.text_max_length)
        text_features = self.encoder.get_text_features(
            **preprocess_res,
            return_dict = True,
            output_hidden_states = False)
        text_embeds = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_image(self,
                     images: List[Image.Image]
        ) -> torch.Tensor:
        preprocess_res = self.preprocessor(
            images = images,
            return_tensors ='pt')
        image_features = self.encoder.get_image_features(
            **preprocess_res,
            return_dict = True,
            output_hidden_states = False)
        image_embeds = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_embeds






