# coding: utf-8
from .args_parser import HfArgumentParser
from .others import set_seed, to_device, load_json, dump_json
from .pretrain import mask_token_ids
from .train import TextTraniner
