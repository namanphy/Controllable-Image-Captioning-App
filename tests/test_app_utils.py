import warnings
warnings.filterwarnings("ignore")

import time
import os
import pytest

import emoji
import torch
import torch.nn as nn

from easydict import EasyDict as edict
from src.app_utils import read_json
from src.app_utils import setup_models, open_image, tfms_image, setup_tokenizer
from src.app_utils import output_caption


@pytest.fixture(scope = 'module')
def encoder_decoder():
    is_cuda = False
    cfg_path = './ckpts/config.json'

    cfg = read_json(cfg_path)
    cfg = edict(cfg)
    
    encoder, decoder = setup_models(cfg, is_cuda = is_cuda)
    assert isinstance(encoder, nn.Module)
    assert isinstance(decoder, nn.Module)

    print('encoder and decoder are set')
    return encoder, decoder


@pytest.fixture(scope = 'module')
def image():
    img_fn = './demo/demo_img1.jpg'
    img = open_image(img_fn, demo_flag=True)
    img = tfms_image(img)

    img_shape = [s for s in img.shape]
    assert isinstance(img, torch.Tensor)
    assert img_shape == [1, 3, 256, 256]

    print('image is set')
    return img
    

@pytest.fixture(scope = 'module')
def word_maps():
    word_map_file = './ckpts/WORDMAP_flickr8k_1_cap_per_img_1_min_word_freq.json'
    word_map = read_json(word_map_file)
    rev_word_map = {v: k for k, v in word_map.items()}

    assert isinstance(word_map, dict)
    assert isinstance(rev_word_map, dict)

    print('word_map and rev_word_map is set')
    return word_map, rev_word_map


@pytest.fixture(scope = 'module')
def tokenizer(word_maps):
    word_map, _ = word_maps
    tokenizer = setup_tokenizer(word_map)

    assert tokenizer is not None
    print('tokenizer is set')
    return tokenizer


def test_predict_one_caption(encoder_decoder, image, tokenizer, word_maps):
    beam_size = 10

    encoder, decoder = encoder_decoder
    word_map, rev_word_map = word_maps
    device = torch.device('cuda' if next(encoder.parameters()).is_cuda else 'cpu')
    
    for len_class in [0, 1, 2]:
        for emoji_class in [0, 1]:
            start = time.time()

            print(f'running len_class: {len_class}, emoji_class: {emoji_class}')
            caption, pred_ids, pred_subwords = output_caption(
                encoder, decoder, image, word_map, rev_word_map,
                tokenizer, len_class, emoji_class, beam_size
                )
            end = time.time()
            
            assert isinstance(pred_ids, list)
            assert isinstance(pred_subwords, list)
            assert isinstance(caption, str)

            print(f'caption ids: {pred_ids}')
            print(f'caption: {caption}')            
            print(f'runtime: {end - start} s \n')
