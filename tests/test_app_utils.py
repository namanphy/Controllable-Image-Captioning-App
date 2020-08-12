import time
import os
import pytest

import torch
import torch.nn as nn

from easydict import EasyDict as edict
from src.app_utils import read_json
from src.app_utils import setup_models, open_image, tfms_image, setup_tokenizer
from src.app_utils import predict_one_caption


@pytest.fixture(scope = 'module')
def encoder_decoder():
    is_cuda = False
    checkpoint_file = './ckpts/BEST_checkpoint_flickr8k_1_cap_per_img_1_min_word_freq.pth'
    cfg_path = './ckpts/config.json'

    cfg = read_json(cfg_path)
    cfg = edict(cfg)
    cfg.checkpoint_file = checkpoint_file
    
    encoder, decoder = setup_models(cfg, is_cuda = is_cuda)
    assert isinstance(encoder, nn.Module)
    assert isinstance(decoder, nn.Module)

    print('encoder and decoder are set')
    return encoder, decoder


@pytest.fixture(scope = 'module')
def image():
    img_fn = './demo/test_image.jpg'
    img = open_image(img_fn)
    img = tfms_image(img)

    img_shape = [s for s in img.shape]
    assert isinstance(img, torch.Tensor)
    assert img_shape == [1, 3, 256, 256]

    print('image is set')
    return img
    

@pytest.fixture(scope = 'module')
def tokenizer():
    tokenizer = setup_tokenizer()

    assert tokenizer is not None
    print('tokenizer is set')
    return tokenizer


@pytest.fixture(scope = 'module')
def word_maps():
    word_map_file = './ckpts/WORDMAP_flickr8k_1_cap_per_img_1_min_word_freq.json'
    word_map = read_json(word_map_file)
    rev_word_map = {v: k for k, v in word_map.items()}

    assert isinstance(word_map, dict)
    assert isinstance(rev_word_map, dict)

    print('word_map and rev_word_map is set')
    return word_map, rev_word_map


def test_predict_one_caption(encoder_decoder, image, tokenizer, word_maps):
    beam_size = 10

    encoder, decoder = encoder_decoder
    word_map, rev_word_map = word_maps
    device = torch.device('cuda' if next(encoder.parameters()).is_cuda else 'cpu')
    
    for len_class in [0, 1, 2]:
        start = time.time()
        print(f'running len_class: {len_class}')

        len_class = torch.as_tensor([len_class]).long().to(device)    
        predict = predict_one_caption(encoder, decoder, image, word_map, 
                                    len_class = len_class, beam_size = beam_size)
        print(f'predict: {predict}')
        predict =[rev_word_map[s] for s in predict]
        assert isinstance(predict, list)

        pred_enc = tokenizer.convert_tokens_to_ids(predict)
        caption = tokenizer.decode(pred_enc)
        print(f'caption: {caption}')
        assert isinstance(caption, str)

        end = time.time()
        print(f'runtime: {end - start} s')
