import torch
import torch.nn as nn

from easydict import EasyDict as edict
from src.app_utils import read_json
from src.app_utils import setup_models, setup_image, setup_tokenizer


def test_get_encoder_decoder():
    cfg_path = './ckpts/config.json'
    cfg = read_json(cfg_path)
    cfg = edict(cfg)
    cfg.checkpoint_file = './ckpts/BEST_checkpoint_flickr8k_1_cap_per_img_1_min_word_freq.pth'
    
    encoder, decoder = setup_models(cfg, is_cuda = False)
    assert isinstance(encoder, nn.Module)
    assert isinstance(decoder, nn.Module)


def test_setup_image():
    img_fn = './tests/test_image.jpg'
    img = setup_image(img_fn)

    img_shape = [s for s in img.shape]
    assert isinstance(img, torch.Tensor)
    assert img_shape == [1, 3, 256, 256]
    

def test_setup_tokenizer():
    tokenizer = setup_tokenizer()
    assert tokenizer is not None