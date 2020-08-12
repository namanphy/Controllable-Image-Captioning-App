import os
import time

import cv2
import numpy as np
import torch
import streamlit as st

from src.app_utils import *
st.set_option('deprecation.showfileUploaderEncoding', False)

DEMO_IMAGE = './demo/test_image.jpg'
CONFIG_FILE = './ckpts/config.json'
CHECKPOINT_FILE = './ckpts/BEST_checkpoint_flickr8k_1_cap_per_img_1_min_word_freq.pth'
WORD_MAP_FILE = './ckpts/WORDMAP_flickr8k_1_cap_per_img_1_min_word_freq.json'
IS_CUDA = False


# preset models and tokenizer
@st.cache
def get_models():
    cfg = read_json(CONFIG_FILE)
    cfg = edict(cfg)
    cfg.checkpoint_file = CHECKPOINT_FILE
    encoder, decoder = setup_models(cfg, is_cuda = IS_CUDA)
    print('model received')
    return encoder, decoder


@st.cache
def get_tokenizer():
    tokenizer = setup_tokenizer()
    print('tokenizer received')
    return tokenizer


@st.cache
def get_word_maps():
    word_map = read_json(WORD_MAP_FILE)
    rev_word_map = {v: k for k, v in word_map.items()}
    print('word map received')
    return word_map, rev_word_map

encoder, decoder = get_models()
device = torch.device('cuda' if next(encoder.parameters()).is_cuda else 'cpu')
tokenizer = get_tokenizer()
word_map, rev_word_map = get_word_maps()


st.title('Instagram-like Controllable Image Captioning')

# user input on sidebar
st.sidebar.header('User Control')
st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text("testing testing")
emoji_class = st.sidebar.checkbox('add emoji') * 1
sentence_class = st.sidebar.selectbox(
    'sentence class', 
    ('short', 'mid', 'long')
)
st.sidebar.text(" \n")
img_buffer = st.sidebar.file_uploader(
    'upload image (png/ jpg/ jpeg)', 
    type = ['png', 'jpg', 'jpeg']
)

img_fn = img_buffer if img_buffer is not None else DEMO_IMAGE
np_img = open_image(img_fn)
st.image(np_img, use_column_width = True)


# run model and propagate to frontend
tensor_img = tfms_image(np_img)
class_map = {'short': 0, 'mid': 1, 'long': 2}
sentence_class = class_map[sentence_class]
sentence_class = torch.as_tensor([sentence_class]).long().to(device)

predict = predict_one_caption(
    encoder, decoder, tensor_img, word_map, 
    len_class = sentence_class, beam_size = 10
)
predict = [rev_word_map[s] for s in predict]
pred_enc = tokenizer.convert_tokens_to_ids(predict)
caption = tokenizer.decode(pred_enc)


# load and show generated caption to frontend
st.markdown(caption)