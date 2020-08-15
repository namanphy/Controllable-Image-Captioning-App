import os
import time

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
import streamlit as st
from PIL import Image

from src.app_utils import *
st.set_option('deprecation.showfileUploaderEncoding', False)

DEMO_IMAGE = './demo/demo_img1.jpg'
CONFIG_FILE = './ckpts/config.json'
IS_CUDA = False
BEAM_SIZE = 10
SENTENCE_CLASS_MAP = {'Short Caption': 0, 'Mid Caption': 1, 'Long Caption': 2}
EMOJI_CLASS_MAP = {'With Emoji': 1, 'Without Emoji': 0}
PADDING_CODE = '&nbsp;'
DEFAULT_PADDING = PADDING_CODE * 10


@st.cache(show_spinner = False, allow_output_mutation = True)
def get_models():
    cfg = edict(read_json(CONFIG_FILE))
    encoder, decoder = setup_models(cfg, is_cuda = False)
    print('model received')
    return encoder, decoder


@st.cache(show_spinner = False, allow_output_mutation = True)
def get_tokenizer():
    tokenizer = setup_tokenizer(word_map)
    print('tokenizer received')
    return tokenizer


@st.cache(show_spinner = False, allow_output_mutation = True)
def get_word_maps():
    cfg = edict(read_json(CONFIG_FILE))
    word_map_file = cfg.word_map_file
    word_map = read_json(word_map_file)
    rev_word_map = {v: k for k, v in word_map.items()}
    print('word map received')
    return word_map, rev_word_map


# preset models and tokenizer etc.
encoder, decoder = get_models()
device = torch.device('cuda' if next(encoder.parameters()).is_cuda else 'cpu')
word_map, rev_word_map = get_word_maps()
tokenizer = get_tokenizer() # set tokenizer only after word_map


# user input on sidebar
st.sidebar.header('Step 1: Upload Image')
img_buffer = st.sidebar.file_uploader(
    '',
    type = ['png', 'jpg', 'jpeg']
)
st.sidebar.text(" \n")
st.sidebar.text(" \n")

st.sidebar.header('Step 2: Select Your Flavors')
st.sidebar.text(" \n")
len_choices = tuple([k for k in SENTENCE_CLASS_MAP.keys()])
sentence_class = st.sidebar.selectbox( '', len_choices)
emoji_choices = tuple([k for k in EMOJI_CLASS_MAP.keys()])
emoji_class = st.sidebar.selectbox('', emoji_choices)
st.sidebar.text(" \n")
st.sidebar.text(" \n")

st.sidebar.header('Step 3: Generate Caption!')
st.sidebar.text(" \n")
is_run = st.sidebar.button('RUN')


# propagate user input to model run
img_fn, demo_flag = (img_buffer, False) if img_buffer is not None else (DEMO_IMAGE, True)
np_img = open_image(img_fn, demo_flag)
caption = emoji.emojize(f"{DEFAULT_PADDING} :backhand_index_pointing_left: {PADDING_CODE} Press RUN Button to Generate Caption")

if is_run:
    tensor_img = tfms_image(np_img)
    sentence_class = SENTENCE_CLASS_MAP[sentence_class]
    emoji_class = EMOJI_CLASS_MAP[emoji_class]

    caption, pred_ids, _ = output_caption(
        encoder, decoder, tensor_img, 
        word_map, rev_word_map, tokenizer, 
        sentence_class, emoji_class, beam_size = BEAM_SIZE
    )
    caption = f'{DEFAULT_PADDING} <b>CAPTION</b> {caption}'


# display
h, w, _ = np_img.shape
np_img = cv2.resize(np_img, (int(h * 2), int(w * 2)))

st.image(Image.fromarray(np_img), use_column_width = False)
st.write('')
st.markdown(f""" <h3> {caption} </h3> """, 
            unsafe_allow_html = True)

