import os
import json

import cv2
import numpy as np
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from transformers import AutoTokenizer

from src.models import get_encoder_decoder


def read_json(json_path):
    assert json_path, f'{json_path} not exist'
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def setup_models(cfg, is_cuda):
    encoder, decoder = get_encoder_decoder(cfg)
    
    device = torch.device('cuda' if is_cuda and torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def setup_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens('@username')
    return tokenizer


def open_image(img_fn, demo_flag):
    if demo_flag:
        img = cv2.imread(img_fn)
    else:
        img = cv2.imdecode(np.fromstring(img_fn.read(), np.int8), 1)
    # from pdb import set_trace
    # set_trace()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis = 2)
    img = cv2.resize(img, (256, 256))

    assert img.shape == (256, 256, 3)
    return img


def tfms_image(img):
    img = img.transpose(2, 0, 1)
    img = torch.FloatTensor(img / 255.)
    normalizer = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                      std = [0.229, 0.224, 0.225])
    tfms = transforms.Compose([normalizer])
    return tfms(img).unsqueeze(0)


def predict_one_caption(encoder, decoder, image, word_map, len_class, beam_size):
    device = torch.device('cuda' if next(encoder.parameters()).is_cuda else 'cpu')
    vocab_size = len(word_map)
    k = beam_size

    # move to GPU device, if available
    image = image.to(device)  # (1, 3, 256, 256)

    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    # enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Lists to store completed sequences and scores
    complete_seqs = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    with torch.no_grad():
        h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    with torch.no_grad():
        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            style_embedding = decoder.length_class_embedding(len_class)
            style_embed_dim = style_embedding.size(-1)
            style_embedding = style_embedding.expand(k, style_embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, style_embedding, awe], dim = 1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim = 1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    predict = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
    return predict
