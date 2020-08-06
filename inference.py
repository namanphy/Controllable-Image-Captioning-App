""" run trained model on test set, get gt captions v.s. predicted captions """
# import os
# import json
#
# import pandas as pd
# from tqdm import tqdm
#
# import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from transformers import AutoTokenizer

from src.models import Encoder, DecoderWithAttention
from src.datasets import CaptionDataset
from src.utils import *

# import psutil
# print("START", psutil.virtual_memory())

checkpoint_dir = './ckpts'
data_folder = './data/meta_wstyle/data_mid_clean_wonumber_wp'
data_name = 'flickr8k_1_cap_per_img_1_min_word_freq'
checkpoint_file = os.path.join(checkpoint_dir, 'BEST_checkpoint_flickr8k_1_cap_per_img_1_min_word_freq.pth')
word_map_file = f'{data_folder}/WORDMAP_{data_name}.json'

with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

cfg_path = os.path.join(checkpoint_dir, 'config.json')
with open(cfg_path, 'r') as f:
    cfg = json.load(f)

attention_dim = cfg['attention_dim']
emb_dim = cfg['emb_dim']
decoder_dim = cfg['decoder_dim']
vocab_size = len(word_map) 
dropout = cfg['dropout']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder()
decoder = DecoderWithAttention(attention_dim = attention_dim,
                               embed_dim = emb_dim,
                               decoder_dim = decoder_dim,
                               vocab_size = len(word_map),
                               dropout = dropout)

checkpoint = torch.load(checkpoint_file)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.to(device)
decoder.to(device)
encoder.eval()
decoder.eval()
# print("END", psutil.virtual_memory())


def predict_captions(beam_size, length_class, data_type = 'TEST', n = -1, subword = False, img_file='', inference=False):
    assert data_type in ['INFERENCE']
    assert length_class in [0, 1, 2]

    # if inference:
    #     assert img_file
    # if img_file:
    #     assert inference

    len_class = torch.as_tensor([length_class]).long().to(device)

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    tfms = transforms.Compose([normalizer])
    dataset = CaptionDataset(img_file=img_file, transform=tfms, inference=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    tokenizer = None
    if subword:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.add_tokens('@username')

    results = []
    for i, (image, caps, _, allcaps, gt_len_class, img_ids) in enumerate(tqdm(dataloader)):
        
        if i == n:
            break

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

        # references & prediction
        # if not inference:
        #     img_cap = caps.tolist()[0]
        #     img_caption = [w for w in img_cap if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        predict = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        
        if subword:
            assert tokenizer is not None
            # if not inference:
            #     img_caption = [rev_word_map[s] for s in img_caption]
            #     ref_enc = tokenizer.convert_tokens_to_ids(img_caption)
            #     img_caption = tokenizer.decode(ref_enc)

            predict = [rev_word_map[s] for s in predict]
            pred_enc = tokenizer.convert_tokens_to_ids(predict)
            predict = tokenizer.decode(pred_enc)            
        else:
            # if not inference:
            #     img_caption = ' '.join([rev_word_map[s] for s in img_caption])
            predict = ' '.join([rev_word_map[s] for s in predict])
        
        result = {
            'img_id': img_ids[0],
            'length_class': gt_len_class , #  int(gt_len_class.cpu().squeeze()),
            'data_type': data_type,
            # 'gt_caption': img_caption,
            f'length_class_{length_class}': predict
            }
        results.append(result)
    return results


if __name__ == '__main__':
    beam_size = 10
    for data_type in ['INFERENCE']:  # other data types - 'TRAIN', 'VAL', 'TEST'
        result_csv = os.path.join(checkpoint_dir, f'benchmarks_{data_type.lower()}.csv')
        
        agg_results = []
        for len_class in [0, 1, 2]:
            print(f'data_type: {data_type}, beam size: {beam_size}, length class: {len_class}')
            results = predict_captions(
                    beam_size, len_class, 
                    data_type = data_type, 
                    n = 200, subword = True,
                    img_file='C:/Users/naman/Pictures/mine.jpg', inference=True
                    )

            if agg_results == []:
                agg_results = results
            else:
                for i in range(len(agg_results)):
                    agg_results[i].update(results[i])

        result_df = pd.DataFrame(agg_results)
        result_df.to_csv(result_csv, index = False)
        print(f'result csv written: {result_csv}')
