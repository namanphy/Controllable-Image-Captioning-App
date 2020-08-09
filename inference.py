""" run trained model on test set, get gt captions v.s. predicted captions """
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from easydict import EasyDict as edict
from transformers import AutoTokenizer

from src.utils import *
from src.datasets import CaptionDataset
from src.app_utils import setup_models, read_json, predict_one_caption


is_cuda = False
cfg_path = os.path.join('ckpts', 'config.json')
data_name = 'flickr8k_1_cap_per_img_1_min_word_freq'
checkpoint_file = os.path.join('ckpts', 'BEST_checkpoint_flickr8k_1_cap_per_img_1_min_word_freq.pth')


word_map_file = f'{data_folder}/WORDMAP_{data_name}.json'
word_map = read_json(word_map_file)
rev_word_map = {v: k for k, v in word_map.items()}

cfg = edict(read_json(cfg_path))
cfg.checkpoint_file = checkpoint_file
encoder, decoder = setup_models(cfg, is_cuda = is_cuda)
device = torch.device('cuda' if is_cuda and torch.cuda.is_available() else 'cpu')


def predict_captions(beam_size, length_class, data_type = 'TEST', n = -1,
                     subword = False, img_file = '', inference = False):
    assert data_type in ['INFERENCE']
    assert length_class in [0, 1, 2]

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

        predict = predict_one_caption(encoder, decoder, image, word_map, len_class, beam_size)

        # references & prediction
        # if not inference:
        #     img_cap = caps.tolist()[0]
        #     img_caption = [w for w in img_cap if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        
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
