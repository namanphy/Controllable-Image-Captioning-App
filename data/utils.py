import os
import json


def read_txt(path):
    with open(path, 'r') as f:
        obj = f.read().splitlines()
    return obj


def read_json(json_path):
    assert os.path.isfile(json_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(obj, json_path):
    with open(json_path, 'w') as f:
        json.dump(obj, f, indent = 2)
    return None


def decode_one_sample(sample, vocab):
    fn, _, _, _, token_seq = sample.split(',')
    fn = fn[:-4]
    token_seq = [int(i) for i in token_seq.split('_')]
    word_seq = [vocab[i] for i in token_seq]
    return fn, token_seq, word_seq

