"""
different kinds of dropouts for AWD-LSTM model
mainly reference from https://github.com/fastai/fastai2/blob/master/fastai2/text/models/awdlstm.py
with minor change
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a


def dropout_mask(x, sz, p):
    return x.new(*sz).bernoulli_(1-p).div_(1-p)


class RNNDropout(nn.Module):
    """ with 1/(1-p) scaling """
    def __init__(self, p = 0.5):
        super(RNNDropout, self).__init__()
        self.p = p
    
    def forward(self, x, reset_mask):
        batch_size = x.size(0)

        if not self.training or self.p == 0:
            return x
        
        # flag to use preceding dropout mask or a new mask
        if reset_mask:
            self.mask = dropout_mask(x.data, (batch_size, x.size(1)), self.p) # (batch size, hidden dim)
        else:
            if not hasattr(self, 'mask'):
                self.mask = dropout_mask(x.data, (batch_size, x.size(1)), self.p) # (batch size, hidden dim)
        return x * self.mask[:batch_size]


class InputDropout(nn.Module):
    """ with 1/(1-p) scaling """
    def __init__(self, p = 0.5):
        super(InputDropout, self).__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0.: 
            return x

        return x * dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)

class WeightDropout(nn.Module):
    def __init__(self, module, weight_p, layer_names = ['weight_hh']):
        super(WeightDropout, self).__init__()
        self.module = module
        self.weight_p = weight_p
        self.layer_names = layer_names

        for layer in self.layer_names:
            w = getattr(self.module, layer)
            delattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            setattr(self.module, layer, F.dropout(w.data, p = self.weight_p, training = False))

    def _setweights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            # note 1: when self.training = False, F.dropout disappear, thus nn.Parameter is set!!
            # note 2: F.dropout imposes scaling on non-masked entry
            setattr(self.module, layer, F.dropout(raw_w, p = self.weight_p, training = self.training))

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            setattr(self.module, layer, F.dropout(raw_w.data, p = self.weight_p, training = False))
        if hasattr(self.module, 'reset'): self.module.reset()

    def forward(self, *args, reset_mask = True):
        if reset_mask and self.training:
            self._setweights()
        elif not self.training:
            self.reset()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.module.forward(*args)


class EmbeddingDropout(nn.Module):
    """ with 1/(1-p) scaling """
    def __init__(self, emb, embed_p):
        super(EmbeddingDropout, self).__init__()
        self.emb = emb
        self.embed_p = embed_p

    def forward(self, words, scale = None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0),1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else: 
            masked_embed = self.emb.weight

        if scale: 
            masked_embed.mul_(scale)

        return F.embedding(words, masked_embed, ifnone(self.emb.padding_idx, -1), 
                           self.emb.max_norm, self.emb.norm_type, 
                           self.emb.scale_grad_by_freq, self.emb.sparse)
