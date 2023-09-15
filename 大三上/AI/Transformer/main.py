import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

seaborn.set_context(context='talk')


# basic architecture
class Encoderdecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)  # embedding and mask
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
class Generator(nn.Module):
    # standard linear + softmax step
    # d_model: in features; vocab: out features
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)   # standard projection
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Normalization
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # giving parameters that enable to train
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Encoder
class Encoder(nn.Module):
    # 6 layers then norm
    # and one residual connect, then norm
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
 
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = dropout
    
    def forward(self, x, sublayer):
        # residual connect
        # it will norm x when it inputs in net at first
        return x + self.dropout(sublayer(self.norm(x)))     # maybe norm(x + dropout(sublayer(x)))
    
class EncoderLayer(nn.Module):
    # multi-attention head + subconnect + feed forward + subconnect
    def __init__(self, size, self_attention, feed_forward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        # subplyaers.forward() need to pass in a x and a function, so use lambda
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)


# Decoder
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attention, src_attention, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attention = self_attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attention(x, m, m, src_mask))
        return self.sublayers[2](x, self.feed_forward)

# mask
# it should be known that the decoder should not be noticed about the output after
# y1 = G(C), y2 = G(C, y1), y3 = G(C, y1, y2), ...
def subsequent_mask(size):
    attention_shape = (1, size, size)
    

