import math
import torch
import torch.nn as nn
from .TransformerBase import MultiHeadAttention,PoswiseFeedForwardNet,get_att_pad_mask

class Encoder(nn.Module):
    def __init__(self,is_embed,d_model,d_k,d_v,d_ff,n_heads,n_layers,feature_size,vocab_size,att_pad):
        super(Encoder,self).__init__()
        self.is_embed=is_embed
        self.att_pad=att_pad

        if self.is_embed:
            self.src_embed=nn.Embedding(vocab_size,d_model)
        else:
            self.src_embed=nn.Linear(feature_size,d_model,bias=False)
        self.layers=nn.ModuleList(EncoderLayer(d_model,d_k,d_v,d_ff,n_heads) for _ in range(n_layers))

    def forward(self,enc_inputs):
        if self.is_embed:
            x=self.src_embed(enc_inputs)
        enc_att_pad_mask=None
        if self.att_pad:
            enc_att_pad_mask=get_att_pad_mask(enc_inputs,enc_inputs)
        enc_self_attns=[]
        for layer in self.layers:
            x,enc_self_attn=layer(x,enc_att_pad_mask)
            enc_self_attns.append(enc_self_attn)
        return x,enc_self_attns





class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_k,d_v,d_ff,n_heads):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn=MultiHeadAttention(d_model,d_k,d_v,n_heads)
        self.pos_ffn=PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self,enc_inputs,enc_self_attn_mask=None):
        enc_outputs,attn=self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs=self.pos_ffn(enc_outputs)
        return enc_outputs,attn









