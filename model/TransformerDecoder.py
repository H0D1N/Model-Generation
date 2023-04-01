import torch
import torch.nn as nn
from .TransformerBase import get_att_pad_mask,PoswiseFeedForwardNet,MultiHeadAttention
class Decoder(nn.Module):
    def __init__(self,d_model,d_k,d_v,d_ff,n_heads,n_layers,attn_pad):
        super(Decoder, self).__init__()
        self.attn_pad=attn_pad
        self.tgt_emb = nn.Embedding(1, d_model)#只需要给开始符号即可
        self.layers = nn.ModuleList([DecoderLayer(d_model,d_k,d_v,d_ff,n_heads) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]

        dec_enc_attn_mask=None
        if self.attn_pad:
            dec_enc_attn_mask = get_att_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class DecoderLayer(nn.Module):
    def __init__(self,d_model,d_k,d_v,d_ff,n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model,d_k,d_v,n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model,d_k,d_v,n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_enc_attn_mask=None):
        # dec_inputs: [batch_size, 1, d_model],enc_outputs: [batch_size, src_len, d_model]
        #dec_enc_attn_mask: [batch_size, 1, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,dec_inputs)
        # dec_outputs: [batch_size, 1, d_model]  dec_self_attn: [batch_size, n_heads, 1, 1]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                enc_outputs, dec_enc_attn_mask)
        return dec_outputs, dec_self_attn, dec_enc_attn
