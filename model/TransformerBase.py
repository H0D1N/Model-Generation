import torch
import torch.nn as nn
import numpy as np

def get_att_pad_mask(seq_q,seq_k):
    batch_size,len_q=seq_q.size(0)
    batch_size,len_k=seq_k.size(0)
    pad_att_mask=seq_k.data.eq(0).unsqueeze(1)
    return pad_att_mask.expand(batch_size,len_q,len_k)

def get_attn_subsequence_mask(seq):
    # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention,self).__init__()
    def forward(self,Q,K,V,attn_mask=None):
        d_k=K.size(-1)
        scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
        #scores:[batch_size,n_heads,len_q,len_k]
        if attn_mask:
            scores.mask_fill_(attn_mask,-1e9)
        attn=nn.Softmax(dim=-1)(scores)
        context=torch.matmul(attn,V)
        return context,attn


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,d_k,d_v,n_heads):
        self.n_heads=n_heads
        self.d_k=d_k
        self.d_v=d_v
        super(MultiHeadAttention,self).__init__()
        self.W_Q=nn.Linear(d_model,d_k*n_heads)
        self.W_K=nn.Linear(d_model,d_k*n_heads)
        self.W_V=nn.Linear(d_model,d_v*n_heads)

        self.linear=nn.Linear(n_heads*d_v,d_model)
        self.layer_norm=nn.LayerNorm(d_model)

    def forward(self,Q,K,V,attn_mask=None):
        residual,batch_size=Q,Q.size(0)

        q_s=self.W_Q(Q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)#q_s:[batch_size,n_heads,len_q,d_k]
        k_s=self.W_K(K).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)#k_s:[batch_size,n_heads,len_k,d_k]
        v_s=self.W_V(V).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2)#v_s:[batch_size,n_heads,len_v,d_v]

        if attn_mask:
            attn_mask=attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)

        context,attn=ScaleDotProductAttention()(q_s,k_s,v_s,attn_mask)
        context=context.transpose(1,2).contiguous().view(batch_size,-1,self.n_heads*self.d_v)#context:[batch_size,len_q,n_head*d_v]
        output=self.linear(context)

        return self.layer_norm(output+residual),attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        self.layerNorm=nn.LayerNorm(d_model)

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return self.layerNorm(output + residual)  # [batch_size, seq_len, d_model]










