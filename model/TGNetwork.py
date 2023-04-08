import torch
import torch.nn as nn
from .TransformerEncoder import Encoder

def get_TGNetwork(task_embed=True,d_model=100,d_ff=2048,d_k=64,d_v=64,n_layers=4,n_heads=8,attn_pad=False,is_embed=False,
                  feature_size=0,vocab_size=100,gate_len=34,select_embbed_len=128,kernel_number=512):
    model=TGNetwork(d_model,d_ff,d_k,d_v,n_layers,n_heads,attn_pad,is_embed,
                    feature_size,vocab_size,gate_len,select_embbed_len,kernel_number,task_embed)
    return model


class TGNetwork(nn.Module):
    def __init__(self,d_model,d_ff,d_v,d_k,n_layers,n_heads,attn_pad=False,is_embed=False,
                 feature_size=0,vocab_size=0,gate_len=0,select_embbed_len=0,kernel_number=0,task_embed=True):
        super(TGNetwork,self).__init__()
        self.gate_len=gate_len
        self.select_embed_len=select_embbed_len
        self.task_embed=task_embed

        if task_embed:
            self.encoder=Encoder(is_embed,d_model,d_k,d_v,d_ff,n_heads,n_layers,feature_size,vocab_size,attn_pad)

        self.TaskLinear=nn.Sequential(
            nn.Linear(d_model,gate_len*16,bias=False),
            nn.BatchNorm1d(gate_len*16),
            nn.ReLU(),
            nn.Linear(gate_len*16, gate_len * 32, bias=False),
            nn.BatchNorm1d(gate_len * 32),
            nn.ReLU(),
            nn.Linear(gate_len * 32, gate_len * 64, bias=False),
            nn.BatchNorm1d(gate_len * 64),
            nn.ReLU(),
            nn.Linear(gate_len*128,gate_len*select_embbed_len,bias=False))

        self.LayerGating=nn.Sequential(
            nn.Linear(select_embbed_len,2*select_embbed_len,bias=False),
            nn.BatchNorm1d(2*select_embbed_len),
            nn.ReLU(),
            nn.Linear(2*select_embbed_len, 4 * select_embbed_len, bias=False),
            nn.BatchNorm1d(4 * select_embbed_len),
            nn.ReLU(),
            nn.Linear(4 * select_embbed_len, 8 * select_embbed_len, bias=False),
            nn.BatchNorm1d(8 * select_embbed_len),
            nn.ReLU(),
            nn.Linear(8*select_embbed_len,kernel_number,bias=False))

    def Improved_SemHash(self,select_weight):
        # select_weight:[batch_size,length,kernel_number]
        binary_selection = torch.lt(torch.zeros_like(select_weight), select_weight).float()
        gradient_selection = torch.max(torch.zeros_like(select_weight), torch.min(torch.ones_like(select_weight), (
                1.2 * torch.sigmoid(select_weight) - 0.1)))
        d = binary_selection + gradient_selection - gradient_selection.detach()
        return d

    def forward(self,prompt,):
        #prompt:[batchsize,prompt_len]
        if self.task_embed:
            enc_outputs,_=self.encoder(prompt)

        #enc_outputs：[batch_size,prompt_len,d_model]

            task_cls=enc_outputs[:,0,:]
        else:
            task_cls=prompt

        layer_encoding=self.TaskLinear(task_cls)

        layer_encoding=layer_encoding.view(-1,self.gate_len,self.select_embed_len)

        #selection_embbeding:[batchsize,layer_len,select_layer_encoding_len]

        layer_selection=self.LayerGating(layer_encoding)

        layer_selection=self.Improved_SemHash(layer_selection)

        return layer_selection







