import torch
import torch.nn as nn
from .TransformerDecoder import Decoder
from .TransformerEncoder import Encoder

def get_TGNetwork(d_model=512,d_ff=2048,d_k=64,d_v=64,n_layers=4,n_heads=8,attn_pad=False,is_embed=False,
                  feature_size=0,vocab_size=100,gate_len=34,select_embbed_len=128,kernel_number=512):
    model=TGNetwork(d_model,d_ff,d_k,d_v,n_layers,n_heads,attn_pad,is_embed,
                    feature_size,vocab_size,gate_len,select_embbed_len,kernel_number)
    return model


class TGNetwork(nn.Module):
    def __init__(self,d_model,d_ff,d_v,d_k,n_layers,n_heads,attn_pad=False,is_embed=False,
                 feature_size=0,vocab_size=0,gate_len=0,select_embbed_len=0,kernel_number=0):
        super(TGNetwork,self).__init__()
        self.gate_len=gate_len
        self.select_embed_len=select_embbed_len

        self.encoder=Encoder(is_embed,d_model,d_k,d_v,d_ff,n_heads,n_layers,feature_size,vocab_size,attn_pad)
        self.decoder=Decoder(d_model,d_k,d_v,d_ff,n_heads,n_layers,attn_pad)

        self.TaskLinear=nn.Sequential(
            nn.Linear(d_model,4*d_model,bias=False),
            nn.ReLU(),
            nn.Linear(4*d_model,gate_len*select_embbed_len,bias=False))

        self.LayerGating=nn.Linear(select_embbed_len,kernel_number)

    def Improved_SemHash(select_weight):
        # select_weight:[batch_size,length,kernel_number]
        binary_selection = torch.lt(torch.zeros_like(select_weight), select_weight).float()
        gradient_selection = torch.max(torch.zeros_like(select_weight), torch.min(torch.ones_like(select_weight), (
                1.2 * torch.sigmoid(select_weight) - 0.1)))
        d = binary_selection + gradient_selection - gradient_selection.detach()
        return d

    def forward(self,prompt,):
        #prompt:[batchsize,len]
        enc_outputs,_=self.encoder(prompt)
        decoder_inputs=torch.zeros(prompt.size(0),1).int()
        dec_outputs,_,_=self.decoder(decoder_inputs,prompt,enc_outputs)

        dec_output=dec_outputs.squeeze()

        selection_embbeding=self.TaskLinear(dec_output)

        selection_embbeding=selection_embbeding.view(-1,self.gate_len,self.select_embed_len)

        layer_selection=self.LayerGating(selection_embbeding)

        layer_selection=self.Improved_SemHash(layer_selection)

        return layer_selection







