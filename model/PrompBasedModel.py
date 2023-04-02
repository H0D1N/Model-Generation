import torch.nn as nn
from .AdaResNet import Ada_ResNet,SelectedResNetBlock
from .TGNetwork import get_TGNetwork

class PromptBasedModel(nn.Module):
    def __init__(self,ada_kernel=True,is_embed=True,attn_pad=False,vocab_size=2,kernel_number=512,
                 block=SelectedResNetBlock,layers=[3,4,6,3],classifer_num=100):
        super(PromptBasedModel, self).__init__()
        self.ada_kernel=ada_kernel
        if ada_kernel:
            self.gate_network=get_TGNetwork(is_embed=is_embed,attn_pad=attn_pad,gate_len=sum(layers)*2,vocab_size=vocab_size,kernel_number=kernel_number)
        self.backbone=Ada_ResNet(ada_kernel,block,layers,kernel_number,classfier_num=classifer_num)
    def forward(self,prompt,img):
        if self.ada_kernel:
            selection=self.gate_network(prompt)
            logits=self.backbone(img,selection)
            return logits,selection
        else:
            logits=self.backbone(img)
            return logits,None