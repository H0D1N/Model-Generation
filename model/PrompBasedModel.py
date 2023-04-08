import torch.nn as nn
from .AdaResNet import ada_ResNet,Bottleneck
from .TGNetwork import get_TGNetwork

class PromptBasedModel(nn.Module):
    def __init__(self,task_embed=True,ada_kernel=True,is_embed=True,attn_pad=False,vocab_size=2,kernel_number=2048,
                 block=Bottleneck,layers=[3,4,6,3], classifier_num=100):
        super(PromptBasedModel, self).__init__()
        self.ada_kernel=ada_kernel
        if ada_kernel:
            self.gate_network=get_TGNetwork(task_embed=task_embed,is_embed=is_embed,attn_pad=attn_pad,gate_len=sum(layers)*3,vocab_size=vocab_size,kernel_number=kernel_number)
        self.backbone=ada_ResNet(block,layers,classifier_num=classifier_num)
    def forward(self,prompt,img):
        if self.ada_kernel:
            selection=self.gate_network(prompt)
            logits=self.backbone(img,selection)
            return logits,selection
        else:
            logits=self.backbone(img)
            return logits,None