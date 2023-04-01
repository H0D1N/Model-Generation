import torch
import torch.nn as nn
import colossalai.nn as col_nn

class sparse_gate_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bass_criterion=col_nn.CrossEntropyLoss()
    def forward(self,logists,label,kernel_selection):

        #交叉熵部分
        loss = self.bass_criterion(logists, label)
        if kernel_selection is not None:
            c=kernel_selection.size()[-1]*kernel_selection.size()[-2]*kernel_selection.size()[-3]
            #限制kernel使用率
            loss=loss+0.1*torch.norm(kernel_selection,p=1)/c
        return loss