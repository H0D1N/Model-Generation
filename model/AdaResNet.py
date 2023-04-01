import torch
import torch.nn as nn
from timm.models.layers import create_classifier


class SelectedResNetBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channel,out_channel,stride=1,kernel_size=3,downsample=None):
        super().__init__()
        #Conv1
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        #Conv2
        self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=kernel_size,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)

        self.downsample=downsample

        #act

        self.act=nn.ReLU(inplace=True)


    def forward(self,x,filter_select=None):
        if filter_select !=None:
            #filter_select: [batach_size,conv_num,out_channels]
            residual=x
            out=self.conv1(x)
            out=self.bn1(out)
            out = out * filter_select[:, 0, :][:, :, None, None]
            out=self.act(out)

            out=self.conv2(out)
            out=self.bn2(out)
            out = out * filter_select[:, 1, :][:, :, None, None]

            if self.downsample is not None:
                residual=self.downsample(x)
            out+=residual* filter_select[:, 1, :][:, :, None, None]
            out=self.act(out)
            return out

        else:
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.act(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.act(out)
            return out


class Ada_ResNet(nn.Module):
    def __init__(self,ada_kernel,block,layers,kernel_num,classfier_num):
        self.inplanes=64
        super(Ada_ResNet,self).__init__()
        self.ada_kernel=ada_kernel
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #block1
        self.layer1=self._make_layer(block,kernel_num,layers[0])
        #block2
        self.layer2=self._make_layer(block,kernel_num,layers[1],stride=2)
        #block3
        self.layer3=self._make_layer(block,kernel_num,layers[2],stride=2)
        #blokc4
        self.layer4=self._make_layer(block,kernel_num,layers[3],stride=2)

        self.num_features=kernel_num * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, classfier_num, pool_type='avg')
        #init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,block,channels,blocks,stride=1):
        downsample=None
        if stride != 1 or self.inplanes != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, channels, stride=stride, downsample=downsample))
        # 每个blocks的第一个residual结构保存在layers列表中。
        self.inplanes = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, channels))
            # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return layers

    def forward(self,x,kernel_selection=None):

        if kernel_selection!=None:
            x=self.conv1(x)
            x=self.bn1(x)
            x=self.act1(x)
            x=self.maxpool(x)

            gate_num=0
            for layer in self.layer1:
                x=layer(x,kernel_selection[:,gate_num:gate_num+2,:])
                gate_num+=2

            for layer in self.layer2:
                x=layer(x,kernel_selection[:,gate_num:gate_num+2,:])
                gate_num+=2

            for layer in self.layer3:
                x=layer(x,kernel_selection[:,gate_num:gate_num+2,:])
                gate_num+=2

            for layer in self.layer4:
                x=layer(x,kernel_selection[:,gate_num:gate_num+2,:])
                gate_num+=2

            x = self.global_pool(x)
            x=self.fc(x)
            return x
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            for layer in self.layer1:
                x = layer(x)
            for layer in self.layer2:
                x = layer(x)
            for layer in self.layer3:
                x = layer(x)
            for layer in self.layer4:
                x = layer(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
