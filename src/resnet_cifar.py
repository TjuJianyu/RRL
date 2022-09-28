# copy from [TODO]
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from lib.normalize import Normalize
import numpy as np 
from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, low_dim=128,width=1, avg_pool2d=True, shortcut=(None, None, None)):
        super(ResNet, self).__init__()
        #print(avg_pool2d)
        self.width = width
        self.avg_pool2d = avg_pool2d
        self.in_planes = 64*self.width
        self.conv1 = nn.Conv2d(3, 64*self.width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64*self.width)
        self.layer1 = self._make_layer(block, 64*self.width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128*self.width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256*self.width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512*self.width, num_blocks[3], stride=2)
        
        pool_expension = 1 if avg_pool2d else 16 
        

        self.shortcut = shortcut
        #self.shortcut = (2,1,512)
        #self.fc = nn.Linear(512*block.expansion*self.width * pool_expension, low_dim)
        # self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.shortcut[0] is None:
            
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            #print(out.shape)
            if self.avg_pool2d:
                out = F.avg_pool2d(out, 4)

            out = out.view(out.size(0), -1)
            #out = self.fc(out)
            # out = self.l2norm(out)
            #print(out.shape)
            return out

        else:
            
            if   self.shortcut[0] == 1:
                for i in range(self.shortcut[1]):
                    out = self.layer1[i](out)

            elif self.shortcut[0] == 2:
                out = self.layer1(out)
                for i in range(self.shortcut[1]):
                    out = self.layer2[i](out)
                #print(out.shape)
            elif self.shortcut[0] == 3:
                out = self.layer1(out)
                out = self.layer2(out)
                for i in range(self.shortcut[1]):
                    out = self.layer3[i](out)
            
            elif self.shortcut[0] == 4:
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                for i in range(self.shortcut[1]):
                    out = self.layer4[i](out)
            

            n_feats_per_channel = self.shortcut[2] // out.shape[1]

            dim = np.sqrt(n_feats_per_channel)
            if dim % 1 != 0:
                dim = int(dim)
                shape = (dim, dim+2) if np.abs(dim * (dim+1) * out.shape[1] - self.shortcut[2]) > np.abs(dim * (dim+2) * out.shape[1] - self.shortcut[2]) else (dim, dim+1)
            else:
                shape = (int(dim), int(dim))

            #print(shape)
            #print(out.shape, n_feats_per_channel, shape)
            #out = F.adaptive_avg_pool2d(out, int(np.sqrt(n_feats_per_channel)))
            #print(out.shape, shape)
            out = F.adaptive_avg_pool2d(out, shape)
            
            out = out.view(out.size(0), -1)

            return out 



def ResNet18(low_dim=128, width=1, avg_pool2d=True, shortcut=None):
    return ResNet(BasicBlock, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut)


# def ResNet34(low_dim=128, width=1, avg_pool2d=True):
#     return ResNet(BasicBlock, [3,4,6,3], low_dim, width, avg_pool2d=avg_pool2d)

# def ResNet50(low_dim=128, width=1, avg_pool2d=True):
#     return ResNet(Bottleneck, [3,4,6,3], low_dim, width, avg_pool2d=avg_pool2d)

# def ResNet101(low_dim=128, width=1, avg_pool2d=True):
#     return ResNet(Bottleneck, [3,4,23,3], low_dim, width, avg_pool2d=avg_pool2d)

# def ResNet152(low_dim=128, width=1, avg_pool2d=True):
#     return ResNet(Bottleneck, [3,8,36,3], low_dim, width, avg_pool2d=avg_pool2d)



def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
