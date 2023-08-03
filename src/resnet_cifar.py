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

class BasicBlockLinear(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockLinear, self).__init__()
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

class IIResNet(nn.Module):
    def __init__(self, block, num_blocks, low_dim=128,width=1, avg_pool2d=True, shortcut=(None, None, None)):
        super(IIResNet, self).__init__()
        #print(avg_pool2d)
        self.width = width
        self.avg_pool2d = avg_pool2d
        self.in_planes = 64*self.width
        self.conv1 = nn.Conv2d(3, 64*self.width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64*self.width)
        self.layer1 = self._make_layer(block, 64*self.width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64*self.width, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64*self.width, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 64*self.width, num_blocks[3], stride=1)
        
        pool_expension = 1 if avg_pool2d else 16 
        
        self.shortcut = shortcut

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # 1
        #print(out.shape)
        out = self.layer1(out)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)

        if self.avg_pool2d:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
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
        out = F.relu(self.bn1(self.conv1(x))) # 1
        #print(self.shortcut) 
        if self.shortcut[0] is None:
            
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            #print(out.shape)
            if self.avg_pool2d:
                out = F.avg_pool2d(out, 4)
            #print(out.shape)
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
            
            # dim = np.sqrt(n_feats_per_channel)
            # if dim % 1 != 0:
            #     dim = int(dim)
            #     shape = (dim, dim+2) if np.abs(dim * (dim+1) * out.shape[1] - self.shortcut[2]) > np.abs(dim * (dim+2) * out.shape[1] - self.shortcut[2]) else (dim, dim+1)
            # else:
            #     shape = (int(dim), int(dim))
            
            w,h = int(np.sqrt(n_feats_per_channel)), int(np.sqrt(n_feats_per_channel))
            if w * h == n_feats_per_channel:
                w, h = w,h 
            elif n_feats_per_channel == 2:
                w,h = 1,2
            elif n_feats_per_channel == 8:
                w,h = 2,4
            elif n_feats_per_channel == 32:
                w,h = 4,8
            elif n_feats_per_channel == 128:
                w,h = 8, 16
            else:
                raise NotImplementedError
            shape = (w,h)
            #print(out.shape,n_feats_per_channel,shape)
            
            #print(shape)
            #print(out.shape, n_feats_per_channel, shape)
            #out = F.adaptive_avg_pool2d(out, int(np.sqrt(n_feats_per_channel)))
            #print(out.shape, shape)
            out = F.adaptive_avg_pool2d(out, shape)
            
            out = out.view(out.size(0), -1)

            return out 

class IIResNetfix1st(IIResNet):
    def __init__(self, block, num_blocks, low_dim=128,width=1, avg_pool2d=True, shortcut=(None, None, None), bn_training=True):
        super(IIResNetfix1st, self).__init__(block, num_blocks, low_dim,width, avg_pool2d, shortcut)
        
        self.bn_training = bn_training
    def train(self, mode):

        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        
        self.conv1.train(False)
        self.bn1.train(self.bn_training)
        
        return self
class ResNetfix1st(ResNet):
    def __init__(self, block, num_blocks, low_dim=128,width=1, avg_pool2d=True, shortcut=(None, None, None), bn_training=True):
        super(ResNetfix1st, self).__init__(block, num_blocks, low_dim,width, avg_pool2d, shortcut)
        
        self.bn_training = bn_training
    def train(self, mode):

        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        
        self.conv1.train(False)
        self.bn1.train(self.bn_training)
        
        return self



def ResNet18(low_dim=128, width=1, avg_pool2d=True, shortcut=(None,None,None)):
    return ResNet(BasicBlock, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut)

def ResNet18linear(low_dim=128, width=1, avg_pool2d=True, shortcut=(None,None,None)):
    return ResNet(BasicBlockLinear, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut)

def ResNet18fix1stbneval(low_dim=128, width=1, avg_pool2d=True, shortcut=(None,None,None),linear=False):
    if linear:
        network = ResNetfix1st(BasicBlockLinear, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut, bn_training=False)
    else:
        network = ResNetfix1st(BasicBlock, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut, bn_training=False)
    
    #print(network.bn1.training)
    for name, param in network.conv1.named_parameters():
        param.requires_grad = False 
    for name, param in network.bn1.named_parameters():
        param.requires_grad = False 

    return network

def IIResNet18fix1stbneval(low_dim=128, width=1, avg_pool2d=True, shortcut=(None,None,None),linear=False):
    if linear:
        network = IIResNetfix1st(BasicBlockLinear, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut, bn_training=False)
    else:
        network = IIResNetfix1st(BasicBlock, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut, bn_training=False)
    
    #print(network.bn1.training)
    for name, param in network.conv1.named_parameters():
        param.requires_grad = False 
    for name, param in network.bn1.named_parameters():
        param.requires_grad = False 

    return network


def ResNet18fix1stbn(low_dim=128, width=1, avg_pool2d=True, shortcut=(None,None,None),linear=False):
    if linear:
        network = ResNetfix1st(BasicBlockLinear, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut, bn_training=True)
    else:
        network = ResNetfix1st(BasicBlock, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut, bn_training=True)
    
    #print(network.bn1.training)
    for name, param in network.conv1.named_parameters():
        param.requires_grad = False 
    for name, param in network.bn1.named_parameters():
        param.requires_grad = False 

    return network


def ResNet18fix1st(low_dim=128, width=1, avg_pool2d=True, shortcut=(None,None,None),linear=False):
    if linear:
        network = ResNetfix1st(BasicBlockLinear, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut)
    else:
        network = ResNetfix1st(BasicBlock, [2,2,2,2], low_dim, width, avg_pool2d=avg_pool2d, shortcut=shortcut)
    
    #print(network.bn1.training)
    for name, param in network.conv1.named_parameters():
        param.requires_grad = False 
    return network


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
