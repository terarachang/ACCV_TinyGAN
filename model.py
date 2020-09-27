import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision import models
import numpy as np
import math


class CBN2d(nn.Module):
    def __init__(self, in_channel, n_condition=128):
        super(CBN2d, self).__init__()
        self.in_channel = in_channel
        self.bn = nn.BatchNorm2d(in_channel, affine=False)
#        self.gamma = nn.Embedding(num_classes, in_channel)
#        self.beta = nn.Embedding(num_classes, in_channel)
        self.embed = nn.Linear(n_condition, in_channel* 2) # generate the affine parameters

        self._initialize()

    def _initialize(self):
#        nn.init.ones_(self.gamma.weight.data)
#        nn.init.zeros_(self.beta.weight.data)
        self.embed.weight.data[:, :self.in_channel] = 1 # init gamma as 1
        self.embed.weight.data[:, self.in_channel:] = 0 # init beta as 0

    def forward(self, h, y):
#        gamma = self.gamma(y).unsqueeze(-1).unsqueeze(-1)
#        beta = self.beta(y).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = self.embed(y).chunk(2, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        out = gamma * self.bn(h) + beta

        return out


class GBlock(nn.Module):
    """Convolution blocks for the generator"""
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True):
        super(GBlock, self).__init__()
        hidden_channel = out_channel 
        
        # depthwise seperable
        self.dw_conv1 = nn.Conv2d(in_channel, in_channel,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=int(in_channel))

        self.dw_conv2 = nn.Conv2d(hidden_channel, hidden_channel, 
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=int(hidden_channel))
        
        self.pw_conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1)
        self.pw_conv2 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)

        self.c_sc = nn.Conv2d(in_channel, out_channel, kernel_size=1)

        self.cbn0 = CBN2d(in_channel)
        self.cbn1 = CBN2d(hidden_channel)
        
        self._initialize()
        
    def _initialize(self):
        nn.init.xavier_uniform_(self.dw_conv1.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.dw_conv2.weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.pw_conv1.weight, gain=1)
        nn.init.xavier_uniform_(self.pw_conv2.weight, gain=1)
        nn.init.xavier_uniform_(self.c_sc.weight, gain=1)


    def _upsample(self, x):
        h, w = x.size()[2:]
        return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')

    def shortcut(self, x):
        h = self._upsample(x)
        h = self.c_sc(h)
        return h

    def forward(self, x, y):
        out = self.cbn0(x, y)
        out = F.relu(out)
        
        out = self._upsample(out)
        out = self.pw_conv1(self.dw_conv1(out))
        out = self.cbn1(out, y)
        out = F.relu(out)
        out = self.pw_conv2(self.dw_conv2(out))
        return out + self.shortcut(x)  # residual


class Generator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, z_dim=128, c_dim=128, repeat_num=5):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.repeat_num = repeat_num
        self.nfilter0 = np.power(2, repeat_num-1)*self.conv_dim
        self.W0 = image_size // np.power(2, repeat_num)
        
        weight = torch.FloatTensor(np.load('cls_weight_reduce.npy'))
        self.embeding = nn.Embedding.from_pretrained(weight, freeze=False)
        
        self.fc = nn.Linear(z_dim, self.nfilter0*self.W0*self.W0)
        # after reshape: (N, self.nfilter0, self.W0, self.W0) = (N, 1024, 4, 4)
        nfilter = self.nfilter0
        blocks = []
        blocks.append(GBlock(nfilter, nfilter, kernel_size=3))
        for i in range(1, repeat_num):
            blocks.append(GBlock(nfilter, nfilter//2))
            nfilter = nfilter // 2
        self.blocks = nn.Sequential(*blocks)
        
        self.bn = nn.BatchNorm2d(nfilter)
        self.colorize = nn.Conv2d(conv_dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, noise, label):
        h = self.fc(noise).view(-1, self.nfilter0, self.W0, self.W0)
        y_emb = self.embeding(label)

        for i in range(self.repeat_num):
            h = self.blocks[i](h, y_emb)
        h = F.relu(self.bn(h))
         
        out = F.tanh(self.colorize(h)) # (batch_size, 3, image_size, image_size)
        
        return out

    def interpolate(self, noise, y_emb):
        h = self.fc(noise).view(-1, self.nfilter0, self.W0, self.W0)
        
        for i in range(self.repeat_num):
            h = self.blocks[i](h, y_emb)
        h = F.relu(self.bn(h))
         
        out = F.tanh(self.colorize(h)) # (batch_size, 3, image_size, image_size)
        
        return out


class Encoder(nn.Module):
    def __init__(self, image_size=128, conv_dim=32, z_dim=128, c_dim=128, repeat_num=5):
        super(Encoder, self).__init__()
        wf = image_size // np.power(2, repeat_num)

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.ReLU(True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(curr_dim*2))
            layers.append(nn.ReLU(True))
            curr_dim = curr_dim * 2

        self.enc = nn.Sequential(*layers)

        self.fc = nn.Linear(wf*wf*curr_dim, z_dim)

    def forward(self, x):
        h = self.enc(x)
        out = self.fc(torch.flatten(h, start_dim=1))
        
        return out


class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        y_dim = 2**(repeat_num-1) * conv_dim # default: 1024
        self.embeding = spectral_norm(nn.Embedding(1000, y_dim)) 

        layers = []
        layers.append(spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.layers = nn.Sequential(*layers)

        self.fc_src = spectral_norm(nn.Linear(y_dim, 1))

    def forward(self, x, label):
        h_list = []
        h = x
        for layer in self.layers:
            h = layer(h)
            h_list.append(h)
        
        h = torch.sum(h, dim=(2,3)) # (bs, 1024) # pooling
        
        out_src = self.fc_src(h)    # (bs, 1)
        out_cls = torch.sum(h * self.embeding(label), dim=1, keepdim=True)

        return out_src + out_cls, h_list    # (bs, 1)

        
class CMPDisLoss(nn.Module):
    def __init__(self):
        super(CMPDisLoss, self).__init__()
        self.criterion = nn.L1Loss() # nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        
    def forward(self, real_list, fake_list):
        loss = 0
        j = 0
        for i in range(1, len(real_list), 2): # compare actvation values of each layer
            loss += self.weights[j] * self.criterion(fake_list[i], real_list[i])
            j += 1

        return loss

class perceptural_loss(nn.Module):
    def __init__(self):
        super(perceptural_loss, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.vgg_relu_3_1 = vgg.features[:12].eval().cuda()

        for param in self.vgg_relu_3_1.parameters():
            param.requires_grad = False

        self.mse = nn.MSELoss()

    def forward(self, real, fake):
        feat_r = self.vgg_relu_3_1(real)
        feat_f = self.vgg_relu_3_1(fake)

        loss = self.mse(feat_f, feat_r)
        
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
