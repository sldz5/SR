
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def ResBlock(x,n_feats,kernel_size):
    bias=True
    bn=False
    act = nn.ReLU(True)
    res_scale = 1
    m = []
    for i in range(2):
        m.append(default_conv(n_feats, n_feats, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(n_feats))
        if i == 0: m.append(act)

    body = nn.Sequential(*m).cuda(0)
    res = body(x).mul(res_scale)
    res += x

    return res

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class Net(nn.Module):
    def __init__(self,input_channels=3):
        super(Net, self).__init__()

        wn = lambda x: nn.utils.weight_norm(x)
        self.conv1 = wn(nn.Conv2d( input_channels, 12, 3 ,padding=1))   #,stride=2, bias=True
        self.conv_1 = wn(nn.Conv2d( input_channels, 12, 1))
        self.conv2 = wn(nn.Conv2d(12, 24, 3, padding=1))   #,stride=2
        self.conv3 = wn(nn.Conv2d(24, 48, 3, padding=1))  #,stride=2
        self.conv4 = wn(nn.Conv2d(48, 96, 3, padding=1)) #,stride=2
        self.conv_4 = wn(nn.Conv2d(48, 96, 1))
        self.skip = wn(nn.Conv2d(48, 3, 5,padding=2))

        self.feat4 = nn.Conv2d(96, 3,3, padding=1)
        self.feat3 = nn.Conv2d(48, 3*4,3, padding=1)
        self.feat2 = nn.Conv2d(24, 3*4,3, padding=1)
        self.feat1 = nn.Conv2d(12, 3*4,3, padding=1)
        self.relu = nn.ReLU()
        #self.relu = nn.LeakyReLU(0.2, True)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.upsample1 = nn.PixelShuffle(2)
        self.upsample2 = nn.PixelShuffle(2)
        self.upsample3 = nn.PixelShuffle(2)
        self.upsample4 = nn.PixelShuffle(2)
        self.pooling = nn.MaxPool2d(2, 2)
        self.dp = nn.Dropout(0.5)
        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
            [0.485, 0.456, 0.406])).view([1,3,1,1])

    def forward(self, x):
        #x = (x - self.rgb_mean.cuda()*255)/127.5
        x1 = self.relu(self.conv1(x))
        #x1 = self.relu(self.conv_1(x1))
        #for _ in range(2):
        #    resBlock1 = ResBlock(x1 , 64, 3)       
        x2 = self.relu(self.conv2(x1))#)self.pooling(
        x3 = self.relu(self.conv3(x2))#)self.pooling(
        x4 = self.relu(self.conv4(x3))#)self.pooling(
        #x4 = self.relu(self.conv_4(x4))
        x4 = self.dp(x4)
        score4 = self.feat4(x4)
        #score4 = self.upsample4(score4)
        #print("score4_1:{}".format(score4.shape))
        #out = F.upsample(score4, [x.shape[2],x.shape[3]], mode='bilinear')
        
        #print("score4_2:{}".format(y.shape))
        x = x + score4
        
        #score3 = self.feat3(x3)
        #score3 = self.upsample3(score3)
        #x = x + score3 
        #score2 = self.feat2(x2)
        #score2 = self.upsample2(score2)
        #x = x + score2
        #score1 = self.feat1(x1)
        #score1 = self.upsample1(score1)
        #x = x + score1
        #x = x * 127.5 + self.rgb_mean.cuda()*255
        return x

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)                #为了将前面多维度的tensor展平成一维


class Discrim(nn.Module):
    def __init__(self, input_nc=1, use_sigmoid=True):   ##use_sigmoid=False
        super(Discrim, self).__init__()
        # ndf=64
        sequence = [
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=2, padding=1),   ## kernel_size=4, stride=2, padding=2
            nn.LeakyReLU(0.2, True)
        ]

        # n=1
        sequence += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),    ####kernel_size=4, stride=2, padding=2
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        # n=2
        sequence += [
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),   ##kernel_size=4, stride=2, padding=2
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  ##kernel_size=4, stride=1, padding=2
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)]  ##kernel_size=4, stride=1, padding=2

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

class SiameseNetwork(nn.Module):
    def __init__(self ,batch_size):
        super(SiameseNetwork, self).__init__()  
        self.output_num = [4,2,1]
        self.batch_size = batch_size    
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),       ###本来5，2，1，改为3，1，1
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            #nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),     ###本来5，2，1，改为3，1，1
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(2, 2),

            #nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(256),
            #nn.MaxPool2d(2,2),

            #nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(512)

            #Flatten(),                    #变平 65536=512*128（最终得到的图片为128*128，512通道）
            #nn.Linear(524288, 128),      #对输入数据做线性变换：\(y = Ax + b\)，Linear(in_features, out_features, bias=True)
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(128)

            #nn.Linear(5120, 256),      
            #nn.ReLU(inplace=True),
            #nn.BatchNorm2d(256)

        )
        self.fc = nn.Sequential(
            nn.Linear(5376, 512),             #
            nn.Sigmoid()
        )    

    def forward_once(self, x):
        x = self.cnn(x)
        spp = spatial_pyramid_pool(x,self.batch_size,[int(x.size(2)),int(x.size(3))],self.output_num)  ##5为batch
        return spp

    def forward(self, input1,input2):  
        output1= self.forward_once(input1)
        output2= self.forward_once(input2)
        #output= self.forward_once(input)
        output = torch.cat((output1,output2),1)
        output = self.fc(output)
        return output     ##

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input,target_tensor.cuda(0))    ###target_tensor.cuda()

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size): #cite#https://github.com/yueruchen/sppnet-pytorch
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = int((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2)
        w_pad = int((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp

### EOF ###
