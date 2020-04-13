import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.dmp = BackEnd()
        self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)

    def forward(self, input):
        vgg_out = self.vgg(input)

        dmp_out = self.dmp(*vgg_out)
        dmp_out = self.conv_out(dmp_out)

        return torch.abs(dmp_out)

    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        old_name = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '3_4', '4_1', '4_2', '4_3', '4_4', '5_1', '5_2', '5_3', '5_4']
        new_dict = {}
        for i in range(16):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[i]) + '.bias']
        self.vgg.load_state_dict(new_dict, strict=False)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv3_4 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv4_4 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv5_4 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        s1 = input.size()
        conv2, id1 = F.max_pool2d(input, 2, 2, return_indices=True)
        input = self.conv2_1(conv2)
        input = self.conv2_2(input)
        
        s2 = input.size()
        conv3, id2 = F.max_pool2d(input, 2, 2, return_indices=True)
        input = self.conv3_1(conv3)
        input = self.conv3_2(input)
        input = self.conv3_3(input)
        input = self.conv3_4(input)

        s3 = input.size()
        conv4, id3 = F.max_pool2d(input, 2, 2, return_indices=True)
        input = self.conv4_1(conv4)
        input = self.conv4_2(input)
        input = self.conv4_3(input)
        input = self.conv4_4(input)
        
        s4 = input.size()
        conv5, id4 = F.max_pool2d(input, 2, 2, return_indices=True)
        
        input = self.conv5_1(conv5)
        input = self.conv5_2(input)
        input = self.conv5_3(input)
        input = self.conv5_4(input)
        
        return [(conv2, id1, s1), (conv3, id2, s2), (conv4, id3, s3), (conv5, id4, s4)], input

class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.conv1 = BaseConv(896, 256, 1, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv3 = BaseConv(384, 128, 1, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)

        self.conv5 = BaseConv(192, 64, 1, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, *input):
        [(conv2, id1, s1), (conv3, id2, s2), (conv4, id3, s3), (conv5, id4, s4)], feature = input
        
        feature = self.reg_layer(F.upsample_bilinear(feature, scale_factor=2))
        conv5 = F.max_unpool2d(conv5, id4, 2, 2, 0, s4)
        input = torch.cat([feature, conv5, conv4], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        
        input = F.max_unpool2d(input, id3, 2, 2, 0, s3)
        input = torch.cat([input, conv3], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        
        input = F.max_unpool2d(input, id2, 2, 2, 0, s2)
        input = torch.cat([input, conv2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)

        return input

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)
        return input
