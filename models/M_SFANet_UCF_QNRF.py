import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from .contextual_layer import ContextualModule

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        return self.relu(x)

class ASPP(nn.Module):
    def __init__(self, inplanes, BatchNorm):
        super(ASPP, self).__init__()
        dilations = [1, 12, 24, 36]

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.relu(x)

        return self.dropout(x)
                
class ScalePyramidModule(nn.Module):
    def __init__(self):
        super(ScalePyramidModule, self).__init__()
        self.assp = ASPP(512, BatchNorm=None)
        self.can = ContextualModule(512, 512)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, *input):
        conv2_2, conv3_3, conv4_4, conv5_4 = input 
        conv4_4 = self.can(conv4_4)
        ### Why don't you apply ASSP in higher resolution ??? ###
        conv5_4 = torch.cat([F.upsample_bilinear(self.assp(conv5_4), scale_factor=2), 
                    self.reg_layer(F.upsample_bilinear(conv5_4, scale_factor=2))], 1)
        
        return conv2_2, conv3_3, conv4_4, conv5_4

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.spm = ScalePyramidModule()
        self.dmp = BackEnd()
        
        self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)

    def forward(self, input):
        input = self.vgg(input)
        
        spm_out = self.spm(*input)
        dmp_out = self.dmp(*spm_out)
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
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        input = self.conv3_3(input)
        conv3_4 = self.conv3_4(input)

        input = self.pool(conv3_4)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        input = self.conv4_3(input)
        conv4_4 = self.conv4_4(input)

        input = self.pool(conv4_4)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        input = self.conv5_3(input)
        conv5_4 = self.conv5_4(input)
        return conv2_2, conv3_4, conv4_4, conv5_4


class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        
        self.conv1 = BaseConv(896, 256, 1, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        
        self.conv3 = BaseConv(896, 128, 1, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(inplace=True), use_bn=False)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_4, conv5_4 = input
        
        input = torch.cat([conv5_4, conv4_4], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        input = F.upsample_bilinear(input, scale_factor=2)
        
        input = torch.cat([input, conv3_3, F.upsample_bilinear(conv5_4, scale_factor=2)], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = F.upsample_bilinear(input, scale_factor=2)

        input = torch.cat([input, conv2_2], 1)
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
