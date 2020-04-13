import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torch.utils import model_zoo

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.amp = BackEnd()
        self.dmp = BackEnd()

        self.conv_att = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)

    def forward(self, input):
        vgg_out = self.vgg(input)

        amp_out = self.amp(*vgg_out)
        dmp_out = self.dmp(*vgg_out)

        amp_out = self.conv_att(amp_out)
        dmp_out = amp_out * dmp_out
        dmp_out = self.conv_out(dmp_out)

        return dmp_out, amp_out

    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(10):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        s1 = input.size()
        conv2_2, id1 = F.max_pool2d(input, 2, 2, return_indices=True)
        input = self.conv2_1(conv2_2)
        input = self.conv2_2(input)
        
        s2 = input.size()
        conv3_3, id2 = F.max_pool2d(input, 2, 2, return_indices=True)
        input = self.conv3_1(conv3_3)
        input = self.conv3_2(input)
        input = self.conv3_3(input)

        s3 = input.size()
        conv4_3, id3 = F.max_pool2d(input, 2, 2, return_indices=True)
        input = self.conv4_1(conv4_3)
        input = self.conv4_2(input)
        input = self.conv4_3(input)
        
        s4 = input.size()
        conv5_3, id4 = F.max_pool2d(input, 2, 2, return_indices=True) 
        
        return [(conv2_2, id1, s1), (conv3_3, id2, s2), (conv4_3, id3, s3), (conv5_3, id4, s4)]

class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.conv1 = BaseConv(768, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        
        self.conv3 = BaseConv(384, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(192, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, *input):
        (conv2_2, id1, s1), (conv3_3, id2, s2), (conv4_3, id3, s3), (conv5_3, id4, s4) = input

        input = F.max_unpool2d(conv5_3, id4, 2, 2, 0, s4)
        input = torch.cat([input, conv4_3], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        
        input = F.max_unpool2d(input, id3, 2, 2, 0, s3)
        input = torch.cat([input, conv3_3], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        
        input = F.max_unpool2d(input, id2, 2, 2, 0, s2)
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
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = SynchronizedBatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input
