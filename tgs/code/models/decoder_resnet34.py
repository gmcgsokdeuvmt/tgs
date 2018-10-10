import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models import model_config
config = model_config.res34unet_params

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ):
        super(Decoder, self).__init__()
        self.conv1 =  nn.Conv2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x ):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        return x

class Res34Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()

        out_e5 = config['out_encoder5']
        out_e4 = config['out_encoder4']
        out_e3 = config['out_encoder3']
        out_e2 = config['out_encoder2']
        #out_e1 = config['out_encoder1']

        out_bot = config['out_bottom']

        mid_d5 = config['mid_decoder5']
        mid_d4 = config['mid_decoder4']
        mid_d3 = config['mid_decoder3']
        mid_d2 = config['mid_decoder2']
        mid_d1 = config['mid_decoder1']

        out_d5 = config['out_decoder5']
        out_d4 = config['out_decoder4']
        out_d3 = config['out_decoder3']
        out_d2 = config['out_decoder2']
        out_d1 = config['out_decoder1']

        self.decoder5 = Decoder(out_e5+out_bot,mid_d5, out_d5)
        self.decoder4 = Decoder(out_e4+out_d5, mid_d4, out_d4)
        self.decoder3 = Decoder(out_e3+out_d4, mid_d3, out_d3)
        self.decoder2 = Decoder(out_e2+out_d3, mid_d2, out_d2)
        self.decoder1 = Decoder(out_d2       , mid_d1, out_d1)
