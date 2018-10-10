import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

## utils
from models import model_config, encoder_resnet34, decoder_resnet34, model_utils
config = model_config.res34unet_params
_c = model_utils._c
##

class Res34Unet(nn.Module):

    def __init__(self):
        super().__init__()
        
        encoder = encoder_resnet34.Res34Encoder()
        decoder = decoder_resnet34.Res34Decoder()
        
        self.encoder1 = encoder.encoder1
        self.encoder2 = encoder.encoder2
        self.encoder3 = encoder.encoder3
        self.encoder4 = encoder.encoder4
        self.encoder5 = encoder.encoder5

        out_encoder5 = config['out_encoder5']
        mid_bot = config['mid_bottom']
        out_bot = config['out_bottom']
        self.bottom = nn.Sequential(
            nn.Conv2d(out_encoder5, mid_bot, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_bot, out_bot, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = decoder.decoder5
        self.decoder4 = decoder.decoder4
        self.decoder3 = decoder.decoder3
        self.decoder2 = decoder.decoder2
        self.decoder1 = decoder.decoder1

        out_decoder1 = config['out_decoder1']
        mid_logit = config['mid_logit']
        out_logit = config['out_logit']
        self.logit = nn.Sequential(
            nn.Conv2d(out_decoder1, mid_logit, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_logit,  out_logit, kernel_size=1, padding=0),
        )

    def forward(self, x):
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        h  = self.bottom(e5)
        
        h  = self.decoder5(_c(h,e5))
        h  = self.decoder4(_c(h,e4))
        h  = self.decoder3(_c(h,e3))
        h  = self.decoder2(_c(h,e2))
        h  = self.decoder1(h)

        h = self.logit(h)
        return h