import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

## utils
from models import model_config, model_utils, model_res34unet, se_layer
config = model_config.res34unet_params
_c = model_utils._c
##

class Interpolate(nn.Module):
    def __init__(self, size=None, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
        return x

class SEHyperRes34Unet(nn.Module):

    def __init__(self):
        super().__init__()
        
        resunet = model_res34unet.Res34Unet()

        

        out_e5 = config['out_encoder5']
        out_e4 = config['out_encoder4']
        out_e3 = config['out_encoder3']
        out_e2 = config['out_encoder2']

        out_bot = config['out_bottom']

        out_d5 = config['out_decoder5']
        out_d4 = config['out_decoder4']
        out_d3 = config['out_decoder3']
        out_d2 = config['out_decoder2']
        out_d1 = config['out_decoder1']

        out_logit = config['out_logit']

        self.encoder1 = nn.Sequential(
            resunet.encoder1
        )
        self.encoder2 = nn.Sequential(
            resunet.encoder2,
            se_layer.scSELayer(out_e2)
        )
        self.encoder3 = nn.Sequential(
            resunet.encoder3,
            se_layer.scSELayer(out_e3)
        )
        self.encoder4 = nn.Sequential(
            resunet.encoder4,
            se_layer.scSELayer(out_e4)
        )
        self.encoder5 = nn.Sequential(
            resunet.encoder5,
            se_layer.scSELayer(out_e5)
        )

        self.bottom = resunet.bottom

        self.decoder5 = nn.Sequential(
            resunet.decoder5,
            se_layer.scSELayer(out_d5)
        )
        self.decoder4 = nn.Sequential(
            resunet.decoder4,
            se_layer.scSELayer(out_d4)
        )
        self.decoder3 = nn.Sequential(
            resunet.decoder3,
            se_layer.scSELayer(out_d3)
        )
        self.decoder2 = nn.Sequential(
            resunet.decoder2,
            se_layer.scSELayer(out_d2)
        )
        self.decoder1 = nn.Sequential(
            resunet.decoder1,
            se_layer.scSELayer(out_d1)
        )

        self.hyper_bot = nn.Conv2d(out_bot,  out_d1, kernel_size=1, padding=0)

        self.hyper5 = nn.Sequential(
            nn.Conv2d(out_d5,  out_d1, kernel_size=1, padding=0),
        )

        self.hyper4 = nn.Sequential(
            nn.Conv2d(out_d4,  out_d1, kernel_size=1, padding=0),
        )

        self.hyper3 = nn.Sequential(
            nn.Conv2d(out_d3,  out_d1, kernel_size=1, padding=0),
        )

        self.hyper2 = nn.Sequential(
            nn.Conv2d(out_d2,  out_d1, kernel_size=1, padding=0),
        )

        self.hyper1 = nn.Sequential(
            nn.Conv2d(out_d1,  out_d1, kernel_size=1, padding=0),
        )

        self.hypercol = nn.Conv2d(out_d1*6,  out_d1, kernel_size=3, padding=1)
        self.dropout  = nn.Dropout2d(p=0.4,inplace=True)

        self.logit = nn.Sequential(
            nn.Conv2d(out_d1,  out_logit, kernel_size=1, padding=0),
        )

    def forward(self, x):
        
        e1  = self.encoder1(x)
        e2  = self.encoder2(e1)
        e3  = self.encoder3(e2)
        e4  = self.encoder4(e3)
        e5  = self.encoder5(e4)

        bot = self.bottom(e5)
        
        d  = self.decoder5(_c(bot,e5))
        h  = _c(
                F.upsample(
                    self.hyper_bot(bot), scale_factor=2, mode='bilinear', align_corners=True),
                self.hyper5(d)
            )
        d  = self.decoder4(_c(d,e4))
        h  = _c(
                F.upsample(
                    h, scale_factor=2, mode='bilinear', align_corners=True),
                self.hyper4(d)
            )
        d  = self.decoder3(_c(d,e3))
        h  = _c(
                F.upsample(
                    h, scale_factor=2, mode='bilinear', align_corners=True),
                self.hyper3(d)
            )
        d  = self.decoder2(_c(d,e2))
        h  = _c(
                F.upsample(
                    h, scale_factor=2, mode='bilinear', align_corners=True),
                self.hyper2(d)
            )
        d  = self.decoder1(d)
        h  = _c(
                F.upsample(
                    h, scale_factor=2, mode='bilinear', align_corners=True),
                self.hyper1(d)
            )

        h  = self.hypercol(h)
        h  = self.dropout(h)

        h  = self.logit(h)
        return h