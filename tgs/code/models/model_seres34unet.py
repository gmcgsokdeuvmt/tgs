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

class SERes34Unet(nn.Module):

    def __init__(self):
        super().__init__()
        
        resunet = model_res34unet.Res34Unet()

        

        out_e5 = config['out_encoder5']
        out_e4 = config['out_encoder4']
        out_e3 = config['out_encoder3']
        out_e2 = config['out_encoder2']
        out_e1 = config['out_encoder1']

        out_d5 = config['out_decoder5']
        out_d4 = config['out_decoder4']
        out_d3 = config['out_decoder3']
        out_d2 = config['out_decoder2']
        out_d1 = config['out_decoder1']

        self.encoder1 = nn.Sequential(
            resunet.encoder1,
            se_layer.scSELayer(out_e1)
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

        self.logit = resunet.logit

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