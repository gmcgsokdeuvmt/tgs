import torch.nn as nn
import torch
import torchvision
import numpy as np

class GlobalAveragePool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (_, c, h, w)
        # Global Average Pooling
        b, c, _, _ = x.size()
        h = self.avg_pool(x).view(b, c)

        # h: (_, c)
        return h

class ELU1(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        h = self.elu(x) + 1
        return h

class cSELayer(nn.Module):
    
    def __init__(self, channels):
        super().__init__()

        self.cse_path = nn.Sequential(
            GlobalAveragePool(),
            nn.Linear(channels, channels//2, bias=False),
            ELU1(),
            nn.Linear(channels//2, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.cse_path(x)

        s = h.size()
        h = h.view(s[0],s[1],1,1)
        hx = h * x
        return hx

class sSELayer(nn.Module):
    
    def __init__(self, channels):
        super().__init__()

        self.sse_path = nn.Sequential(
            nn.Conv2d(channels,  1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()            
        )

    def forward(self, x):
        h = self.sse_path(x)
        hx = h * x
        return hx

class scSELayer(nn.Module):
    
    def __init__(self, channels):
        super().__init__()

        self.cse_layer = cSELayer(channels)
        self.sse_layer = sSELayer(channels)

    def forward(self, x):
        hcse = self.cse_layer(x)
        hsse = self.sse_layer(x)
        h = hcse + hsse
        return h
        
    
