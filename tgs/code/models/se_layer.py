import torch.nn as nn
import torch
import torchvision
import numpy as np

class GlobalAveragePool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (_, c, h, w)
        # Global Average Pooling
        s = x.size()
        h = torch.mean(x.view(s[0],s[1],-1),2)

        # h: (_, c)
        return h

class cSELayer(nn.Module):
    
    def __init__(self, channels):
        super().__init__()

        self.cse_path = nn.Sequential(
            GlobalAveragePool(),
            nn.Linear(channels, channels//2),
            nn.ReLU(inplace=True),
            nn.Linear(channels//2, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.cse_path(x)

        h = h.unsqueeze(2)
        h = h.unsqueeze(3)
        hx = h * x
        return hx

class sSELayer(nn.Module):
    
    def __init__(self, channels):
        super().__init__()

        self.sse_path = nn.Sequential(
            nn.Conv2d(channels,  1, kernel_size=1, padding=0),
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
        h = torch.max(hcse, hsse)
        return h
        
    
