import torch.nn as nn
import torch
import torchvision
class Res34Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.resnet = torchvision.models.resnet34(pretrained=True)

        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.encoder2 = nn.Sequential(  
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.resnet.layer1
        ) #64
        self.encoder3 = self.resnet.layer2  #128
        self.encoder4 = self.resnet.layer3  #256
        self.encoder5 = self.resnet.layer4  #512
