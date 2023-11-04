import torch
import torch.nn as nn

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, stride=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5, stride=1),
            nn.ReLU(inplace=True), 
            nn.AvgPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 120, 5, stride=1),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10) 
        )
        
        self.dequant = torch.ao.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x