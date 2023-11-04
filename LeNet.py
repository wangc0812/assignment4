import torch
import torch.nn as nn

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1)
        self.relu  = nn.ReLU(inplace=True) 
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, 5, stride=1) 
        self.fc1   = nn.Linear(120, 84)
        self.fc2   = nn.Linear(84, 10)  
        
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
        
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):  
        # x = self.quant(x)
        # x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        # x = self.dequant(x)
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
       
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        x = self.dequant(x)
         
        return x