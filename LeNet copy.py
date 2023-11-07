import torch
import torch.nn

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()  
        self.conv1 = torch.nn.Conv2d(1, 6, 5, stride=1)
        self.relu  = torch.nn.ReLU(inplace=True) 
        self.pool  = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, stride=1)
        self.conv3 = torch.nn.Conv2d(16, 120, 5, stride=1) 
        self.fc1   = torch.nn.Linear(120, 84)
        self.fc2   = torch.nn.Linear(84, 10)
        
        
    def forward(self, x):  
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