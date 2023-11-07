import torch
import torch.nn as nn
import time

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()  
        
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
        
        
    def forward(self, x):  
        # eval_start_time = time.time()
        x = self.quant(x) 
        # eval_end_time = time.time()
        # eval_duration_time1 = eval_end_time - eval_start_time
        # print(result)
        
        # eval_start_time = time.time()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # eval_end_time = time.time()
        # eval_duration_time = eval_end_time - eval_start_time
        # print("Evaluate conv computing time (seconds): {0:.6f}".format(eval_duration_time))
        
        # eval_start_time = time.time() 
        x = self.dequant(x)
        # eval_end_time = time.time()
        # eval_duration_time2 = eval_end_time - eval_start_time
        # print("Evaluate quant and dequant computing time (seconds): {0:.6f}".format(eval_duration_time1+eval_duration_time2)) 
        return x