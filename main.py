import torch
import torch.nn as nn
from dataset_loader import load_mnist
from LeNet import LeNet
from util import test, train, print_size_of_model, time_model_evaluation
import copy

#define the hyper-paramters
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Load MNIST dataset
train_loader, test_loader = load_mnist(batch_size_train,batch_size_test)

# Load the Network
network = LeNet()

# Define the optimizer and loss
optimizer = torch.optim.SGD(network.parameters(), lr=0.1, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

try:
  network.load_state_dict(torch.load('./results/model.pth'))
except:
  #train the network
  test(network, loss_func, test_loader)
  for epoch in range(1, n_epochs + 1):
    train(epoch,network, train_loader, optimizer, loss_func, log_interval)
    test(network, loss_func, test_loader)

print("original model:")
print_size_of_model(network)
onv1_weights = network.features[0].weight.data
time_model_evaluation(network, loss_func, test_loader)

# Dynamic Quantization / print model_size/execution_time/test precision
print("\ndynamic quantized model:")
import torch.quantization
dquantized_network = torch.quantization.quantize_dynamic(network, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
# print(dquantized_network)
# qconv1_weights = quantized_network.features[0].weight.data
print_size_of_model(dquantized_network )
time_model_evaluation(dquantized_network, loss_func, test_loader)

# Static Quantization / print model_size/execution_time/test precision
print("\nstatic quantized model:")
