import torch
import torch.nn as nn
from dataset_loader import load_mnist
from LeNet import LeNet
from util import test, train, print_size_of_model, time_model_evaluation
import torch.quantization
import warnings
# from torch.autograd import profiler



warnings.filterwarnings('ignore')
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

# test(network, loss_func, test_loader)
# for epoch in range(1, n_epochs + 1):
#   train(epoch,network, train_loader, optimizer, loss_func, log_interval)
#   test(network, loss_func, test_loader)
  
try:
  network.load_state_dict(torch.load('./results/model.pth'))
except:
  #train the network
  test(network, loss_func, test_loader)
  for epoch in range(1, n_epochs + 1):
    train(epoch,network, train_loader, optimizer, loss_func, log_interval)
    test(network, loss_func, test_loader)

network.eval()
print("\n-----------------original model:-----------------------")
# print("original model: \n")
print_size_of_model(network)
onv1_weights = network.features[0].weight.data
time_model_evaluation(network, loss_func, test_loader)

original_state_dict = network.state_dict()

# Dynamic Quantization / print model_size/execution_time/test precision
print("\n----------------dynamic quantized model:----------------")

dquantized_network = LeNet()
dquantized_network.load_state_dict(original_state_dict)
dquantized_network.eval()

dquantized_network = torch.quantization.quantize_dynamic(network, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
torch.save(dquantized_network.state_dict(), './results/dynamic_quantized_model.pth')
# print(dquantized_network)
print_size_of_model(dquantized_network )
time_model_evaluation(dquantized_network, loss_func, test_loader)

# Static Quantization / print model_size/execution_time/test precision
print("\n----------------static quantized model:----------------")

squantized_network = LeNet()
squantized_network.load_state_dict(original_state_dict)
squantized_network.eval()

backend = "x86"
squantized_network.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
torch.quantization.prepare(squantized_network, inplace=True)
torch.quantization.convert(squantized_network, inplace=True)
torch.save(squantized_network.state_dict(), './results/static_quantized_model.pth')

# print(squantized_network)
print_size_of_model(squantized_network )
time_model_evaluation(squantized_network, loss_func, test_loader)

print(" ")