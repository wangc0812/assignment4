## The report for assignment 4

***Cong Wang***



### 1. The implementation and training of the LeNet

#### 1.1 Structure of modified LeNet

Table 1 shows the structure of the modified LeNet employed in this assignment, detailing the layer configurations, dimensions, strides, and activation functions used within our implementation.

##### Table 1. The LeNet CNN Structure


| layer   | size       | stride | Activation |
| ------- | ---------- | ------ | ---------- |
| Conv    | 5x5x1x6    | 1      | ReLU       |
| Avgpool | 2x2        | 2      |            |
| Conv    | 5x5x6x16   | 1      | ReLU       |
| Avgpool | 2x2        | 2      |            |
| Conv    | 5x5x16x120 | 1      | ReLU       |
| FC      | 120x84     |        | ReLU       |
| FC      | 84x10      |        |            |

#### 1.2 Implementation of LeNet in Python using Pytorch

The LeNet network in Table1 was first implemented, and the corresponding Python code is as follows:

````python
import torch
import torch.nn as nn

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
        x = self.quant(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x
````

It's worth mentioning that within our LeNet implementation, we included the lines `self.quant = torch.quantization.QuantStub()` and `self.dequant = torch.quantization.DeQuantStub()`, which are necessary for the static quantization process we will undertake later. 

Then, this LeNet network was trained using these hyper-parameter:

````python
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
````

After 3 expochs’ training, we get the well-trained model with $average\  loss=0.0813, Accuracy: 9768/10000 \approx 98\%$.

### 2. Quantization

#### 2.1 Dynamic Quantization

Dynamic quantization in PyTorch is a technique that converts weights to int8 precision, aiming to reduce model size and increase inference speed with minimal impact on accuracy. 

Unlike static quantization, which quantizes both weights and activations ahead of time, dynamic quantization only quantizes weights before inference and computes activations in int8 on-the-fly. In dynamic quantization, the activations are read and written to memory in floating point format. This approach is particularly useful for models with recurrent layers or when the computational overhead of static quantization is too high. 

We apply the dynamic quantization to the model we have trained in Section1:

```python
dquantized_network.eval()
dquantized_network = torch.quantization.quantize_dynamic(network, {nn.Conv2d, nn.Linear}, dtype=torch.qint8)
torch.save(dquantized_network.state_dict(), './results/dynamic_quantized_model.pth')
```

After we perform the dynamic quantization, we get the dynamic quantized model as below:

```shell
LeNet(
  (quant): QuantStub()
  (dequant): DeQuantStub()
  (features): Sequential(
    (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU(inplace=True)
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): ReLU(inplace=True)
    (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (6): Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
    (7): ReLU(inplace=True)
  )
  (classifier): Sequential(
    (0): DynamicQuantizedLinear(in_features=120, out_features=84, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
    (1): ReLU(inplace=True)
    (2): DynamicQuantizedLinear(in_features=84, out_features=10, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
  )
)
```

I noticed that even though I specified the quantization of both fully connected and convolutional layers during the process, however, only the fully connected layers were actually quantized into`DynamicQuantizedLinear` . The reason for that is the dynamic quantization in current version of Pytorch does not support Conv layer [1].

#### 2.2 Static Quantization

Static quantization in PyTorch is a more involved process than dynamic quantization, as it includes additional steps to quantize not just the weights but also the activations. The process involves:

1. **Model Preparation**: The model is modified to include quantization modules like `QuantStub` and `DeQuantStub`.

2. **Model Calibration**: The model is run with representative data (calibration dataset) to capture the distribution of activations, which is then used to determine the scale and zero-point for quantization.

3. **Model Conversion**: The prepared and calibrated model is converted to a statically quantized model where the weights and activations are quantized.

Static quantization typically results in better performance than dynamic quantization because it quantizes the entire model, including both weights and activations, which can be executed using integer-only arithmetic. This can lead to significant improvements in runtime and memory usage on supported hardware.

As we have mentioned before, we need to add`self.quant = torch.quantization.QuantStub()` and`self.dequant = torch.quantization.DeQuantStub()`  to support the quant and dequant operation in the forward process. The following code is to deploy static quantization on the well-trained model:

````python
squantized_network.eval()
backend = "x86"
squantized_network.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
torch.quantization.prepare(squantized_network, inplace=True)
torch.quantization.convert(squantized_network, inplace=True)
torch.save(squantized_network.state_dict(), './results/static_quantized_model.pth')
````

The quantized model we get is:

````shell
LeNet(
  (quant): Quantize(scale=tensor([1.]), zero_point=tensor([0]), dtype=torch.quint8)
  (dequant): DeQuantize()
  (features): Sequential(
    (0): QuantizedConv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), scale=1.0, zero_point=0)
    (1): ReLU(inplace=True)
    (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (3): QuantizedConv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), scale=1.0, zero_point=0)
    (4): ReLU(inplace=True)
    (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (6): QuantizedConv2d(16, 120, kernel_size=(5, 5), stride=(1, 1), scale=1.0, zero_point=0)
    (7): ReLU(inplace=True)
  )
  (classifier): Sequential(
    (0): QuantizedLinear(in_features=120, out_features=84, scale=1.0, zero_point=0, qscheme=torch.per_channel_affine)
    (1): ReLU(inplace=True)
    (2): QuantizedLinear(in_features=84, out_features=10, scale=1.0, zero_point=0, qscheme=torch.per_channel_affine)
  )
)
````

It shows both fully connected layer and conv layer are quantized.



### 3. Performance and Analysis

#### 3.1 Experiment Setup

All experiments were executed on a hardware setup consisting of a 13th Generation Intel Core™ i7-13700F processor with 24 threads, complemented by 16.0 GiB of system memory. The software environment was standardized across all tests, utilizing Python 3.8.18 alongside PyTorch 2.1.0. The OS for these experiments was Ubuntu 22.04.3 LTS.

#### 3.2 Experiment Results and Discuss

Figure 1 compares the performance of two quantization methods and the original model( i.e. floating point) in inference, including model size, accuracy, and execution time. 

##### Figure 1. The performance results

```
-----------------original model:-----------------------
Size (MB): 0.250382

Test set: Avg. loss: 0.0992, Accuracy: 9719/10000 (97%)

Evaluate total time (seconds): 0.735183

----------------dynamic quantized model:----------------
Size (MB): 0.219062

Test set: Avg. loss: 0.0997, Accuracy: 9719/10000 (97%)

Evaluate total time (seconds): 0.749665

----------------static quantized model:----------------
Size (MB): 0.075466

Test set: Avg. loss: 0.1505, Accuracy: 9570/10000 (96%)

Evaluate total time (seconds): 0.741795
```

It is clear that the model size after quantion can reduce.This is  because, compared to the 32-bit floating-point numbers used in PyTorch, int8 occupies less storage space. It is noticed that the static quantized model has less size than dynamic quantized model, 0.21MB and 0.075MB, respectively. This outcome arises from the intrinsic limitation of dynamic quantization, which is constrained to quantize solely the fully connected layers, as opposed to static quantization that encompasses both convolutional and fully connected layers within its scope.

However, the accuracy of the static quantized model diminished from 97% to 96%, whereas the dynamica quantized model maintained the original accuracy in this experiment. This discrepancy can be attributed to the fact that static quantization additionally quantizes the activations, thereby introducing more errors compared to dynamic quantization.

When it comes to the execution time, both quantized model do not perform as well as the original model. One reason is both quantization methods will introduce more computation overhead. Dynamic quantization converts the activations to int8 on the fly before doing the computation[2], while the static quantization need to perform `quant` and`dequant`  in forward process. As to this model, the acceleration benefits gained by quantization may be outweighed by the additional computational overhead, which results in the expected acceleration effect not being attained.

### Reference：

[1] https://pytorch.org/docs/stable/quantization.html#model-preparation-for-eager-mode-static-quantization

[2] https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#post-training-static-quantization