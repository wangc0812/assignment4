import torch
import torchvision


def load_mnist(batch_size_train,batch_size_test):
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./mnist_dataset', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((32,32)),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./mnist_dataset', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((32, 32)),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_test, shuffle=True)
    return train_loader,test_loader