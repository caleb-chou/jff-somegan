import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms

data_transfroms = transforms.Compose([transforms.ToTensor()])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transfroms)

batch_size = 10

dataloader_mnist_train = t_data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

def noisify():
    return torch.rand(batch_size,100)

class generator(nn.Module):
    def __init__(self, inp, out):
        super(generator,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300,1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000,800),
            nn.ReLU(inplace=True),
            nn.Linear(800,out)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x
    
class discriminator(nn.Module):

    def __init__(self,inp,out):
        super(discriminator,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inp,300),
            nn.ReLU(inplace=True),
            nn.Linear(300,300),
            nn.ReLU(inplace=True),
            nn.Linear(300,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,out),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.net(x)
        return x

def plot_image(array, number=None):
    array = array.detatch()
    array = array.reshape(28,28)
    plt.imshow(array, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    if number:
        plt.xlabel(number,fontsize='x-large')
    plt.show()

