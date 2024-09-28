import torch.nn as nn
import torch
import cv2
import random
from torchvision.transforms import ToTensor
import os
import torch.nn.functional as F
from generators import generateDataThreads

from torchvision import transforms
from train import train

from PIL import Image
from IPython.display import Image 

import keyboard


def generateInputFromOutput(outputImg):
    # Completez aici cu ce vreau sa fac
    # In cazul asta o sa returnez tot outputImg

    inputImg = outputImg.copy()

    return inputImg

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 9, padding = 2, padding_mode = 'replicate')
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 1, padding = 2, padding_mode = 'replicate')
        self.conv3 = nn.Conv2d(32, 3, kernel_size = 5, padding = 2, padding_mode = 'replicate')
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = NeuralNetwork().cuda()
try:
    model.load_state_dict(torch.load("model.pth"))
except:
    pass


for i in range(1, 101):
    training_data = generateDataThreads(100, 100, 128, "data", generateInputFromOutput)
    lr = 1e-4

    model = train(i, model, 1000, training_data, 64, lr, nn.MSELoss()) 
    #torch.save(model.state_dict(), "model.pth")