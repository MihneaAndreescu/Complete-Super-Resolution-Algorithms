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

def generateInputFromOutput(outputImg):

    # Completez aici cu ce vreau sa fac
    # In cazul asta o sa returnez tot outputImg
    small = cv2.resize(outputImg, (0,0), fx = 0.5, fy = 0.5) 
    inputImg = cv2.resize(small, (0,0), fx = 2, fy = 2) 

    assert inputImg.shape == outputImg.shape


    return inputImg

class preUpsampling16(nn.Module):
    modelName = "preUpsampling16.pth"

    def __init__(self):
        super(preUpsampling16, self).__init__()

        self.conv1 = nn.Conv2d(1, 2, kernel_size = 7, padding = 3, padding_mode = 'replicate')
        self.conv2 = nn.Conv2d(2, 4, kernel_size = 7, padding = 3, padding_mode = 'replicate')
        self.conv3 = nn.Conv2d(4, 8, kernel_size = 7, padding = 3, padding_mode = 'replicate')
        self.conv4 = nn.Conv2d(8, 16, kernel_size = 7, padding = 3, padding_mode = 'replicate')
        self.conv5 = nn.Conv2d(16, 8, kernel_size = 7, padding = 3, padding_mode = 'replicate')
        self.conv6 = nn.Conv2d(8, 4, kernel_size = 7, padding = 3, padding_mode = 'replicate')
        self.conv7 = nn.Conv2d(4, 2, kernel_size = 7, padding = 3, padding_mode = 'replicate')
        self.conv8 = nn.Conv2d(2, 1, kernel_size = 7, padding = 3, padding_mode = 'replicate')
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.conv8(x)
        return x



def preUpsampling16train():
    model = preUpsampling16().cuda()
    try:
        model.load_state_dict(torch.load(model.modelName))
    except:
        pass
    for i in range(1, 101):
        training_data = generateDataThreads(100, 100, 2048, "data", generateInputFromOutput)
        lr = 1e-4
        print("64 bro")
        model = train(i, model, 30000, training_data, 64, lr, nn.MSELoss(), model.modelName)

''' info :
           94% din CUDA
           1% din GPU ???????
           invatat 30000 epoci cu acesti parametrii
           scade constant
           folosesc MSELoss
           65 grade la mine in camera cu ventilator pornit pe pozitia 1
           train with SGD
'''


def preUpsampling16evaluate(img):
    model = preUpsampling16().cuda()
    try:
        model.load_state_dict(torch.load(model.modelName))
    except:
        pass
    transToPil = transforms.ToPILImage()
    transToTensor = transforms.ToTensor()
    inp = transToTensor(img)
    inp = inp.unsqueeze_(0)
    inp = inp.cuda()
    img = model(inp)
    img = img.squeeze(0)
    img = transToPil(img)

    return img
