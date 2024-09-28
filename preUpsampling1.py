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

    inputImg = outputImg.copy()

    return inputImg

class preUpsampling1(nn.Module):
    modelName = "preUpsampling1.pth"
    
    def __init__(self):
        super(preUpsampling1, self).__init__()
  
        self.conv1 = nn.Conv2d(1, 1, kernel_size = 1, padding = 0, padding_mode = 'replicate')
    def forward(self, x):
        x = self.conv1(x)
        return x



def preUpsampling1train():
    model = preUpsampling1().cuda()
    try:
        model.load_state_dict(torch.load(model.modelName))
    except:
        pass
    for i in range(1, 101):
        training_data = generateDataThreads(100, 100, 2048, "data", generateInputFromOutput)
        lr = 1e-4

        model = train(i, model, 30000, training_data, 190, lr, nn.MSELoss(), model.modelName) 
  
''' info : 
           81% din CUDA
           65% din GPU
           invatat 30000 epoci cu acesti parametrii
           scade constant
           folosesc MSELoss
           67 grade la mine in camera cu ventilator pornit pe pozitia 1
           train with SGD
'''


def preUpsampling1evaluate(img):
    model = preUpsampling1().cuda()
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
