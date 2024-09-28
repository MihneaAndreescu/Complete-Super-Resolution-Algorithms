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
    #small = cv2.resize(outputImg, (0,0), fx = 0.5, fy = 0.5)
    #inputImg = cv2.resize(small, (0,0), fx = 2, fy = 2)

    inputImg = cv2.resize(outputImg, (0,0), fx = 0.5, fy = 0.5)

    #assert inputImg.shape == outputImg.shape


    return inputImg

class awesomepreUpsampling(nn.Module):
    modelName = "awesomepreUpsampling.pth"

    def __init__(self):
        super(awesomepreUpsampling, self).__init__()


        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.conv2 = nn.Conv2d(8, 7, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.conv3 = nn.Conv2d(7, 6, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.conv4 = nn.Conv2d(6, 5, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.conv5 = nn.Conv2d(5, 4, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.conv6 = nn.Conv2d(4, 3, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.conv7 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1, padding_mode = 'replicate')
        self.conv8 = nn.ConvTranspose2d(3, 2, kernel_size = 4, stride = 2)
        self.conv9 = nn.Conv2d(2, 1, kernel_size = 3, padding = 0, padding_mode = 'replicate')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.conv9(x)
        return x



def awesomepreUpsamplingtrain():
    model = awesomepreUpsampling().cuda()
    try:
        model.load_state_dict(torch.load(model.modelName))
    except:
        print("model nou yaaaaay !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pass
    for i in range(1, 101):
        training_data = generateDataThreads(200, 200, 4096, "data", generateInputFromOutput)
        lr = 1e-4
        model = train(i, model, 3000000, training_data, 128, lr, nn.MSELoss(), model.modelName)


def awesomepreUpsamplingevaluate(img):
    model = awesomepreUpsampling().cuda()
    try:
        model.load_state_dict(torch.load(model.modelName))
    except:
        print("model nou yaaaaay")
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

awesomepreUpsamplingtrain()

exit(0)
def getFilenames(folder):
    names = []
    for filename in os.listdir(folder):
        names.append(filename)
    return names

def transformToGRAYSCALE(folder):
    filenames = getFilenames(folder)
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(folder, filename), img)
        print("A :", img.shape)
def deleteSmall(folder):
    filenames = getFilenames(folder)
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if min(img.shape[0], img.shape[1]) < 2000:
            os.remove(os.path.join(folder, filename))
        else:
            print("B :", img.shape)
def rezdim0(folder):
    filenames = getFilenames(folder)
    itr = 0
    sz = len(filenames)
    for filename in filenames:
        itr += 1
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img.shape[0] >= 2000:

            pos = img.shape[0] // 2


            img1 = img[:pos, :]
            img2 = img[pos:, :]

            '''print(img.shape)
            print(img1.shape)
            print(img2.shape)'''

            befdot = filename.split(".")
            assert len(befdot) == 2
            filename1 = befdot[0] + "K1K." + befdot[1]
            filename2 = befdot[0] + "K2K." + befdot[1]

            '''
            print(filename1)
            exit(0)'''

            os.remove(os.path.join(folder, filename))
            cv2.imwrite(os.path.join(folder, filename1), img1)
            cv2.imwrite(os.path.join(folder, filename2), img2)
            sz += 1


        print(itr, "out of", len(filenames), ":", img.shape, "|", len(getFilenames(folder)), " and sz =", sz)
    if len(filenames) == sz:
        print("NOOO CHANGEEE!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    filenames = getFilenames(folder)
    print(sz, len(filenames))
    assert len(filenames) == sz
    rezdim0(folder)


def rezdim1(folder):
    filenames = getFilenames(folder)
    itr = 0
    sz = len(filenames)
    for filename in filenames:
        itr += 1
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img.shape[1] >= 2000:

            pos = img.shape[1] // 2
            img1 = img[:, :pos]
            img2 = img[:, pos:]

            befdot = filename.split(".")
            assert len(befdot) == 2
            filename1 = befdot[0] + "K1K." + befdot[1]
            filename2 = befdot[0] + "K2K." + befdot[1]

            '''
            print(filename1)
            exit(0)'''

            os.remove(os.path.join(folder, filename))
            cv2.imwrite(os.path.join(folder, filename1), img1)
            cv2.imwrite(os.path.join(folder, filename2), img2)
            sz += 1


        print(itr, "out of", len(filenames), ":", img.shape, "|", len(getFilenames(folder)), " and sz =", sz)
    if len(filenames) == sz:
        print("NOOO CHANGEEE!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    filenames = getFilenames(folder)
    print(sz, len(filenames))
    assert len(filenames) == sz
    rezdim1(folder)

def resolveData(folder):
    transformToGRAYSCALE(folder)
    deleteSmall(folder)
    rezdim0(folder)
    rezdim1(folder)
#resolveData("bigdata")
