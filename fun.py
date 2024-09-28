import os
import cv2
import random
import matplotlib.pyplot as plt
# Read image

def getFilenames(folder):
    names = []
    for filename in os.listdir(folder):
        names.append(filename)
    return names

folder = "data"

filenames = getFilenames(folder)
index = random.randint(0, len(filenames) - 1)
filename = filenames[index]
img = cv2.imread(os.path.join(folder, filename))

# kek

#plt.imshow(img[:,:,::-1])
#plt.show()



print("lol")
sr = cv2.dnn_superres.DnnSuperResImpl_create()

print("kek")
path = "EDSR_x4.pb"
sr.readModel(path)
print("lol")
sr.setModel("edsr",4)
result = sr.upsample(img)
# Resized image
resized = cv2.resize(img,dsize=None,fx=4,fy=4)
plt.figure(figsize=(12,8))
plt.subplot(1,3,1)
# Original image
plt.imshow(img[:,:,::-1])
plt.subplot(1,3,2)
# SR upscaled
plt.imshow(result[:,:,::-1])
plt.subplot(1,3,3)
# OpenCV upscaled
plt.imshow(resized[:,:,::-1])
plt.show()