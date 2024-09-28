from torchvision import transforms
import random
import cv2
import os
import threading

def getFilenames(folder):
    names = []
    for filename in os.listdir(folder):
        names.append(filename)
    return names


def generateData(H, W, Cnt, folder, generateInputFromOutput):
    func = transforms.ToTensor()
    filenames = getFilenames(folder)
    solution = []


    sol = []
    for iter in range(Cnt):
        if iter % 10 == 0:
            print(iter)
        while True:
            index = random.randint(0, len(filenames) - 1)
            filename = filenames[index]
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE) # citesc grayscale

            if img.shape[0] < H or img.shape[1] < W:
                continue

            y = random.randint(0, img.shape[0] - H)
            x = random.randint(0, img.shape[1] - W)
            img = img[y : y + H, x : x + W]
            sol.append((func(generateInputFromOutput(img)).cuda(), func(img).cuda()))
            break
            
    return sol


lock = threading.Lock()

def generateBatch(data, H, W, batch_ind, batch_size, folder, generateInputFromOutput):

    v = generateData(H, W, batch_size, folder, generateInputFromOutput)

    for i in range(batch_size):
        lock.acquire()
        data[i + batch_ind * batch_size] = v[i]
        lock.release()

def generateDataThreads(H, W, Cnt, folder, generateInputFromOutput):
    data = [{}] * Cnt

    threads = 4
    t = []

    for i in range(threads):
        t.append(threading.Thread(target = generateBatch, args = (data, H, W, i, Cnt // threads, folder, generateInputFromOutput)))

    for i in range(threads):
        t[i].start()

    for i in range(threads):
        t[i].join()

    return data

