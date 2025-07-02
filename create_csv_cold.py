import os
import csv
import random
import numpy as np
from sklearn.neighbors import KDTree
from functions import create_path
import torchvision.datasets as dset


csvDir = create_path(os.path.join("CSV", "EXP_COLD"))
datasetDir = "/home/arvc/Marcos/INVESTIGACION/0_DATASETS/COLD/"


condIlum = ['Cloudy', 'Night', 'Sunny']


def get_coords(imageDir):
    idxX, idxY, idxA = imageDir.index('_x'), imageDir.index('_y'), imageDir.index('_a')
    x, y = float(imageDir[idxX + 2:idxY]), float(imageDir[idxY + 2:idxA])
    return x, y


def train(epochLength, tree, rPos, rNeg, env="FR_A"):

    with open(f'{csvDir}/Train_{env}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImgAnc", "ImgPos", "ImgNeg"])

        for i in range(epochLength):
            idxAnc = random.randrange(len(imgsList))
            imgAnc = imgsList[idxAnc]
            coordsAnc = coordsList[idxAnc]

            indices = tree.query_radius(coordsAnc.reshape(1, -1), r=rPos)[0]
            idxPos = random.choice(indices)
            while idxAnc == idxPos:
                idxPos = random.choice(indices)
            imgPos = imgsList[idxPos]

            indices = tree.query_radius(coordsAnc.reshape(1, -1), r=rNeg)[0]
            idxNeg = random.randrange(len(imgsList))
            while idxNeg in indices or idxAnc == idxNeg:
                idxNeg = random.randrange(len(imgsList))
            imgNeg = imgsList[idxNeg]

            writer.writerow([imgAnc, imgPos, imgNeg])
    return


def validation(env="FR_A"):
    valDir = os.path.join(datasetDir, env, "Validation")
    with open(f'{csvDir}/Validation_{env}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "CoordX", "CoordY"])

        for room in rooms:
            roomDir = os.path.join(valDir, room)
            imgsVal = os.listdir(roomDir)
            for image in imgsVal:
                x, y = get_coords(image)
                writer.writerow([os.path.join(room, image), x, y])
    return


def test(env="FR_A", il="Cloudy"):
    testDir = os.path.join(datasetDir, env, "Test" + il)
    if os.path.exists(testDir):
        with open(f'{csvDir}/Test_{env}_{il}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Img", "CoordX", "CoordY"])
            for room in rooms:
                roomDir = os.path.join(testDir, room)
                imgsTest = os.listdir(roomDir)
                for image in imgsTest:
                    x, y = get_coords(image)
                    writer.writerow([os.path.join(room, image), x, y])
    return


def visual_model(env="FR_A"):
    vmDir = os.path.join(datasetDir, env, "Train")
    with open(f'{csvDir}/VisualModel_{env}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "CoordX", "CoordY"])

        imgsVM, coordsVM = [], []
        for room in rooms:
            roomDir = os.path.join(vmDir, room)
            imgsDir = os.listdir(roomDir)
            for image in imgsDir:
                x, y = get_coords(image)
                imgsVM.append(os.path.join(room, image))
                coordsVM.append(np.array([x, y]))
                writer.writerow([os.path.join(room, image), x, y])

    return imgsVM, coordsVM


envs = ["FR_B", "SA_A", "SA_B"]
for env in envs:
    trainDataset = dset.ImageFolder(root=f"{datasetDir}{env}/Train")
    rooms = trainDataset.classes
    imgsList, coordsList = visual_model(env=env)
    for ilum in condIlum:
        test(il=ilum, env=env)
    if env == "FR_A":
        treeVM = KDTree(coordsList, leaf_size=2)
        train(epochLength=100000, tree=treeVM, rPos=0.4, rNeg=0.4)
        validation()
