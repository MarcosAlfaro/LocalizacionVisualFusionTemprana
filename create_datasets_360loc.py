"""
THIS PROGRAM CONTAINS ALL THE CLASSES THAT CREATE THE REQUIRED IMAGE SETS TO DO A TRAINING, VALIDATION OR TEST
These classes will be called by training and test programmes
"""
from operator import index

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dset
from PIL import Image
import pandas as pd
import numpy as np
from normalize_features import normalize_npy_files_360Loc
import os


csvDir = os.path.join("CSV", "EXP_360LOC")

datasetDir = "/home/arvc/Marcos/INVESTIGACION/0_DATASETS/360LOC/"



def get_coords(imageDir):
    idxX, idxY, idxA = imageDir.index('_x'), imageDir.index('_y'), imageDir.index('_a')
    x, y = float(imageDir[idxX + 2:idxY]), float(imageDir[idxY + 2:idxA])
    return [x, y]


def process_image(image, f=None, tf=None, maxValue=None):
    if image.endswith((".jpeg", ".jpg", ".png")):
        image = Image.open(image)
    elif image.endswith(".npy"):
        image = np.load(image)
        image /= maxValue
    if tf is not None:
        image = tf(image)
    return image


class Train(Dataset):

    def __init__(self, transform=transforms.ToTensor(), features=None, env="atrium", seq="0", ilum="daytime"):

        trainCSV = pd.read_csv(f'{csvDir}/train360_{env}_{ilum}{seq}.csv')

        self.imgsAnc, self.imgsPos, self.imgsNeg = trainCSV['ImgAnc'], trainCSV['ImgPos'], trainCSV['ImgNeg']
        self.transform = transform
        self.env, self.seq, self.ilum = env, seq, ilum
        self.features = features
        self.maxValues = []
        for f in self.features:
            globalMax = normalize_npy_files_360Loc(f, self.env, self.seq, self.ilum, database=True)
            self.maxValues.append(globalMax)

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]

        anc = process_image(image=imgAnc, tf=self.transform)
        pos = process_image(image=imgPos, tf=self.transform)
        neg = process_image(image=imgNeg, tf=self.transform)

        for f in self.features:
            imgAnc_f = imgAnc.replace(".jpg", ".npy").replace("image_resized", f"FEATURES/{f}")
            imgPos_f = imgPos.replace(".jpg", ".npy").replace("image_resized", f"FEATURES/{f}")
            imgNeg_f = imgNeg.replace(".jpg", ".npy").replace("image_resized", f"FEATURES/{f}")

            anc_f = process_image(image=imgAnc_f, f=f, tf=self.transform, maxValue=self.maxValues[self.features.index(f)])
            pos_f = process_image(image=imgPos_f, f=f, tf=self.transform, maxValue=self.maxValues[self.features.index(f)])
            neg_f = process_image(image=imgNeg_f, f=f, tf=self.transform, maxValue=self.maxValues[self.features.index(f)])

            anc, pos, neg = (torch.cat((anc, anc_f), dim=0),
                             torch.cat((pos, pos_f), dim=0),
                             torch.cat((neg, neg_f), dim=0))

        return anc, pos, neg


    def __len__(self):
        return len(self.imgsAnc)


class Test(Dataset):

    def __init__(self, env="FR_A", ilum="daytime1", features=None, transform=transforms.ToTensor()):

        self.env, self.ilum, self.seq = env, ilum[0:-1], ilum[-1]
        self.features = features

        testCSV = pd.read_csv(f'{csvDir}/test360_{env}_{ilum}.csv')

        self.transform = transform
        self.imgList, self.coordX, self.coordY = testCSV['Img'], testCSV['CoordX'], testCSV['CoordY']
        self.maxValues = []
        for f in self.features:
            globalMax = normalize_npy_files_360Loc(f, self.env, self.seq, self.ilum, database=False)
            self.maxValues.append(globalMax)

    def __getitem__(self, index):

        imgPath, coordX, coordY = self.imgList[index], self.coordX[index].astype(np.float32), self.coordY[index].astype(np.float32)

        img = process_image(image=imgPath, tf=self.transform)

        for f in self.features:
            img_f = imgPath.replace(".jpg", ".npy").replace("image_resized", f"FEATURES/{f}")
            img_f = process_image(image=img_f, f=f, tf=self.transform, maxValue=self.maxValues[self.features.index(f)])
            img = torch.cat((img, img_f), dim=0)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgList)


class VisualModel(Dataset):

    def __init__(self, env="FR_A", ilum="daytime", seq="0", features=None, transform=transforms.ToTensor()):

        vmCSV = pd.read_csv(f'{csvDir}/database360_{env}_{ilum}{seq}.csv')

        self.imgList, self.coordX, self.coordY = vmCSV['Img'], vmCSV['CoordX'], vmCSV['CoordY']
        self.env, self.ilum, self.seq = env, ilum, seq
        self.transform = transform
        self.features = features
        self.maxValues = []
        for f in self.features:
            globalMax = normalize_npy_files_360Loc(f, self.env, self.seq, self.ilum, database=True)
            self.maxValues.append(globalMax)

    def __getitem__(self, index):

        imgPath, coordX, coordY = self.imgList[index], self.coordX[index].astype(np.float32), self.coordY[index].astype(np.float32)

        img = process_image(image=imgPath, tf=self.transform)

        for f in self.features:
            img_f = imgPath.replace(".jpg", ".npy").replace("image_resized", f"FEATURES/{f}")
            img_f = process_image(image=img_f, f=f, tf=self.transform, maxValue=self.maxValues[self.features.index(f)])

            img = torch.cat((img, img_f), dim=0)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgList)
