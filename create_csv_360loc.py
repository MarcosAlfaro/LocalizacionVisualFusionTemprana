import os
import csv
import random
import math
import numpy as np
from sklearn.neighbors import KDTree
from functions import create_path


datasetDir = "/home/arvc/Marcos/INVESTIGACION/0_DATASETS/360LOC/"
csvDir = create_path(os.path.join("CSV", "EXP_360LOC"))


def read_coords_txt(env="atrium", imgType="360", ilum="daytime", imgSet="database", seq="0"):
    
    if imgSet == "database":
        txtName = f'{datasetDir}{env}/pose/{imgType}_mapping_gt.txt'
    else:
        if ilum == "daytime":
            il = "day"
        elif ilum == "nighttime":   
            il = "night"
        txtName = f'{datasetDir}{env}/pose/query_gt_{imgType}_{il}.txt'

    with open(txtName, 'r') as file:
        lines = file.readlines()
    coords = {}
    for line in lines:
        line = line.split()
        imgPath = line[0]
        seqIdx = imgPath.split("/")[-3].split("_")[-1]
        if seqIdx != seq and imgSet != "database":
            continue
        coordX = float(line[1])
        coordY = float(line[2])
        coordZ = float(line[3])
        coords[imgPath] = (coordX, coordY, coordZ)

    return coords




# this function has to write a .csv file with the following columns: image path, coordX, coordY, coordZ
def write_csv(env="atrium", imgType="360", imgSet="database", ilum="daytime", seq="0"):

    csvName = f'{csvDir}/{imgSet}{imgType}_{env}_{ilum}{seq}.csv'

    coordsDict = read_coords_txt(env=env, imgType=imgType, ilum=ilum, imgSet=imgSet, seq=seq)
    coords = []
    for img in coordsDict:
        coords.append(np.array(coordsDict[img]))
    imgsList = list(coordsDict.keys())

    with open(csvName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "CoordX", "CoordY", "CoordZ"])
        for img in imgsList:
            imgPath = os.path.join(datasetDir, env, img)
            imgPath = imgPath.replace("image", "image_resized")
            coordX, coordY, coordZ = coordsDict[img]
            writer.writerow([imgPath, coordX, coordY, coordZ])

    return 



def write_train_csv(env="atrium", imgType="360", ilum="daytime", seq="0", rPos=0.5, rNeg=0.5):

    imgDir = f"{datasetDir}{env}/mapping/daytime_{imgType}_{seq}/image_resized"
    with open(f'{csvDir}/train{imgType}_{env}_{ilum}{seq}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImgAnc", "ImgPos", "ImgNeg"])

        coordsDict = read_coords_txt(env=env, imgType=imgType, ilum=ilum, imgSet="database")
        coordsList = []
        for img in coordsDict:
            coordsList.append(np.array(coordsDict[img]))
        imgsList = list(coordsDict.keys())
        tree = KDTree(coordsList, leaf_size=2)

        for _ in range(100000):
            idxAnc = random.randrange(len(imgsList))
            anc = imgsList[idxAnc]
            coordsAnc = coordsList[idxAnc]

            indexes = tree.query_radius(coordsAnc.reshape(1, -1), r=rPos)[0]
            idxPos = random.choice(indexes)
            while idxAnc == idxPos:
                idxPos = random.choice(indexes)
            pos = imgsList[idxPos]

            indexes = tree.query_radius(coordsAnc.reshape(1, -1), r=rNeg)[0]
            idxNeg = random.randrange(len(imgsList))
            while idxNeg in indexes:
                idxNeg = random.randrange(len(imgsList))
            neg = imgsList[idxNeg]
            anc, pos, neg = anc.replace("image", "image_resized"), pos.replace("image", "image_resized"), neg.replace("image", "image_resized")
            writer.writerow([os.path.join(datasetDir, env, anc), os.path.join(datasetDir, env, pos), os.path.join(datasetDir, env, neg)])
    return


write_csv(env="atrium", imgType="360", ilum="daytime", imgSet="database", seq="0")
write_csv(env="atrium", imgType="360", ilum="daytime", imgSet="test", seq="1")
write_csv(env="atrium", imgType="360", ilum="daytime", imgSet="test", seq="2")
write_csv(env="atrium", imgType="360", ilum="nighttime", imgSet="test", seq="1")
write_csv(env="atrium", imgType="360", ilum="nighttime", imgSet="test", seq="2")
write_train_csv(env="atrium", imgType="360", ilum="daytime", seq="0", rPos=1, rNeg=1)

write_csv(env="concourse", imgType="360", ilum="daytime", imgSet="database", seq="1")
write_csv(env="concourse", imgType="360", ilum="daytime", imgSet="test", seq="0")
write_csv(env="concourse", imgType="360", ilum="daytime", imgSet="test", seq="2")
write_csv(env="concourse", imgType="360", ilum="nighttime", imgSet="test", seq="0")
write_train_csv(env="concourse", imgType="360", ilum="daytime", seq="1", rPos=1, rNeg=1)

write_csv(env="hall", imgType="360", ilum="daytime", imgSet="database", seq="0")
write_csv(env="hall", imgType="360", ilum="daytime", imgSet="test", seq="1")
write_csv(env="hall", imgType="360", ilum="daytime", imgSet="test", seq="2")
write_csv(env="hall", imgType="360", ilum="nighttime", imgSet="test", seq="1")
write_csv(env="hall", imgType="360", ilum="nighttime", imgSet="test", seq="2")
write_train_csv(env="hall", imgType="360", ilum="daytime", seq="0", rPos=1, rNeg=1)

write_csv(env="piatrium", imgType="360", ilum="daytime", imgSet="database", seq="2")
write_csv(env="piatrium", imgType="360", ilum="daytime", imgSet="test", seq="0")
write_csv(env="piatrium", imgType="360", ilum="daytime", imgSet="test", seq="1")
write_csv(env="piatrium", imgType="360", ilum="nighttime", imgSet="test", seq="0")
write_train_csv(env="piatrium", imgType="360", ilum="daytime", seq="2", rPos=1, rNeg=1)

