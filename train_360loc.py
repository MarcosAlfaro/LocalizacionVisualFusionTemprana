import torch
import random
import os
import csv
import sys
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
from functions import get_cond_ilum, load_model_ef
import torch.nn as nn
import torch.nn.functional as F
from tests import test_GL, visual_model
import create_datasets_360loc


"""COPIAR MODELOS EXP_360LOC DE DISCO DURO"""
"""HACER CÓDIGO TEST 360LOC"""
"""SACAR RESULTADOS VISUALES (GIFS TRAYECTORIAS)"""


def create_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


class LazyTripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(LazyTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, margin):
        distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
        distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)
        losses = torch.relu(distance_positive - distance_negative + margin)

        return losses.max()



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

csvDir = create_path(os.path.join("CSV", "EXP_COLD", "TRAIN_DATA"))

env = "atrium"

_360Loc_envs = ["atrium", "concourse", "hall", "piatrium"]
_360Loc_trainSeq = ["0", "1", "0", "2"]
_360Loc_condIlum = [["daytime1", "daytime2", "nighttime1", "nighttime2"],
                     ["daytime0", "daytime2", "nighttime0"],
                     ["daytime1", "daytime2", "nighttime1", "nighttime2"],
                    ["daytime0", "daytime1", "nighttime0"]]


"""NETWORK TRAINING"""

with open(f"{csvDir}/Exp_360Loc_Train.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Features", "Iteration", "R@1 daytime1", "R@1 daytime2", "R@1 nightime1", "R@1 nightime2", "R@1 Avg."])

    lossFunction = "lazy triplet"
    margin = 0.5

    baseModelDir = create_path("/home/arvc/Marcos/INVESTIGACION/0_SAVED_MODELS/JA_2025/EXP_360Loc/")
    featuresList = [["INTENSITY"], ["MAGNITUDE"], ["ANGLE"], ["HUE"],
                    ["INTENSITY", "MAGNITUDE"], ["INTENSITY", "HUE"], ["MAGNITUDE", "HUE"],
                    ["INTENSITY", "HUE", "MAGNITUDE"], ["INTENSITY", "MAGNITUDE", "ANGLE", "HUE"]                
]

    for features in featuresList:

        if features == []:
            feat = "RGB"
        else:
            feat = ""
            for f in features:
                feat += f
                if features.index(f) != (len(features) - 1):
                    feat = feat + "_"

        vmDataset = create_datasets_360loc.VisualModel(features=features, ilum="daytime", env="atrium")
        vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

        if featuresList.index(features) == 0:
            coordsVM = []
            for i, vmData in enumerate(vmDataloader, 0):
                _, coords = vmData
                coordsVM.append(coords.detach().numpy()[0])
            treeCoordsVM = KDTree(coordsVM, leaf_size=2)

        criterion = LazyTripletLoss(margin=0.5)
        if criterion == -1:
            sys.exit()

        trainDataset = create_datasets_360loc.Train(features=features)
        trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=4)

        netDir = create_path(os.path.join(baseModelDir, feat))

        net = load_model_ef(pretrained_model="cosplace", num_features=len(features)).float().to(device)

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        

        print("\nNEW TRAINING: ")
        print(f"Early fusion, Features: {features}\n")

        bestRecall = 0

        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for i, data in enumerate(trainDataloader, 0):

            anc, pos, neg = data
            
            anc, pos, neg =  anc.float().to(device), pos.float().to(device), neg.float().to(device)


            optimizer.zero_grad()

            output1, output2, output3 = net(anc), net(pos), net(neg)

            loss = criterion(output1, output2, output3, margin)

            loss.backward()

            optimizer.step()

            if i % int(len(trainDataloader) / 10) == 0 and i > 0:
                print(f"It{i}, Loss:{loss}")

                condIlum = condIlum = _360Loc_condIlum[_360Loc_envs.index(env)]
                recall_at_1,  geomError = np.zeros((len(condIlum), 1)), np.zeros((len(condIlum), 1))

                descriptorsVM, coordsVM, treeCoords = visual_model(model=net, dataloader=vmDataloader)
                for ilum in condIlum:
                    idxIlum = condIlum.index(ilum)

                    testDataset = create_datasets_360loc.Test(ilum=ilum, env=env, features=features)
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    resultsTest = test_GL(model=net, testDataloader=testDataloader, descriptorsVM=descriptorsVM,
                                          treeCoords=treeCoords, coordsVM=coordsVM,
                                          env=env, ilum=ilum, results=["recall_at_1", "geom_error"],
                                          graphics=[], mapDir="")
                    recall_at_1[idxIlum], geomError[idxIlum] = resultsTest[0], resultsTest[1]

                avg_recall_at_1, avg_geomError = np.average(recall_at_1), np.average(geomError)
                print(f"Env: {env}, R@1 = {avg_recall_at_1}, Geom. Error = {avg_geomError}\n")

                if avg_recall_at_1 > bestRecall:
                    bestRecall = avg_recall_at_1
                    netName = os.path.join(netDir, f"net_exp1.pth")
                    torch.save(net.state_dict(), netName)
                    print("Saved model")
                    writer.writerow([features, str(i + 1), recall_at_1[0][0], recall_at_1[1][0], recall_at_1[2][0], avg_recall_at_1])

                net.train(True)

        print(f"Training finished, Current Loss: {loss}")

        condIlum = get_cond_ilum(env)
        recall_at_1, geomError = np.zeros((len(condIlum), 1)), np.zeros((len(condIlum), 1))

        descriptorsVM, coordsVM, treeCoords = visual_model(model=net, dataloader=vmDataloader)
        for ilum in condIlum:
            idxIlum = condIlum.index(ilum)

            testDataset = create_datasets_360loc.Test(il=ilum, env=env, features=features)
            testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

            resultsTest = test_GL(model=net, testDataloader=testDataloader, descriptorsVM=descriptorsVM,
                                  treeCoords=treeCoords, coordsVM=coordsVM,
                                  env=env, ilum=ilum, results=["recall_at_1", "geom_error"],
                                  graphics=[], mapDir="")
            recall_at_1[idxIlum], geomError[idxIlum] = resultsTest[0], resultsTest[1]

        avg_recall_at_1, avg_geomError = np.average(recall_at_1), np.average(geomError)
        print(
            f"Env: {env}, R@1 = {avg_recall_at_1}, Geom. Error = {avg_geomError}\n")

        if avg_recall_at_1 > bestRecall:
            bestRecall = avg_recall_at_1
            netName = os.path.join(netDir, f"net_exp1.pth")
            torch.save(net.state_dict(), netName)
            print("Modelo guardado")
            writer.writerow([features, "End", recall_at_1[0][0], recall_at_1[1][0], recall_at_1[2][0], avg_recall_at_1])
