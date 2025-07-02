import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import csv
import numpy as np
import create_datasets_cold
from functions import create_path, load_model_ef, get_loss_metric, get_cond_ilum
from tests import test_GL, visual_model

# if the computer has cuda available, we will use cuda, else, cpu will be used
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# parameter definition
csvDir = create_path(os.path.join("CSV", "EXP_COLD", "RESULTS"))
figuresDir = create_path(os.path.join("FIGURES", "EXP_COLD"))

# tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((384, 192))])
tf = transforms.ToTensor()



envs = ["FR_A", "FR_B", "SA_A", "SA_B"]


with open(csvDir + "/Exp_COLD_results.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Features", 
                     "R@1 FR-A Cloudy", "R@1 FR-A Night", "R@1 FR-A Sunny", "R@1 FR-A Avg.",
                     "R@1 FR-B Cloudy", "R@1 FR-B Sunny", "R@1 FR-B Avg.",
                     "R@1 SA-A Cloudy", "R@1 SA-A Night", "R@1 SA-A Avg.",
                     "R@1 SA-B Cloudy", "R@1 SA-B Night", "R@1 SA-B Sunny", "R@1 SA-B Avg."])

    savedModelsDir = "/home/arvc/Marcos/INVESTIGACION/0_SAVED_MODELS/JA_2025/EXP_COLD/"

    featuresList = os.listdir(savedModelsDir)


    for feat in featuresList:

        if "RGB" in feat:
            num_features = 0
        else:
            num_features = feat.count("_") + 1
            features = list(feat.split("_"))

        net = load_model_ef(pretrained_model="eigenplaces", weightDir=f"{savedModelsDir}{feat}/net_exp1.pth",
                            num_features=num_features).float().to(device)
        
        print(f"TEST EARLY FUSION, FEATURES: {feat}")

        rowCSV = [feat]

        for env in envs:

            vmDataset = create_datasets_cold.VisualModel(env=env, features=features)
            vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

            
            print(f"Environment: {env}\n\n")

            net.eval()

            with torch.no_grad():

                condIlum = get_cond_ilum(env)
                recall_at_1, recall_at_5, recall_at_10, geomError = np.zeros((len(condIlum), 1)), np.zeros((len(condIlum), 1)), np.zeros((len(condIlum), 1)), np.zeros((len(condIlum), 1))

                descriptorsVM, coordsVM, treeCoords = visual_model(model=net,  dataloader=vmDataloader)

                for ilum in condIlum:
                    idxIlum = condIlum.index(ilum)

                    testDataset = create_datasets_cold.Test(il=ilum, env=env, features=features)
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    mapDir = create_path(f"{figuresDir}{env}/{ilum}")
                    resultsTest = test_GL(model=net, testDataloader=testDataloader, descriptorsVM=descriptorsVM,
                                          treeCoords=treeCoords, coordsVM= coordsVM,
                                          env=env, ilum=ilum, results=["recall_at_1", "recall_at_5", "recall_at_10", "geom_error"], 
                                          graphics=["prediction_map"], mapDir=mapDir)
                    recall_at_1[idxIlum], recall_at_5[idxIlum], recall_at_10[idxIlum], geomError[idxIlum] = resultsTest[0], resultsTest[1], resultsTest[2], resultsTest[3]
                    rowCSV.append(resultsTest[0])

                avg_recall_at_1, avg_recall_at_5, avg_recall_at_10, avg_geomError = np.average(recall_at_1), np.average(recall_at_5), np.average(recall_at_10), np.average(geomError)
                rowCSV.append(avg_recall_at_1)
                print(f"Env: {env}, R@1 = {avg_recall_at_1}, R@5 = {avg_recall_at_5}, R@10 = {avg_recall_at_10}, Geom. Error = {avg_geomError}\n")
        writer.writerow(rowCSV)
