import torch
import numpy as np
from sklearn.neighbors import KDTree
import torch.nn.functional as F
import create_figures



def visual_model(model, dataloader, device="cuda"):
    coordsVM, descriptorsVM = [], []
    model.eval()
    with torch.no_grad():
        for i, vmData in enumerate(dataloader, 0):
            img, imgCoords = vmData
            img = img.float().to(device)
            output = model(img) 
            descriptorsVM.append(output)
            coordsVM.append(imgCoords.detach().numpy()[0])

    descriptorsVM = torch.squeeze(torch.stack(descriptorsVM)).to(device)
    treeCoordsVM = KDTree(coordsVM, leaf_size=2)
    model.train(True)
    return descriptorsVM, coordsVM, treeCoordsVM


def test_GL(model, testDataloader, descriptorsVM, treeCoords, coordsVM,
            ilum, results, graphics, mapDir, env="FR_A", device="cuda"):

    if env in ["FR_A", "FR_B", "SA_A", "SA_B"]:
        d_threshold = 0.5
    elif env in ["atrium", "hall", "piatrium"]:
        d_threshold = 10
    else:
        d_threshold = 5

    """TEST"""
    geomError, recall_at_1, recall_at_5, recall_at_10, resultValues = 0, 0, 0, 0, []
    model.eval()
    with torch.no_grad():
        
        coordsMapTest = []
        for i, data in enumerate(testDataloader, 0):

            imgTest, coordsImgTest = data
            imgTest = imgTest.float().to(device)

            output = model(imgTest)
            idxDesc = torch.argsort(F.pairwise_distance(output, descriptorsVM), dim=0, descending=False).cpu().numpy()
            
            coordsImgTest = coordsImgTest.detach().numpy()[0]
            _, idxGeom = treeCoords.query(coordsImgTest.reshape(1, -1), k=len(descriptorsVM))
            idxMinReal = idxGeom[0][0]

            coordsClosestImg = coordsVM[idxMinReal]
            
            for k in range(10):
                coordsPredictedImg = coordsVM[idxDesc[k]]
                errorDist = np.linalg.norm(coordsImgTest - coordsPredictedImg)
                if k == 0:
                    geomError += errorDist
                if errorDist <= d_threshold:
                    break
                k += 1

            if k == 0:
                recall_at_1 += 1
                recall_at_5 += 1
                recall_at_10 += 1
            elif k < 5:
                recall_at_5 += 1
                recall_at_10 += 1
            elif k < 10:
                recall_at_10 += 1
            
            

            coordsMapTest.append([coordsPredictedImg[0], coordsPredictedImg[1],
                                              coordsImgTest[0], coordsImgTest[1], k])
            

        # Results for each lighting condition
        recall_at_1 *= 100 / len(testDataloader)
        recall_at_5 *= 100 / len(testDataloader)
        recall_at_10 *= 100 / len(testDataloader)
        geomError /= len(testDataloader)

        print(f"Env: {env}, Ilum: {ilum}, "
              f"Error = {geomError} m, R@1 = {recall_at_1} %\n")

        for r in results:
            if r == "recall_at_1":
                resultValues.append(recall_at_1)
            if r == "recall_at_5":
                resultValues.append(recall_at_5)
            if r == "recall_at_10":
                resultValues.append(recall_at_10)
            if r == "geom_error":
                resultValues.append(geomError)

        for g in graphics:
            if g == "prediction_map":
                create_figures.display_coord_map(direc=mapDir, mapVM=coordsVM, mapTest=coordsMapTest,
                                                  imgFormat=env, ilum=ilum, k=10)

    return resultValues



