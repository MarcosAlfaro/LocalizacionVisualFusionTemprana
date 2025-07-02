"""
This script includes several functions that are often used by the other scripts
"""

import torch
import numpy as np
import os
from PIL import Image
from itertools import combinations
import torchvision.datasets as dset


# from config import PARAMS
#
# device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')


def create_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


def get_coords(imageDir):
    idxX, idxY, idxA = imageDir.index('_x'), imageDir.index('_y'), imageDir.index('_a')
    x, y = float(imageDir[idxX + 2:idxY]), float(imageDir[idxY + 2:idxA])
    return x, y


def get_rooms_env(env):
    rooms = dset.ImageFolder(root=os.path.join(PARAMS.datasetDir, env, "Train")).classes
    return rooms


def get_cond_ilum(env):
    if env in ["FR_A", "SA_B"]:
        condIlum = ['Cloudy', 'Night', 'Sunny']
    elif env == "FR_B":
        condIlum = ['Cloudy', 'Sunny']
    elif env == "SA_A":
        condIlum = ['Cloudy', 'Night']
    else:
        raise ValueError("Environment not available. Valid environments: FR_A, FR_B, SA_A, SA_B")
    return condIlum


def get_loss_metric(loss):
    cosine_sim_list = ["circle loss", "CL", "angular loss", "AL", "lazy cosine", "LC",
                       "lazy cosine v2", "LC2", "batch cosine", "BC", "batch cosine v2", "BC2"]
    if loss in cosine_sim_list:
        loss_metric = "cos_similarity"
    else:
        loss_metric = "eucl_distance"
    return loss_metric


def calculate_geometric_distance(xa, ya, xb, yb):
    d = np.linalg.norm(np.array([xa, ya]) - np.array([xb, yb]))
    return d


def max_distance(points):
    max_dist = 0

    for p1, p2 in combinations(points, 2):
        dist = np.linalg.norm(p1 - p2)
        if dist > max_dist:
            max_dist = dist
    return max_dist


def write_csv_from_dir(directory, writer):
    imgs_path, coords = [], []
    rooms = sorted(os.listdir(directory))
    for room in rooms:
        idxRoom = rooms.index(room)
        roomDir = os.path.join(directory, room)
        imgsList = os.listdir(roomDir)
        for image in imgsList:
            image_path = os.path.join(roomDir, image)
            x, y = get_coords(image)
            imgs_path.append(image_path)
            coords.append(np.array([x, y]))
            writer.writerow([image_path, idxRoom, x, y])
    return imgs_path, coords


def process_image(image, tf):
    image = Image.open(image)
    if tf is not None:
        image = tf(image)
    return image


def late_fusion(rgb, depth, method):
    if method == "concat":
        out = torch.cat((rgb, depth), dim=1)
    elif method == "sum":
        out = rgb + depth
    elif method == "weighted":
        w = PARAMS.w
        out = w * rgb + (1 - w) * depth
    else:
        raise ValueError("Non-valid method. Late fusion methods: concat, sum, weighted")
    return out


def load_model_ef(pretrained_model, num_features=0, weightDir="", device="cuda:0"):
    if pretrained_model == "eigenplaces":
        model = torch.hub.load(repo_or_dir="gmberton/eigenplaces", model="get_trained_model",
                               backbone="VGG16", fc_output_dim=512)
    elif pretrained_model == "cosplace":
        model = torch.hub.load(repo_or_dir="gmberton/cosplace", model="get_trained_model",
                               backbone="VGG16", fc_output_dim=512)
    else:
        raise ValueError("Load a valid model")

    if weightDir != "":
        model.backbone[0] = torch.nn.Conv2d(in_channels=3+num_features, out_channels=64, kernel_size=(3, 3),
                                                    stride=(1, 1), padding=(1, 1))
        state_dict = torch.load(weightDir, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"Test {weightDir}")
    else:
        if num_features != 0:
            with torch.no_grad():
                state_dict = model.state_dict()
                weights_conv1 = state_dict['backbone.0.weight'] # Copiar los pesos originales de RGB
                # mean_weights_conv1 = weights_conv1.mean(dim=1, keepdim=True)
                new_weights_conv1 = weights_conv1
                for n in range(num_features):
                    mean_weights_conv1 = weights_conv1[:, n % 3, :, :].unsqueeze(1)  # Use RGB channels for mean
                    new_weights_conv1 = torch.cat((new_weights_conv1, mean_weights_conv1), dim=1)
                state_dict['backbone.0.weight'] = new_weights_conv1
                model.backbone[0] = torch.nn.Conv2d(in_channels=3+num_features, out_channels=64, kernel_size=(3, 3),
                                                    stride=(1, 1), padding=(1, 1))
                model.load_state_dict(state_dict)
        print(f"Using pretrained model")

    return model


def load_model_lf(pretrained_model, backbone="VGG16", output_dim=512, num_features=0, weightDir="", device="cuda:0"):
    if pretrained_model == "eigenplaces":
        model = torch.hub.load(repo_or_dir="gmberton/eigenplaces", model="get_trained_model",
                               backbone=backbone, fc_output_dim=output_dim)
    elif pretrained_model == "cosplace":
        model = torch.hub.load(repo_or_dir="gmberton/cosplace", model="get_trained_model",
                               backbone=backbone, fc_output_dim=output_dim)
    else:
        raise ValueError("Load a valid model")

    if weightDir != "":
        if num_features == 0:
            with torch.no_grad():
                state_dict = torch.load(weightDir, map_location=device, weights_only=False)
                model.load_state_dict(state_dict)
        else:
            if backbone == "VGG16":
                model.backbone[0] = torch.nn.Conv2d(in_channels=num_features, out_channels=64, kernel_size=(3, 3),
                                                            stride=(1, 1), padding=(1, 1))
            else:
                model.backbone[0] = torch.nn.Conv2d(in_channels=num_features, out_channels=64, kernel_size=(7, 7),
                                                            stride=(2, 2), padding=(3, 3), bias=False)
            state_dict = torch.load(weightDir, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            print(f"Test {weightDir}")
    else:
        if num_features != 0:
            with torch.no_grad():
                state_dict = model.state_dict()
                weights_conv1 = state_dict['backbone.0.weight'] # Copiar los pesos originales de RGB
                mean_weights_conv1 = weights_conv1.mean(dim=1, keepdim=True)
                new_weights_conv1 = mean_weights_conv1
                for n in range(num_features-1):
                    new_weights_conv1 = torch.cat((new_weights_conv1, mean_weights_conv1), dim=1)
                state_dict['backbone.0.weight'] = new_weights_conv1
                if backbone == "VGG16":
                    model.backbone[0] = torch.nn.Conv2d(in_channels=num_features, out_channels=64, kernel_size=(3, 3),
                                                            stride=(1, 1), padding=(1, 1))
                else:
                    model.backbone[0] = torch.nn.Conv2d(in_channels=num_features, out_channels=64, kernel_size=(7, 7),
                                                            stride=(2, 2), padding=(3, 3), bias=False)
                model.load_state_dict(state_dict)
        print(f"Using pretrained model")

    return model