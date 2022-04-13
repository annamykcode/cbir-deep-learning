import os
import sys
import logging
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from flask import Flask

import joblib
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from constants import *
# from models import *
from models import pretrained_models
from models.image_dataset import ImageDataset
from models.utils import predict, label_to_vector

# TODO: change to simple logger
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# hook variable for VGG-16 image embeddings
hook_features = []


# FIXME: change hooks into simple output
def get_features():
    """ Hook for extracting image embeddings from the layer that is attached to.

    Returns:
        hook, as callable.
    """
    def hook(model, input, output):
        global hook_features
        hook_features = output.detach().cpu().numpy()
    return hook


def create_docs(directory, model, pca, transform, mapping):
    """ Read CIFAR-10 train data and create Elasticsearch indexable documents.

    The image documents structure is the following: ("id", "filename", "path", "features").
    The "features" field refers to the image feature vector which consists of:
        * the image embeddings found by the deep-learning model and then reduced using PCA,
        * the one-hot class label vector.

    Args:
        directory:
            CIFAR-10 train data directory, as string.
        model:
            deep-learning model, as Pytorch object.
        pca:
           Principal Component Analysis (PCA), as scikit-learn model.
        transform:
            image transformations, as Pytorch object.
        mapping:
            CIFAR-10 label to index mapping, as dictionary.

    Returns:
        image documents, as list of dictionaries.
        number of total features, as integer.
    """
    if not os.path.isdir(directory):
        app.logger.error(f"Provided path doesn't exist or isn't a directory ...")
        return None, 0
    elif model is None:
        app.logger.error(f"Provided deep-learning model is None ...")
        return None, 0
    elif pca is None:
        app.logger.error(f"Provided PCA model is None ...")
        return None, 0

    data = []
    num_features = 0
    for file in tqdm(os.listdir(directory), desc="creating docs for images"):
        path = os.path.join(directory, file)

        with Image.open(path) as image:
            # create dataset and dataloader objects for Pytorch
            dataset = ImageDataset([image], transform)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

            # pass image trough deep-learning model to gain the image embedding vector
            predict(dataloader, model, device)
            # extract the image embeddings vector
            embedding = hook_features
            # reduce the dimensionality of the embedding vector
            embedding = pca.transform(embedding)

            # get image class label as one-hot vector
            label_str = file[file.find('-') + 1: file.find('.')]
            # FIXME: change labels from imagename to labels from file
            label_vec = label_to_vector(label_str, mapping)

            # concatenate embeddings and label vector
            features_vec = np.concatenate((embedding, label_vec), axis=None)
            num_features = features_vec.shape[0]  # total number of image features

            doc = {
                'id': file[0: file.find('-')],
                'filename': file,
                'path': path,
                'features': features_vec.tolist()
            }
            data.append(doc)

    return data, num_features


if __name__ == "__main__":
    # path for CIFAR-10 train and test datasets
    # dir_train = r'C:\Users\ann\Code\challenges\cbir-deep-learning\static\cifar10\train'
    dir_train = r'../static/ford/train'
    # dir_test = '../static/cifar10/test'

    save_path = "../es_db/ford.json"

    # get available device (CPU/GPU)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    app.logger.info(f'Using {device} device ...')

    app.logger.info(f'Loading VGG-16 model from {PATH_VGG_16} ...')
    # initialize VGG-16
    model = pretrained_models.initialize_model(pretrained=True,
                                               num_labels=len(LABEL_MAPPING),
                                               feature_extracting=True)
    # load VGG-16 pretrained weights
    # model.load_state_dict(torch.load(path_vgg_16, map_location='cuda:0'))
    model.load_state_dict(torch.load(PATH_VGG_16, map_location='cpu'))
    # send VGG-16 to CPU/GPU
    model.to(device)
    # register hook
    model.classifier[5].register_forward_hook(get_features())

    app.logger.info(f'Loading PCA model from {PATH_PCA} ...')
    # load PCA pretrained model
    pca = joblib.load(PATH_PCA)

    # image transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    app.logger.info("Loading CIFAR-10 train data and creating Elasticsearch documents ...")
    images, num_features = create_docs(dir_train, model, pca, transform, LABEL_MAPPING)
    if (images is None) or (num_features == 0):
        app.logger.error("Number of Elasticsearch documents is 0 ...")
        sys.exit(1)

    # save into json to file to be readable
    with open(save_path, "w") as outfile:
        json.dump(images, outfile)
    print(f"saved into {save_path}")

    # logger.info("Loading CIFAR-10 test data and creating Elasticsearch queries ...")
    # queries = create_queries(dir_test, model, pca, transform, len(label_mapping))
    # if queries is None:
    #     logger.error("Number of Elasticsearch queries is 0 ...")
    #     sys.exit(1)
