import os
import numpy as np
import joblib
from PIL import Image
import torch
import torchvision.transforms as transforms
from models.image_dataset import ImageDataset
from torch.utils.data import DataLoader
from models import utils as model_utils
from models import pretrained_models
from index.indexer import Indexer
from index.searcher import Searcher

from elasticsearch import Elasticsearch
# from flask import Flask
# from flask import render_template
# from flask import request

hook_features = None


def load_cifar10_data(directory, model, pca, mapping):
    """ Read CIFAR-10 train data and prepare them for indexing.

    For the CIFAR-10 images to be indexed in Elasticsearch, they need to have the structure of a document with indexable
    fields. The structure chosen is the following: ("id", "filename", "path", "features").

    Args:
        directory:
            CIFAR-10 train data directory.
        model:
            deep-learning model, as Pytorch object.
        pca:
           Principal Component Analysis (PCA), as scikit-learn model.
        mapping:
            CIFAR-10 label to index mapping, as dictionary.

    Returns:
        images (documents), as list of dictionaries.
        number of total features, as integer.
    """
    if not os.path.isdir(directory):
        # TODO: log error not a dir or doesn't exist
        return None, 0

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print('Using {} device'.format(device))

    data = []
    num_features = 0
    for file in os.listdir(directory):
        path = os.path.join(directory, file)

        image = Image.open(path)
        if image is None:
            # TODO: log error
            return None, 0

        dataset = ImageDataset([image], transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        model_utils.predict(dataloader, model, device)

        # extract features
        features = hook_features

        # dimensionality reduction
        features = pca.transform(features)

        # get image class label as one-hot vector
        label_str = file[file.find('-') + 1: file.find('.')]
        label_vec = model_utils.label_to_vector(label_str, mapping)

        # concatenate features and label vector
        features_vec = np.concatenate((features, label_vec), axis=None)

        num_features = features_vec.shape[0]  # total number of features of an image

        doc = {
            'id': file[0: file.find('-')],
            'filename': file,
            'path': path,
            'features': features_vec
        }
        data.append(doc)

    return data, num_features


def load_cifar10_queries(directory, model, pca, num_labels):
    """ Read CIFAR-10 test data and create queries from them.

    For the CIFAR-10 images to be valid queries, they need to have the following structure:
    ("id", "filename", "path", "features").

    Args:
        directory:
            CIFAR-10 test data directory.
        model:
            deep-learning model, as Pytorch object.
        pca:
           Principal Component Analysis (PCA), as scikit-learn model.
        num_labels:
            number of class labels in CIFAR-10, as integer.

    Returns:
        query images, as list of dictionaries.
    """
    if not os.path.isdir(directory):
        # TODO: log error not a dir or doesn't exist
        return None

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print('Using {} device'.format(device))

    queries = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)

        image = Image.open(path)
        if image is None:
            # TODO: log error
            return None

        dataset = ImageDataset([image], transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        pred = model_utils.predict(dataloader, model, device)

        # extract features
        features = hook_features

        # dimensionality reduction
        features = pca.transform(features)

        # get image class label as one-hot vector
        label_vec = np.zeros(num_labels, dtype='int64')
        label_vec[pred] = 1

        # concatenate features and label vector
        features_vec = np.concatenate((features, label_vec), axis=None)

        query = {
            'id': file[0: file.find('-')],
            'filename': file,
            'path': path,
            'features': features_vec
        }
        queries.append(query)

    return queries


def write_results(results, path):
    """ Create search results file (.txt) according to trec_eval specifications.

    The results file has records of the form: (query_id, iteration, doc_id, rank, similarity, run_id).

    Args:
        results:
            search results of the form (query_id, images: [id, filename, path, score]), as list of dictionaries.
        path:
             file path, as string.
    """
    if (results is None) or (not results):
        # TODO: logging
        return
    elif os.path.isdir(path):
        # TODO: logging not a file
        return

    with open(path, 'w') as f:
        iteration = "0"
        rank = "0"
        run_id = "STANDARD"
        for result in results:
            # results file contains records of the form: (query_id, iteration, doc_id, rank, similarity, run_id)
            for image in result["images"]:
                record = f"{result['query_id']} {iteration} {image['id']} {rank} {image['score']} {run_id}\n"
                f.write(record)


def get_features():
    def hook(model, input, output):
        global hook_features
        hook_features = output.detach().cpu().numpy()
    return hook


# @app.route('/')
# def load_page():
#     return render_template('index.html')


# @app.route('/', methods=['POST'])
# def search():
#     query = request.form['query']
#
#     print(f'[INFO] - Searching index for "{query}..."')
#
#     query_image_rgb = load_image(query)
#     query_image_grayscale = to_grayscale(query_image_rgb)
#
#     query_kp_sift, query_des_sift = features_sift(query_image_grayscale)
#     query_des_hog = features_hog(query_image_grayscale)
#
#     query_sift_features = bovw.get_vector_representation(bovw_model, query_des_sift)
#
#     # early fusion (HOG & SIFT)
#     query_fused_features = np.concatenate((query_des_hog, query_sift_features), axis=None)
#
#     # get image label one-hot vector
#     pred = clf.predict(query_fused_features.reshape(1, -1))
#     query_label = to_vector(pred[0])
#
#     # concatenate features and label vector
#     query_vec = np.concatenate((query_fused_features, query_label), axis=None)
#
#     results = []
#     res = es.query_index(index_name=INDEX_NAME, features=query_vec)
#
#     for hit in res['hits']['hits']:
#         res = {
#             'src': 'data/train/' + hit["_source"]["path"].split('\\')[-1],
#             'score': hit["_score"]
#         }
#         results.append(res)
#
#     return render_template('index.html', results=results)

DIR_TRAIN = 'static/cifar10/train'
DIR_TEST = 'static/cifar10/test'
PATH_VGG_16 = 'saved-model/vgg16-weights.pth'
PATH_PCA = 'saved-model/pca.joblib'
INDEX_NAME = 'image-retrieval'
LABEL_MAPPING = {'airplane': 0,
                 'automobile': 1,
                 'bird': 2,
                 'cat': 3,
                 'deer': 4,
                 'dog': 5,
                 'frog': 6,
                 'horse': 7,
                 'ship': 8,
                 'truck': 9,
                 }
# app = Flask(__name__)

if __name__ == '__main__':
    model = pretrained_models.initialize_model(pretrained=True,
                                               num_labels=len(LABEL_MAPPING),
                                               feature_extracting=True)
    model.load_state_dict(torch.load(PATH_VGG_16, map_location='cuda:0'))

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print('Using {} device'.format(device))

    model.to(device)

    # register hook
    model.classifier[5].register_forward_hook(get_features())

    pca = joblib.load(PATH_PCA)

    # print('[INFO] main - Loading CIFAR-10 data...')
    # images, num_features = load_cifar10_data(DIR_TRAIN, model, pca, LABEL_MAPPING)
    # TODO log

    print('[INFO] main - Loading CIFAR-10 queries...')
    queries = load_cifar10_queries(DIR_TEST, model, pca, len(LABEL_MAPPING))
    # TODO log

    # run Elasticsearch on localhost
    print('[INFO] main - Starting Elasticsearch...')
    es = Elasticsearch(hosts=['localhost:9200'], timeout=60, retry_on_timeout=True)

    # print('[INFO] main - Creating index...')
    # indexer = Indexer()
    # indexer.create_index(es=es, name="cifar10", number_of_shards=30, number_of_replicas=0, num_features=num_features)

    # print('[INFO] main - Indexing image files...')
    # indexer.index_images(es=es, name="cifar10", images=images)

    print('[INFO] main - Searching index...')
    searcher = Searcher()
    results = searcher.search_index(es=es, name="cifar10", queries=queries, k=100)

    print('[INFO] main - Writing results...')
    write_results(results, 'search-engine-results/vgg-16/results-100.txt')

    # print('[INFO] main - Running application...')
    # app.run()
