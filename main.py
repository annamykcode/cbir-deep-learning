import os
import numpy as np

from image import imageprocessor
from image import extractor
from image import utils as image_utils
from models.bovw import BoVW
from models import utils as model_utils
from index.indexer import Indexer
from index.searcher import Searcher

from elasticsearch import Elasticsearch
# from flask import Flask
# from flask import render_template
# from flask import request


def load_cifar10_data(directory, bovw, mapping):
    """ Read CIFAR-10 train data and prepare them for indexing.

    For the CIFAR-10 images to be indexed in Elasticsearch, they need to have the structure of a document with indexable
    fields. The structure chosen is the following: ("id", "filename", "path", "features").

    Args:
        directory:
            CIFAR-10 train data directory.
        bovw:
            bag of visual words model, as BoVW object.
        mapping:
            CIFAR-10 label to index mapping, as dictionary.

    Returns:
        images (documents), as list of dictionaries.
        number of total features, as integer.
    """
    if not os.path.isdir(directory):
        # TODO: log error not a dir or doesn't exist
        return None, 0

    data = []
    num_features = 0
    for file in os.listdir(directory):
        path = os.path.join(directory, file)

        image_rgb = (image_utils.load_image(path)).astype('uint8')
        if image_rgb is None:
            # TODO: log error
            return None, 0

        # process image (from RGB to grayscale)
        image_grayscale = (imageprocessor.to_grayscale(image_rgb)).astype('uint8')

        # extract HOG and SIFT descriptors
        des_hog = extractor.hog_features(image_grayscale)
        kp_sift, des_sift = extractor.sift_features(image_grayscale)

        # get bag of visual words representation of SIFT features
        sift_features = bovw.get_vector_representation(des_sift)

        # early fusion (HOG & SIFT features)
        fused_features = np.concatenate((des_hog, sift_features), axis=None)

        # get image class label as one-hot vector
        label = file[file.find('-') + 1: file.find('.')]
        label_vec = model_utils.label_to_vector(label, mapping)

        # concatenate features and label vector
        features_vec = np.concatenate((fused_features, label_vec), axis=None)

        num_features = features_vec.shape[0]  # total number of features of an image

        doc = {
            'id': file[0: file.find('-')],
            'filename': file,
            'path': path,
            'features': features_vec
        }
        data.append(doc)

    return data, num_features


def load_cifar10_queries(directory, bovw, clf, num_labels):
    """ Read CIFAR-10 test data and create queries from them.

    For the CIFAR-10 images to be valid queries, they need to have the following structure:
    ("id", "filename", "path", "features").

    Args:
        directory:
            CIFAR-10 test data directory.
        bovw:
            bag of visual words model, as BoVW object.
        clf:
           classifier for image prediction, as Scikit-learn object.
        num_labels:
            number of class labels in CIFAR-10, as integer.

    Returns:
        query images, as list of dictionaries.
    """
    if not os.path.isdir(directory):
        # TODO: log error not a dir or doesn't exist
        return None

    queries = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)

        image_rgb = (image_utils.load_image(path)).astype('uint8')
        if image_rgb is None:
            # TODO: log error
            return None

        # process image (from RGB to grayscale)
        image_grayscale = (imageprocessor.to_grayscale(image_rgb)).astype('uint8')

        # extract HOG and SIFT descriptors
        des_hog = extractor.hog_features(image_grayscale)
        kp_sift, des_sift = extractor.sift_features(image_grayscale)

        # get bag of visual words representation of SIFT features
        sift_features = bovw.get_vector_representation(des_sift)

        # early fusion (HOG & SIFT features)
        fused_features = np.concatenate((des_hog, sift_features), axis=None)

        # predict the query image class label
        pred = clf.predict(fused_features.reshape(1, -1))
        # get image class label as one-hot vector
        label_vec = np.zeros(num_labels)
        label_vec[pred[0]] = 1

        # concatenate features and label vector
        features_vec = np.concatenate((fused_features, label_vec), axis=None)

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
PATH_BOVW_MODEL = 'saved-model/bovw.joblib'
PATH_CLASSIFIER = 'saved-model/logistic-regression.joblib'
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
    records = []
    with open('data/qrels.txt', 'r') as f:
        records = f.readlines()

    with open('data/rels.txt', 'w') as f:
        for record in records:
            f.write(record)
    # bovw_model = model_utils.load_model(PATH_BOVW_MODEL)
    # bovw = BoVW(bovw_model)
    # clf = model_utils.load_model(PATH_CLASSIFIER)

    # print('[INFO] main - Loading CIFAR-10 data...')
    # images, num_features = load_cifar10_data(DIR_TRAIN, bovw, LABEL_MAPPING)
    # TODO log

    # print('[INFO] main - Loading CIFAR-10 queries...')
    # queries = load_cifar10_queries(DIR_TEST, bovw, clf, len(LABEL_MAPPING))
    # TODO log

    # print('[INFO] main - Starting Elasticsearch...')
    # run Elasticsearch on localhost
    # es = Elasticsearch(hosts=['localhost:9200'], timeout=30, retry_on_timeout=True)

    # print('[INFO] main - Creating index...')
    # indexer = Indexer()
    # indexer.create_index(es=es, name="cifar10", number_of_shards=30, number_of_replicas=0, num_features=num_features)

    # print('[INFO] main - Indexing image files...')
    # indexer.index_images(es=es, name="cifar10", images=images)

    # print('[INFO] main - Searching index...')
    # searcher = Searcher()
    # results = searcher.search_index(es=es, name="cifar10", queries=queries, k=100)

    # print('[INFO] main - Writing results...')
    # write_results(results, 'search-engine-results/bovw/results-100.txt')

    # print('[INFO] main - Running application...')
    # app.run()
