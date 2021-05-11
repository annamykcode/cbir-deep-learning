import os
from flask import Flask
from flask import render_template
from flask import request
import numpy as np
from indexer.image_indexer import ImageIndexer
from preprocessor.preprocessor import to_grayscale
from features.extractors import features_sift, features_hog
from models import bovw
from utils.utils import load_image, load_model

PATH_BOVW_MODEL = '.\\saved-model\\kmeans-500.joblib'
PATH_CLASSIFIER = '.\\saved-model\\logistic-regression.joblib'
INDEX_NAME = 'image-retrieval'

bovw_model = load_model(PATH_BOVW_MODEL)
clf = load_model(PATH_CLASSIFIER)
app = Flask(__name__)


@app.route('/')
def load_page():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def search():
    query = request.form['query']

    print(f'[INFO] - Searching index for "{query}..."')

    query_image_rgb = load_image(query)
    query_image_grayscale = to_grayscale(query_image_rgb)

    query_kp_sift, query_des_sift = features_sift(query_image_grayscale)
    query_des_hog = features_hog(query_image_grayscale)

    query_sift_features = bovw.get_vector_representation(bovw_model, query_des_sift)

    # early fusion (HOG & SIFT)
    query_fused_features = np.concatenate((query_des_hog, query_sift_features), axis=None)

    # get image label one-hot vector
    pred = clf.predict(query_fused_features.reshape(1, -1))
    query_label = to_vector(pred[0])

    # concatenate features and label vector
    query_vec = np.concatenate((query_fused_features, query_label), axis=None)

    results = []
    res = es.query_index(index_name=INDEX_NAME, features=query_vec)

    for hit in res['hits']['hits']:
        res = {
            'src': 'data/train/' + hit["_source"]["path"].split('\\')[-1],
            'score': hit["_score"]
        }
        results.append(res)

    return render_template('index.html', results=results)


def label_vector(label):
    labels = {'airplane': 0,
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

    vec = np.zeros(10)
    vec[labels[label]] = 1

    return vec


def to_vector(label):
    vec = np.zeros(10)

    vec[label] = 1

    return vec


if __name__ == '__main__':
    print('[INFO] main - Loading CIFAR-10 data for indexing...')

    DIR_TRAIN = '.\\static\\data\\train'
    files = [file for file in os.listdir(DIR_TRAIN)]

    print('[INFO] main - Extracting HOG and SIFT features based on BoVW model...')

    dataset = []
    num_features = 0
    for file in files:
        path = os.path.join(DIR_TRAIN, file)

        image_rgb = load_image(path)
        image_grayscale = to_grayscale(image_rgb)

        kp_sift, des_sift = features_sift(image_grayscale)
        des_hog = features_hog(image_grayscale)

        sift_features = bovw.get_vector_representation(bovw_model, des_sift)
        # skip images w/o features
        if (sift_features is None) or (des_hog is None):
            continue

        # early fusion (HOG & SIFT)
        fused_features = np.concatenate((des_hog, sift_features), axis=None)

        # get image label one-hot vector
        label = file.split('-')[0]
        label_vec = label_vector(label)

        # concatenate features and label vector
        features_vec = np.concatenate((fused_features, label_vec), axis=None)

        num_features = features_vec.shape[0]

        doc = {
            'path': path,
            'features': features_vec
        }
        dataset.append(doc)

    print('[INFO] main - Creating index...')

    # run Elasticsearch on localhost
    es = ImageIndexer(hosts=['localhost:9200'])

    es.create_index(index_name=INDEX_NAME, num_features=num_features)

    print('[INFO] main - Indexing image files...')

    es.index_docs(index_name=INDEX_NAME, dataset=dataset)

    print('[INFO] main - Running application...')

    app.run()
