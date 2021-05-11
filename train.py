from keras.datasets import cifar10

import numpy as np
from sklearn.linear_model import LogisticRegression
from preprocessor.preprocessor import to_grayscale
from features.extractors import features_sift, features_hog
from models import bovw
from utils.utils import save_model


if __name__ == '__main__':
    print(f'[INFO] main - Loading CIFAR-10 dataset...')

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype('uint8')
    X_test = X_test.astype('uint8')

    print(f'[INFO] main - Converting to grayscale...')

    X_train_grayscale = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2])).astype('uint8')
    for i in range(X_train.shape[0]):
        X_train_grayscale[i] = to_grayscale(X_train[i])

    X_test_grayscale = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2])).astype('uint8')
    for i in range(X_test.shape[0]):
        X_test_grayscale[i] = to_grayscale(X_test[i])

    print(f'[INFO] main - Extracting SIFT features...')

    kp_sift_train = []
    des_sift_train = []
    for i in range(X_train_grayscale.shape[0]):
        kp, des = features_sift(X_train_grayscale[i])
        kp_sift_train.append(kp)
        des_sift_train.append(des)

    kp_sift_test = []
    des_sift_test = []
    for i in range(X_test_grayscale.shape[0]):
        kp, des = features_sift(X_test_grayscale[i])
        kp_sift_test.append(kp)
        des_sift_test.append(des)

    print(f'[INFO] main - Extracting HOG features...')

    des_hog_train = []
    for i in range(X_train_grayscale.shape[0]):
        des_hog = features_hog(X_train_grayscale[i])
        des_hog_train.append(des_hog)

    des_hog_test = []
    for i in range(X_test_grayscale.shape[0]):
        des_hog = features_hog(X_test_grayscale[i])
        des_hog_test.append(des_hog)

    print(f'[INFO] main - Extracting SIFT vocabulary...')

    n_clusters = 500
    visual_words = bovw.get_visual_words(descriptors=des_sift_train, n_clusters=n_clusters)

    print(f'[INFO] main - Saving SIFT BoVW model...')

    save_model('.\\saved-model', 'kmeans-500.joblib', visual_words)

    print(f'[INFO] main - Creating SIFT vector representations...')

    X_train_sift = []
    for i in range(X_train_grayscale.shape[0]):
        features = bovw.get_vector_representation(visual_words, des_sift_train[i])
        X_train_sift.append(features)

    X_test_sift = []
    for i in range(X_test_grayscale.shape[0]):
        features = bovw.get_vector_representation(visual_words, des_sift_test[i])
        X_test_sift.append(features)

    print(f'[INFO] main - Early fusion with HOG and SIFT features...')

    X_train_hog_sift = []
    for idx in range(X_train_grayscale.shape[0]):
        fused = np.concatenate((des_hog_train[idx], X_train_sift[idx]), axis=None)
        X_train_hog_sift.append(fused)

    X_test_hog_sift = []
    for idx in range(X_test_grayscale.shape[0]):
        fused = np.concatenate((des_hog_test[idx], X_test_sift[idx]), axis=None)
        X_test_hog_sift.append(fused)

    print(f'[INFO] main - Fitting Logistic Regression...')

    clf = LogisticRegression(C=10, class_weight=None, max_iter=5000)
    clf.fit(X_train_hog_sift, Y_train.ravel())

    print(f'[INFO] main - Saving Logistic Regression...')

    save_model('.\\saved-model', 'logistic-regression.joblib', clf)


