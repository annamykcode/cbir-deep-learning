import os
import numpy as np
import joblib
import cv2 as cv


def load_cifar10(directory):
    """ Load CIFAR-10 dataset.

    Args:
        directory:
            Directory path, as a string.

    Returns:
        Data and labels, as numpy arrays.
    """
    files = [file for file in os.listdir(directory)]

    images = []
    labels = []
    for file in files:
        path = os.path.join(directory, file)
        image = cv.imread(path)

        images.append(image)
        labels.append(file.split('-')[0])

    return np.stack(images, axis=0), np.stack(labels, axis=0)


def load_image(path):
    """ Loads image from provided path

    Args:
        path:
            Image path, as a string.

    Returns:
        Image, as a numpy array
    """
    image = cv.imread(path)

    return image


def load_model(path):
    """ Loads model saved in .joblib format.

    Args:
        path:
            Path of .joblib model file.

    Returns:
        Model, as Scikit-learn model.
    """
    model = joblib.load(path)

    return model


def save_model(directory, name, model):
    """ Save model in .joblib format.

    Args:
        directory:
            Directory path, as a string.
        name:
            Filename (.joblib extension), as string
        model:
            Model, as a Scikit-learn  model.
    """
    if not os.path.isdir(directory):
        try:
            os.mkdir(directory)
        except OSError as ex:
            print(f'[WARNING] utils.utils.save_model - {ex}')

    file = os.path.join(directory, name)
    joblib.dump(model, file)
