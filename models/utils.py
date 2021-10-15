import numpy as np
import joblib


def load_model(path):
    """ Loads machine learning model.

     The model has to be of .joblib format.

    Args:
        path:
            model file path, as string.

    Returns:
        model, as Scikit-learn model.
    """
    model = joblib.load(path)

    return model


def label_to_vector(label, mapping):
    """ Translate a string label to one-hot vector.

    Args:
        label:
            class label, as string.
        mapping:
            string (label) to index mapping, as dictionary.

    Returns:
        label one-hot vector, as numpy array.
    """
    vec = np.zeros(len(mapping))
    vec[mapping[label]] = 1

    return vec