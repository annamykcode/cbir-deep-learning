import numpy as np
import joblib


def load_bovw_model(path):
    """ Loads Bag of Visual Words (BoVW) model.

     The BoVW model has to be of .joblib format.

    Args:
        path:
            BoVW model file path, as string.

    Returns:
        BoVW model, as Scikit-learn model.
    """
    model = joblib.load(path)

    return model


class BoVW:
    """ Represents Bag of Visual Words model.

    The general idea is to represent an image as a set of features. Features consists of keypoints and descriptors.
    Keypoints are the “stand out” points in an image, and descriptors are the description of the keypoints. We use
    the keypoints and descriptors to construct vocabularies and represent each image as a frequency histogram of
    features that are in the image.

    Attributes:
        model:
            clustering model, as Scikit-learn object.
    """

    def __init__(self, model):
        """ Initialized Bag of Visual Words model, with given clustering model.

        Args:
            model:
                clustering model, as Scikit-learn object.
        """
        self.model = model

    def get_visual_words(self, descriptors):
        """ Finds the vocabulary of visual words.

        To find the vocabulary of visual words, a clustering model is used. The centroids found by the clustering
        algorithm form the vocabulary.

        Args:
            descriptors:
                descriptors, as a list of numpy arrays.

        Returns:
            fitted clustering model, holding the vocabulary of visual words .
        """
        visual_words = self.model.fit(descriptors)

        return visual_words

    def get_vector_representation(self, descriptors):
        """ Computes the vector representation of an image.

        The representation is based on the image descriptors and a predefined BoVW model. Specifically,
        it's a histogram of the frequencies of visual words from the vocabulary of the BoVW.

        Args:
            descriptors:
                descriptors, as numpy array.

        Returns:
            Vector representations of image, as numpy array.
        """
        histogram = None

        if descriptors is not None:
            histogram = np.zeros(self.model.cluster_centers_.shape[0])
            for desc in descriptors:
                # find the cluster each descriptor is close to
                cluster_idx = self.model.predict([desc.astype(float)])
                histogram[cluster_idx] += 1

        return histogram
