import numpy as np
from sklearn.cluster import MiniBatchKMeans


def get_visual_words(descriptors, n_clusters=8, n_init=10, max_iter=300):
    """ Finds the vocabulary of visual words.

    To find the vocabulary of visual words, k-means is used. The centroids found by k-means
    form the vocabulary.

    Args:
        descriptors:
            descriptors, as a list of numpy arrays.
        n_clusters:
            The number of clusters to form as well as the number of centroids to generate, as int (default=8).
        n_init:
            Number of time the k-means algorithm will be run with different centroid seeds, as int (default=10).
        max_iter:
            Maximum number of iterations of the k-means algorithm for a single run, as int (default=300)

    Returns:
        Fitted k-means, holding the vocabulary of visual words .
    """
    descriptors_raw = []
    for des in descriptors:
        if des is not None:
            descriptors_raw.extend(des)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=64, n_init=n_init, max_iter=max_iter)

    visual_words = kmeans.fit(descriptors_raw)

    return visual_words


def get_vector_representation(model, descriptors):
    """ Computes the vector representation of an image.

    The representation is based on the image descriptors and a predefined BoVW model. Specifically,
    it's a histogram of the frequencies of visual words from the vocabulary of the BoVW.

    Args:
        model:
            precomputed k-means model with the vocabulary of visual words.
        descriptors:
            descriptors, as numpy array.

    Returns:
        Vector representations of image, as numpy array.
    """
    histogram = None

    if descriptors is not None:
        histogram = np.zeros(model.cluster_centers_.shape[0])
        for desc in descriptors:
            # find the cluster each descriptor is close to
            cluster_idx = model.predict([desc.astype(float)])
            histogram[cluster_idx] += 1

    return histogram
