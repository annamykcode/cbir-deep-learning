import cv2 as cv
from skimage.feature import hog


def hog_features(image):
    """ Extracts histogram of oriented gradients..

    Args:
        image:
            image in grayscale format, as numpy array.

    Returns:
        HOG descriptors, as numpy array.
    """
    hog_descriptors = hog(image,
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          block_norm='L2-Hys',
                          transform_sqrt=True,
                          feature_vector=True,
                          visualize=False)

    return hog_descriptors


def sift_features(image):
    """ Extracts image SIFT keypoints and descriptors..

    Args:
        image:
            image in grayscale format, as numpy array.

    Returns:
        SIFT keypoints, as list.
        SIFT descriptors, as numpy array.
    """
    sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.02, sigma=0.9)

    kp, des = sift.detectAndCompute(image, None)

    return kp, des
