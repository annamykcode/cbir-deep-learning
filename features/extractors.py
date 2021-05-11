import cv2 as cv
from skimage.feature import hog


def features_sift(image):
    """ Extracts image keypoints and descriptors using SIFT.

    Args:
        image:
            Image, as numpy array.

    Returns:
        Keypoints and descriptors, as lists.
    """
    sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.02, sigma=0.9)

    kp, des = sift.detectAndCompute(image, None)

    return kp, des


def features_hog(image):
    """ Extracts image histogram of oriented gradients.

    Args:
        image:
            Image, as numpy array.

    Returns:
        HOG descriptor, as list.
    """
    hog_descriptor = hog(image,
                         orientations=9,
                         pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2),
                         block_norm='L2-Hys',
                         transform_sqrt=True,
                         feature_vector=True,
                         visualize=False)

    return hog_descriptor
