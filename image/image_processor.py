import cv2 as cv


def to_grayscale(image):
    """ Converts an RGB image to grayscale.

    Args:
        image:
            RGB image, as numpy array.

    Returns:
        grayscale image, as numpy array.
    """
    grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    return grayscale
