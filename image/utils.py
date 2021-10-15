import cv2 as cv


def load_image(path):
    """ Loads image from provided path.

    Args:
        path:
            image path, as string.

    Returns:
        image, as numpy array.
    """
    image = cv.imread(path)

    return image