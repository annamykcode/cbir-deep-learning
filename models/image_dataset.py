from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """ Represents a Pytorch image dataset.

    Attributes:
        images:
            RGB images, as list of PIL images.
        transforms:
            image transformations, as Pytorch object
    """

    def __init__(self, images, transforms):
        """ Initializes ImageDataset object.

        Args:
            images:
                RGB images, as list of PIL images.
            transforms:
                image transformations, as Pytorch object
        """
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = self.transforms(image)

        return image
