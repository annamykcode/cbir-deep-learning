from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = self.transforms(image)

        return image
