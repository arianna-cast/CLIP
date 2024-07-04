
import os
from PIL import Image
from torch.utils.data import Dataset

class MyImagenetR(Dataset):
    N_CLASSES = 200

    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        files = [] # leggo i file da BigEarth
        for d in os.listdir('/work/tesi_acastagni/EuroSAT/'):
            files.append(os.listdir('/work/tesi_acastagni/EuroSAT/' + d))

        self.data = ... # list of paths to images
        self.targets = ... # list of targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target