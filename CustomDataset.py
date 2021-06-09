import torchvision.transforms as T
from torch.utils.data import Dataset


class CancerDataset(Dataset):
    """Custom Dataset to work with MIBI pictures and databases"""
    def __init__(self, data, random_crop, initial_transform=None):
        """
        Args:
            data (tensor): Tensor with your MIBI pictures
            random_crop (int): Dimensions of the crop to be taken, use from config (e.g. - cfg.exp.random_crop)
            initial_transform (function): Torchvision CenterCrop func if you want to load smaller pictures as dataset
        """
        super().__init__()
        if initial_transform:
            self.data = initial_transform(data)
        else:
            self.data = data

        self.random_crop = random_crop
        # Here i assembled list of different augmentations change to your needs the parameters/augmentations
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(30),
            T.RandomPerspective(),
            T.RandomCrop(size=random_crop),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.01, 1), value=0, inplace=False)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])
