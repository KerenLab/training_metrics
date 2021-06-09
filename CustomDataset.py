class CancerDataset(Dataset):
    def __init__(self, data, random_crop, initial_transform=None):
        super().__init__()
        if initial_transform:
            self.data = initial_transform(data)
        else:
            self.data = data

        self.batch_size = batch_size
        self.random_crop = random_crop
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
