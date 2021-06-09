import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import torch
from PIL import Image
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Data:
    def __init__(self, X, m, crop_size, batch_size, index, dropout=0, train=True):
        self.X = torch.from_numpy(X).float()  # [N x C x H x W]
        # if train:
        #     self.X.clamp_max_(10)
        # self.X.clamp_max_(10)

        self.index = index
        self.m = torch.from_numpy(m).float().unsqueeze(-1).unsqueeze(-1)
        self.m = self.m.to(device)

        self.dropout = dropout
        self.train = train
        self.batch_size = batch_size
        self.crop_size = crop_size

        self.data_transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor()
        ])

    def get(self):
        x, y = self.get_train() if self.train else self.get_val()
        if self.train:
            x = x[:, self.index, :, :]
        else:
            x = x[:, :, self.index, :, :]
        return x, y

    def get_val(self):
        B = 1
        H = W = self.crop_size
        C_X = self.m.shape[1]
        C_Y = self.m.shape[0]

        x = self.X.to(device)

        batch_X = x.unfold(0, B, B).unfold(2, H, H).unfold(3, W, W).permute(0, 2, 3, 4, 1, 5, 6).contiguous().view(
            -1, B, C_X,
            H, W)

        Y = F.conv2d(x, self.m)
        batch_Y = Y.unfold(0, B, B).unfold(2, H, H).unfold(3, W, W).permute(0, 2, 3, 4, 1, 5, 6).contiguous().view(
            -1, B, C_Y,
            H, W)

        return batch_X, batch_Y

    def get_train(self):
        batch = []
        for _ in range(self.batch_size):
            x = []
            # select a fov
            n = np.random.randint(self.X.shape[0])
            img = self.X[n]
            # set a seed so the same transforms are applied to each channel
            seed = np.random.randint(2147483647)
            for ch in img:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                x.append(self.data_transform(Image.fromarray(ch.cpu().numpy())))

            # this is the multichannel transformed image
            img_tfm = torch.cat(x)

            # TODO add gaussian blur
            # transforms.

            batch.append(img_tfm)
        x = torch.stack(batch)
        x = x.to(device)

        # channels dropout
        F.dropout2d(x, p=self.dropout, inplace=True)

        y = F.conv2d(x, self.m)
        return x, y
