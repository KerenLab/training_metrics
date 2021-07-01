from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter
import argparse
import logging
import os
from pathlib import Path
import numpy as np
import xarray as xr
from torch import optim
import logging
import hydra
from hydra import utils
from omegaconf import OmegaConf

from pathlib import Path
# from utils import load_data
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from clearml import Task
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


def load_data(src):
    channels_src = 'channels.txt'
    with open(os.path.join(src, channels_src)) as f:
        channels = f.read().splitlines()

    mat = np.loadtxt(os.path.join(src, 'mat.txt')).astype(np.float32)
    mat = (mat / np.linalg.norm(mat, ord=1, axis=0))
    return mat, channels


class CancerDataset(Dataset):
    def __init__(self, data, batch_size, random_crop, initial_transform=None):
        super().__init__()
        if initial_transform:
            self.data = initial_transform(data)
        else:
            self.data = data

        self.batch_size = batch_size
        self.random_crop = random_crop
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # T.RandomRotation(180),
            # T.RandomPerspective(),
            T.RandomCrop(size=random_crop),
            # T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # T.RandomErasing(p=0.1, scale=(0.02, 0.3), ratio=(0.01, 1), value=0, inplace=False)
        ])

    def __len__(self):
        return len(self.data)

    def fetch(self):
        batch = []
        for _ in range(self.batch_size):
            fov = np.random.randint(self.data.shape[0])
            batch.append(self.transform(self.data[fov]))
        return torch.stack(batch)


class Loss(nn.Module):
    def __init__(self, A, ref_corr):
        super(Loss, self).__init__()

        self.A = A
        self.ref_corr = ref_corr

    @staticmethod
    def ContinuityLoss(x):
        """
        x: [B, C, H, W]
        """

        dif_left = F.l1_loss(x[:, :, 1:, :], x[:, :, :-1, :])
        dif_up = F.l1_loss(x[:, :, :, 1:], x[:, :, :, :-1])
        return dif_left + dif_up

    @staticmethod
    def cov(x):
        """
        x: [B, C, H, W]
        res: [B, C, C]
        """

        # [B, C, H, W] -> [B, C, H * W]
        x = x.view((x.size(0), x.size(1), -1))

        # estimated covariance
        x = x - x.mean(dim=-1, keepdim=True)
        factor = 1 / (x.shape[-1] - 1)
        est_cov = factor * (x @ x.transpose(-1, -2))

        return est_cov

    @staticmethod
    def corr(x):
        """
        x: [B, C, H, W]
        res: [ C, C]
        """

        # [B, C, H, W] -> [C, B * H * W]
        x = x.view((x.size(1), -1))

        # estimated correlation
        x = x - x.mean(dim=-1, keepdim=True)

        factor = 1 / (x.shape[-1] - 1)

        corr = factor * (x @ x.transpose(-1, -2))
        diagonal = torch.diagonal(corr)
        stddev = torch.sqrt(diagonal)

        corr /= (stddev[:, None] + 1e-6)
        corr /= (stddev[None, :] + 1e-6)
        corr = torch.clip(corr, -1, 1)

        return corr

    def forward(self, x, y):
        mse = F.mse_loss(y, F.conv2d(x, self.A))
        reg = torch.abs(x).mean()
        con = self.ContinuityLoss(x)
        corr = F.mse_loss(self.corr(x), self.ref_corr)
        return mse, reg, con, corr


@hydra.main(config_path='conf/opt', config_name='config')
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'{device} detected.')
    log.info(OmegaConf.to_yaml(cfg))
    os.chdir(hydra.utils.get_original_cwd())
    writer = SummaryWriter(log_dir=f'{cfg.logging.checkpoint_dir}',
                           comment=f'LR_{cfg.training.optimizer.lr}_reg_{cfg.training.reg}_con_{cfg.training.con}')

    checkpoint_dir = Path(f'{cfg.logging.checkpoint_dir}')
    config_path = checkpoint_dir / 'config.yaml'
    res_dir = checkpoint_dir / 'x_hat'
    res_dir.mkdir(exist_ok=True)

    # this part is to make sure /n are replaced with newlines
    def repr_str(dumper: RoundTripRepresenter, data: str):
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    yaml = YAML()
    yaml.representer.add_representer(str, repr_str)

    with open(config_path, 'w') as f:
        yaml.dump(OmegaConf.to_yaml(cfg), f)

    # set seed
    if cfg.exp.seed:
        torch.manual_seed(cfg.exp.seed)
        np.random.seed(cfg.exp.seed)

    mat, channels = load_data(cfg.dataset.panel)
    ref_corr = torch.tensor(np.load(r'/home/labs/leeat/vovam/scripts/correlation_mat/corr_all.npy'), requires_grad=False).float().to(device)
    log.info('x dim: {}, y dim: {}'.format(mat.shape[1], mat.shape[0]))
    log.debug(channels)

    if cfg.dataset.pretraining:
        X = xr.load_dataarray(cfg.dataset.src).sel(channels=channels).values.astype(np.float32)
    else:
        X = xr.load_dataarray(cfg.dataset.src).sel(fovs=cfg.dataset.fovs, channels=channels).values.astype(np.float32)

    # X = xr.load_dataarray(cfg.dataset.src).sel(fovs=cfg.dataset.fovs, channels=channels).values.astype(np.float32)

    if X.ndim != 4:
        X = X.reshape((1, *X.shape))
    A = torch.tensor(mat, requires_grad=False).unsqueeze(-1).unsqueeze(-1).float().to(device)
    X = torch.tensor(X.astype(np.float32), requires_grad=False).to(device)

    # check if initial crop is needed
    if cfg.training.center_crop:
        take_crop = T.Compose([T.CenterCrop(cfg.training.center_crop)])
        ds = CancerDataset(X, cfg.training.batch_size, cfg.training.random_crop, take_crop)
    else:
        ds = CancerDataset(X, cfg.training.batch_size, cfg.training.random_crop)

    model = UNet(in_channels=10, out_channels=25)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr)

    loss_fn = Loss(A, ref_corr)
    if cfg.training.resume:
        log.info("Resume checkpoint from: {}:".format(cfg.training.resume))
        resume_path = utils.to_absolute_path(cfg.training.resume)
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    for i in range(global_step + 1, cfg.training.epochs + 1):
        X = ds.fetch()  # .to(device)
        Y = F.conv2d(X, A).to(device)

        x_binary = (X > 0).float()

        optimizer.zero_grad()
        _, x_hat = model(Y)
        ls, reg, tv, corr = loss_fn(x_hat, Y)
        loss = ls + cfg.training.reg * reg + cfg.training.con * tv + corr
        # loss = corr
        loss.backward()

        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_value_(x_hat, cfg.training.grad_clip)

        optimizer.step()

        e = F.mse_loss(X, x_hat).item()
        x_hat_binary = (x_hat > 0).float()
        f1 = 2 * (x_binary * x_hat_binary).sum() / (x_binary.sum() + x_hat_binary.sum())

        if i % cfg.logging.checkpoint_interval == 0:
            checkpoint_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": i,
            }

            # if cfg.dataset.pretraining:
            #     torch.save(checkpoint_state, os.path.join(res_dir, 'pretrained_cp.pt'))
            # else:
            #     torch.save(checkpoint_state, os.path.join(res_dir, 'cp.pt'))
            # torch.save(x_hat, os.path.join(res_dir, 'x_hat.pt'))
            torch.save(checkpoint_state, os.path.join(res_dir, 'cp.pt'))
            log.info('checkpoint saved!')

        log.info(
            'Iter {} total: {:.3f} least_squares: {:.3f} | reg: {:.3f} con: {:.3f} | recon_mse: {:.3f} | f1: {:.3f}| corr:{:.3f}'.format(
                i, loss.item(), ls.item(), reg.item(), tv.item(), e, f1.item(), corr.item())
        )

        writer.add_scalar('Loss/total', loss.item(), i)
        writer.add_scalar('Loss/least_squares', ls.item(), i)
        writer.add_scalar('Loss/L1', reg.item(), i)
        writer.add_scalar('Loss/smoothness', tv.item(), i)
        writer.add_scalar('Evaluation/recovery_mse', e, i)
        writer.add_scalar('Evaluation/f1', f1.item(), i)


if __name__ == "__main__":
    train()
