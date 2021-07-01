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
log = logging.getLogger(__name__)


def load_data(src):
    channels_src = 'channels.txt'
    with open(os.path.join(src, channels_src)) as f:
        channels = f.read().splitlines()

    mat = np.loadtxt(os.path.join(src, 'mat.txt')).astype(np.float32)
    mat = (mat / np.linalg.norm(mat, ord=1, axis=0))
    return mat, channels


class Loss(nn.Module):
    def __init__(self, A, y):
        super(Loss, self).__init__()

        self.A = A
        self.y = y
        # self.ref_cov = ref_cov

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

    def forward(self, x, y):
        mse = F.mse_loss(y, F.conv2d(x, self.A))
        reg = torch.abs(x).mean()
        con = self.ContinuityLoss(x)
        # cov = F.mse_loss(self.cov(x), self.ref_cov)
        return mse, reg, con  # , cov


@hydra.main(config_path='conf/opt', config_name='config')
def train(cfg):
    task = Task.init(project_name="DIP optimization")
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
    log.info('x dim: {}, y dim: {}'.format(mat.shape[1], mat.shape[0]))
    log.debug(channels)

    X = xr.load_dataarray(cfg.dataset.src).sel(fovs=cfg.dataset.fovs, channels=channels).values.astype(np.float32)

    if X.ndim != 4:
        X = X.reshape((1, *X.shape))
    X = X[:,:,768:1280,768:1280]
    A = torch.tensor(mat, requires_grad=False).unsqueeze(-1).unsqueeze(-1).float().to(device)
    X = torch.tensor(X.astype(np.float32), requires_grad=False).to(device)
    # x_binary = (x > 0).float()
    Y = F.conv2d(X, A).to(device)

    model = UNet(in_channels=8, out_channels=25)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr)

    loss_fn = Loss(A, Y)
    if cfg.training.resume:
        log.info("Resume checkpoint from: {}:".format(cfg.training.resume))
        resume_path = utils.to_absolute_path(cfg.training.resume)
        checkpoint = torch.load(resume_path)  # , map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    for i in range(global_step + 1, cfg.training.epochs + 1):

        a = np.random.randint(X.shape[2]-512)
        b = np.random.randint(X.shape[3]-512)
        x = X[:, :, a:a+512, b:b+512]
        y = Y[:, :, a:a+512, b:b+512]
        # k = np.random.randint(1, 5)

        outputx = []
        outputy = []

        for k in range(4):
            rotx = torch.rot90(x, k, [2, 3])
            roty = torch.rot90(y, k, [2, 3])
            outputx.append(rotx)
            outputx.append(torch.flip(rotx, [2]))
            outputy.append(roty)
            outputy.append(torch.flip(roty, [2]))

        y = torch.stack(outputy).squeeze(1)
        x = torch.stack(outputx).squeeze(1)
        # x = torch.rot90(X, k, [2, 3])
        # y = torch.rot90(Y, k, [2, 3])
        # flip = np.random.choice([True, False])
        # if flip:
        #    x = torch.flip(x, [2])
        #    y = torch.flip(y, [2])
        x_binary = (x > 0).float()

        optimizer.zero_grad()
        _, x_hat = model(y)
        ls, reg, tv = loss_fn(x_hat,y)
        loss = ls + cfg.training.reg * reg + cfg.training.con * tv
        loss.backward()

        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_value_(x_hat, cfg.training.grad_clip)

        optimizer.step()

        e = F.mse_loss(x, x_hat).item()
        x_hat_binary = (x_hat > 0).float()
        f1 = 2 * (x_binary * x_hat_binary).sum() / (x_binary.sum() + x_hat_binary.sum())

        if i % cfg.logging.checkpoint_interval == 0:
            checkpoint_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": i,
            }

            torch.save(checkpoint_state, os.path.join(res_dir, 'cp.pt'))
            torch.save(x_hat,  os.path.join(res_dir, 'x_hat.pt'))
            log.info('checkpoint saved!')

        log.info(
            'Iter {} total: {:.3f} least_squares: {:.3f} | reg: {:.3f} con: {:.3f} | recon_mse: {:.3f} | f1: {:.3f}'.format(
                i, loss.item(), ls.item(), reg.item(), tv.item(), e, f1.item())
        )

        writer.add_scalar('Loss/total', loss.item(), i)
        writer.add_scalar('Loss/least_squares', ls.item(), i)
        writer.add_scalar('Loss/L1', reg.item(), i)
        writer.add_scalar('Loss/smoothness', tv.item(), i)
        writer.add_scalar('Evaluation/recovery_mse', e, i)
        writer.add_scalar('Evaluation/f1', f1.item(), i)


if __name__ == "__main__":
    train()
