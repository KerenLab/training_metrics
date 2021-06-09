import logging
import hydra
from hydra import utils
from pathlib import Path
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import xarray as xr
from dataset import Data
from eval import eval_net
from loss import *
from utils import *
from unet.unet_model import UNet
import random
import torch

from clearml import Task

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger(__name__)


@hydra.main(config_path='conf/unet', config_name='25-8')
def train_model(cfg):
    task = Task.init(project_name="Compressed Sensing")
    log.info(cfg.exp.summary)
    log.info(f'{device} detected.')
    log.debug(OmegaConf.to_yaml(cfg))
    os.chdir(hydra.utils.get_original_cwd())
    writer = SummaryWriter(log_dir=f'{cfg.logging.checkpoint_dir}')
    checkpoint_dir = Path(f'{cfg.logging.checkpoint_dir}')
    dump_config(cfg, save_path=checkpoint_dir / 'config.yaml')

    # load dataset and sensing matrix
    mat, channels, weights = load_data(cfg.dataset.panel)
    weights = weights.to(device)

    index = cfg.training.index if cfg.training.index else [i for i in range(mat.shape[1])]

    dataset = xr.load_dataarray(cfg.dataset.src).sel(channels=channels)
    val_set = dataset.sel(fovs=cfg.training.val_fovs).values.astype(np.float32)

    ###
    X_val = torch.from_numpy(dataset.sel(fovs=['Point5']).values.astype(np.float32)).to(device)
    m = torch.from_numpy(mat).float().unsqueeze(-1).unsqueeze(-1).to(device)
    Y_val = F.conv2d(X_val, m)
    X_val = X_val[:, index, :, :]
    target_val = (X_val > 0).float().squeeze()
    target_val = target_val.view(target_val.size(0), -1)  # [C, H * W]
    ###

    if not cfg.training.train_fovs:
        train_set = dataset.drop_sel(fovs=cfg.training.val_fovs).values.astype(np.float32)
    else:
        train_set = dataset.sel(fovs=cfg.training.train_fovs).values.astype(np.float32)

    train_fetcher = Data(train_set, mat, crop_size=cfg.dataset.crop_size, batch_size=cfg.dataset.batch_size,
                         dropout=cfg.dataset.dropout, index=index)
    val_fetcher = Data(val_set, mat, crop_size=cfg.dataset.crop_size, batch_size=cfg.dataset.batch_size, train=False,
                       index=index)

    # set seed
    if cfg.exp.seed:
        random.seed(cfg.exp.seed)
        torch.manual_seed(cfg.exp.seed)
        np.random.seed(cfg.exp.seed)

    model = torch.nn.DataParallel(UNet(**cfg.model.net))
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr)

    # define loss functions
    # recon_fn = LossWrapper(nn.MSELoss(reduction='none'), k=cfg.training.top_k,
    #                        weights=weights, use_positive_weights=cfg.training.use_positive_weights)
    # class_fn = LossWrapper(nn.BCELoss(reduction='none'), k=cfg.training.top_k,
    #                        weights=weights, use_positive_weights=cfg.training.use_positive_weights)
    class_fn = BinaryFocalLossWithLogits(**cfg.training.bce)

    if cfg.training.resume:
        log.info("Resume checkpoint from: {}:".format(cfg.training.resume))
        resume_path = utils.to_absolute_path(cfg.training.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    for step in range(global_step + 1, cfg.training.n_steps + 1):

        x, y = train_fetcher.get()
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        optimizer.zero_grad()

        logits, x_hat = model(y)

        binary_target = (x > 0).float()

        pred = (torch.sigmoid(logits) > 0.5).float()
        f1 = 2 * torch.true_divide((binary_target * pred).sum(), (binary_target.sum() + pred.sum()))

        classification_loss = class_fn(logits, binary_target)
        # classification_loss = F.binary_cross_entropy_with_logits(logits, binary_target)
        # classification_loss = class_fn(torch.sigmoid(logits), binary_target)

        # y_hat = F.conv2d(x_hat, train_fetcher.m.to(device))
        # ls_error = F.mse_loss(y_hat, y)
        # recon_loss = F.mse_loss(x_hat, x)
        # recon_loss = recon_fn(x_hat, x)
        # cov_loss = cov_fn(x_hat, x)

        # loss = cfg.training.recon * recon_loss + cfg.training.cov * cov_loss + cfg.training.ls * ls_error
        # loss = cfg.training.recon * recon_loss + cfg.training.ls * ls_error + cfg.training.cl * classification_loss
        loss = classification_loss
        loss.backward()

        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.training.grad_clip)

        optimizer.step()

        # log.info('[{} / {}] | TRAIN loss: {:.2E} | mse: {:.2E} | bce: {:.2E} | f1: {:.2E} | ls: {:.2E}'.format(step,
        #                                                                                                        cfg.training.n_steps,
        #                                                                                                        loss.item(),
        #                                                                                                        recon_loss.item(),
        #                                                                                                        classification_loss.item(),
        #                                                                                                        f1.item(),
        #                                                                                                        ls_error.item()))
        log.info('[{} / {}] | TRAIN loss: {:.2E} | bce: {:.2E} | f1: {:.2E}'.format(step,
                                                                                    cfg.training.n_steps,
                                                                                    loss.item(),
                                                                                    classification_loss.item(),
                                                                                    f1.item()))

        # writer.add_scalar('TRAIN/mse', recon_loss.item(), step)
        writer.add_scalar('TRAIN_LOSS/bce', classification_loss.item(), step)
        writer.add_scalar('TRAIN_ACCURACY/f1', f1.item(), step)
        # writer.add_scalar('TRAIN/ls', ls_error.item(), step)
        # writer.add_scalar('TRAIN/loss_total', loss.item(), step)
        writer.add_scalar('Model/LR', optimizer.param_groups[0]['lr'], step)

        if step % cfg.logging.eval_interval == 0 or step == cfg.training.n_steps - 1 or step == 1:
            val_dice, val_bce = eval_net(model, val_fetcher, class_fn, device)
            log.info('[{} / {}] | VAL bce: {:.2E} | f1: {:.2E}'.format(step, cfg.training.n_steps,
                                                                       val_bce.item(),
                                                                       val_dice.item()))
            # writer.add_scalar('VAL/mse', val_mse.item(), step)
            writer.add_scalar('VAL_LOSS/bce', val_bce.item(), step)
            writer.add_scalar('VAL_ACCURACY/f1', val_dice.item(), step)

            ###
            model.eval()
            with torch.no_grad():
                logits, x_hat = model(Y_val)
            pred = (torch.sigmoid(logits) > 0).float().squeeze()  # [C, H, W]
            pred = pred.view(pred.size(0), -1)  # [C, H * W]
            f1 = 2 * ((target_val * pred).sum(dim=1) / (target_val + pred).sum(dim=1))  # [C,]
            for i in index:
                writer.add_scalar(f'Point5/{channels[i]}', f1[i].sum().item(), step)
                log.info('{} \t {:.2E}'.format(channels[i], f1[i].sum().item()))
            # for score, channel in zip(f1, channels):
            #     writer.add_scalar(f'Point5/{channel}', score.sum().item(), step)
            #     log.info('{} \t {:.2E}'.format(channel, score.sum().item()))
            model.train()
            ###

        if step % cfg.logging.checkpoint_interval == 0 or step == cfg.training.n_steps - 1 or step == 1:
            save_checkpoint(log, model, optimizer, step, checkpoint_dir)


if __name__ == "__main__":
    train_model()
