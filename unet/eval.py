import torch
import torch.nn.functional as F
from loss import *


def eval_net(net, loader, class_fn, device):
    net.eval()

    X, Y = loader.get()
    X = X.to(device=device, dtype=torch.float32)
    Y = Y.to(device=device, dtype=torch.float32)

    total_recon = total_f1 = total_bce = 0
    for x, y in zip(X, Y):
        with torch.no_grad():
            logits, x_hat = net(y)
        # total_recon += F.mse_loss(x_hat, x)
        target = (x > 0).float()
        pred = (torch.sigmoid(logits) > 0.5).float()
        total_f1 += 2 * ((target * pred).sum() / (target.sum() + pred.sum()))
        # total_bce += F.binary_cross_entropy_with_logits(logits, target)
        total_bce += class_fn(logits, target)

    net.train()
    return total_f1 / X.shape[0], total_bce / X.shape[0]
    # return total_recon / X.shape[0], total_f1 / X.shape[0], total_bce / X.shape[0]
