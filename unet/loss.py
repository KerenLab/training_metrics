import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 0.0) -> None:
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.eps: float = 1e-8

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(input)
        loss = - self.alpha * torch.pow((1. - probs + self.eps), self.gamma) * target * torch.log(probs + self.eps) \
               - (1 - self.alpha) * torch.pow(probs + self.eps, self.gamma) * (1. - target) * torch.log(
            1. - probs + self.eps)
        return loss.mean()


class LossWrapper(nn.Module):
    def __init__(self, criterion, k=1.0, weights=None, use_positive_weights=False):
        super(LossWrapper, self).__init__()
        self.criterion = criterion
        self.k = k
        self.weights = weights.view(1, -1, 1, 1)
        self.use_positive_weights = use_positive_weights

    def forward(self, input, target):
        e = self.criterion(input, target)

        if self.use_positive_weights:
            # amplify positive pixels
            e += e * (self.weights - 1) * target.ge(0)

        loss, _ = torch.topk(e.flatten(), k=int(self.k * e.numel()))
        return loss.mean()


class CovarianceLoss(nn.Module):
    def __init__(self):
        super(CovarianceLoss, self).__init__()

    def cov(self, x):
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

    def forward(self, input, target):
        est_cov = self.cov(input)
        ref_cov = self.cov(target)
        return F.mse_loss(est_cov, ref_cov)
