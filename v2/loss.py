import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        h_count = self._tensor_size(x[:, :, 1:, :])
        w_count = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return self.weight * (h_tv / max(1, h_count) + w_tv / max(1, w_count)) / max(1, batch_size)

    @staticmethod
    def _tensor_size(t: torch.Tensor):
        return int(t.size(1) * t.size(2) * t.size(3))


class TVLossSpectral(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        c_count = self._tensor_size(x[:, 1:, :, :])
        c_tv = torch.pow(x[:, 1:, :, :] - x[:, :-1, :, :], 2).sum()
        return self.weight * 2.0 * (c_tv / max(1, c_count)) / max(1, batch_size)

    @staticmethod
    def _tensor_size(t: torch.Tensor):
        return int(t.size(1) * t.size(2) * t.size(3))


class HybridLoss(nn.Module):
    def __init__(
        self,
        spatial_tv: bool = False,
        spectral_tv: bool = False,
        spatial_tv_weight: float = 0.0,
        spectral_tv_weight: float = 1e-4,
    ):
        super().__init__()
        self.use_spatial_tv = bool(spatial_tv)
        self.use_spectral_tv = bool(spectral_tv)
        self.fidelity = nn.L1Loss()
        self.spatial = TVLoss(weight=spatial_tv_weight)
        self.spectral = TVLossSpectral(weight=spectral_tv_weight)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        total = self.fidelity(pred, gt)
        if self.use_spatial_tv:
            total = total + self.spatial(pred)
        if self.use_spectral_tv:
            total = total + self.spectral(pred)
        return total


def build_loss():
    return HybridLoss(
        spatial_tv=False,
        spectral_tv=True,
        spatial_tv_weight=0.0,
        spectral_tv_weight=1e-4,
    )
