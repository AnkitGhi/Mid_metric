# mid_extension.py
import torch
from torch import Tensor
from metrics.mid import MutualInformationDivergence, log_det, robust_inv

class MIDWithBatchPMI(MutualInformationDivergence):
    """
    Extends MID to add `batch_pmi(...)` so you can score arbitrary (x,y) batches.
    """
    def __init__(self, feature: int = 512, limit: int = 30000, eps: float = 1e-6, **kwargs):
        super().__init__(feature=feature, limit=limit, **kwargs)
        self._reference_ready = False
        self.eps = eps

    def _prepare_reference(self):
        # concatenate reference features
        x = torch.cat(self.x_feat, dim=0)
        y = torch.cat(self.y_feat, dim=0)
        z = torch.cat([x, y], dim=1)

        # means
        self.mu_x = x.mean(dim=0)
        self.mu_y = y.mean(dim=0)
        self.mu_z = z.mean(dim=0)

        # covariances
        N = x.shape[0]
        cov_x = (x - self.mu_x).T @ (x - self.mu_x) / (N - 1)
        cov_y = (y - self.mu_y).T @ (y - self.mu_y) / (N - 1)
        cov_z = (z - self.mu_z).T @ (z - self.mu_z) / (N - 1)

        # constant MI term
        self.MI_const = 0.5 * (log_det(cov_x) + log_det(cov_y) - log_det(cov_z))

        # inverses for Mahalanobis
        self.Sig_x_inv = robust_inv(cov_x, eps=self.eps)
        self.Sig_y_inv = robust_inv(cov_y, eps=self.eps)
        self.Sig_z_inv = robust_inv(cov_z, eps=self.eps)

        self._reference_ready = True

    def batch_pmi(self, x_new: Tensor, y_new: Tensor) -> Tensor:
        """
        Compute PMI per sample for a batch of new (x_new, y_new) features.
        """
        if not self._reference_ready:
            self._prepare_reference()

        dx = x_new - self.mu_x
        dy = y_new - self.mu_y
        dz = torch.cat([dx, dy], dim=1)

        D2_x = (dx @ self.Sig_x_inv * dx).sum(dim=1)
        D2_y = (dy @ self.Sig_y_inv * dy).sum(dim=1)
        D2_z = (dz @ self.Sig_z_inv * dz).sum(dim=1)

        return self.MI_const + 0.5 * (D2_x + D2_y - D2_z)
