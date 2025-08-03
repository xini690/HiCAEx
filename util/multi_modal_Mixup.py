from timm.data import Mixup
import numpy as np
import torch
import torch.nn.functional as F



def mixup_target(target, num_classes, lam, smoothing=0.0):
    """
    生成混合标签
    """
    if target.dim() == 1:
        target_onehot = F.one_hot(target, num_classes).float()
    else:
        target_onehot = target
    mixed_target = lam * target_onehot + (1 - lam) * target_onehot.flip(0)
    if smoothing > 0:
        mixed_target = mixed_target * (1 - smoothing) + smoothing / num_classes
    return mixed_target


class MultiModalMixup(Mixup):
    def _sample_lambda(self):
        lam, _ = self._params_per_batch()
        return lam

    def mixup_data(self, x, lam, index):
        if self.cutmix_alpha > 0 and self.cutmix_minmax is None and self.mode == 'batch':
            pass
        return lam * x + (1 - lam) * x[index]

    def mixup_target_custom(self, target, lam, index):
        if target.dim() == 1:
            num_classes = self.num_classes
            target_onehot = F.one_hot(target, num_classes).float()
            mixed_target = lam * target_onehot + (1 - lam) * target_onehot[index]
        else:
            mixed_target = lam * target + (1 - lam) * target[index]
        if self.label_smoothing > 0:
            smoothing = self.label_smoothing
            num_classes = self.num_classes
            mixed_target = mixed_target * (1 - smoothing) + smoothing / num_classes
        return mixed_target

    def mixup_two_modalities(self, x1, x2, target):
        lam = self._sample_lambda()
        batch_size = x1.size(0)
        index = torch.randperm(batch_size).to(x1.device)

        x1_mixed = self.mixup_data(x1, lam, index)
        x2_mixed = self.mixup_data(x2, lam, index)
        target_mixed = self.mixup_target_custom(target, lam, index)
        return x1_mixed, x2_mixed, target_mixed
