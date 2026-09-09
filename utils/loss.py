import torch
import torch.nn as nn
import torch.nn.functional as F


class DFL_loss(nn.Module):
    def __init__(self,reg_max: int=16)->None:
        super().__init__()
        self.reg_max=reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1-0.01) 
        tl=target.long()
        tr=tl+1 
        wl=tr-target
        wr=1-wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)