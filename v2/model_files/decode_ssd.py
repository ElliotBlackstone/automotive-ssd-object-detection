import torch
from typing import Tuple


def decode_ssd(loc: torch.Tensor,
                   priors: torch.Tensor,
                   variances: Tuple[float, float],
                   ) -> torch.Tensor:
        """
        Inputs
        loc: Tensor of shape [P, 4] containing (tx, ty, tw, th)
        priors: Priors of shape [P, 4] containing (cx_a, cy_a, w_a, h_a), normalized [0,1]
        variances: Tuple containing two positive floats, default (0.1, 0.2)

        Outputs
        boxes_cxcywh normalized to [0,1], shape [P, 4]
        (P is the number of priors, 8732)
        """
        v_c, v_s = variances
        # centers
        cx = loc[:, 0] * v_c * priors[:, 2] + priors[:, 0]
        cy = loc[:, 1] * v_c * priors[:, 3] + priors[:, 1]
        # sizes
        w  = priors[:, 2] * torch.exp(loc[:, 2] * v_s)
        h  = priors[:, 3] * torch.exp(loc[:, 3] * v_s)

        boxes = torch.stack([cx, cy, w, h], dim=1)
        return boxes