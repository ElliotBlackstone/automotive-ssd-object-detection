import torch
import numpy as np

def create_default_boxes(s_min: float = 0.2, s_max: float = 0.9, clip: bool = True) -> torch.Tensor:
        """
        Create default boxes.
        Default settings create boxes as per SSD paper https://arxiv.org/abs/1512.02325

        Inputs:
        s_min - float between 0 and 1, default 0.2
        s_max - float between s_min and 1, default 0.9
        clip - bool, default True

        Output:
        Tensor of shape [8732, 4] where boxes are normalized and are of the form (cx, cy, w, h)
        """
        feature_map_sizes = [(38, 38), (19, 19), (10,10), (5,5), (3,3), (1,1)]
        aspect_ratios_per_level = [[2], [2,3], [2,3], [2,3], [2], [2]]

        # Example: aspect_ratios_per_level = 2
        # This will produce 4 default boxes (per center).
        # Given scales s, sp, create squares of with side length s and sp.
        # Create rectangles with scale s and aspect ratio 2, 1/2.
        # A total of 4 boxes are created (per center).

        L = len(feature_map_sizes)
        
        # scales s_0..s_{L-1}, and s_L = 1.0 for the s'_l computation
        s = [s_min + (s_max - s_min) * (l / (L - 1)) for l in range(L)]
        s.append(1.0)  # s_L

        priors = []
        for l, (H, W) in enumerate(feature_map_sizes):
            s_l  = s[l]
            s_lp = np.sqrt(s[l] * s[l+1])  # extra square

            # per-location widths/heights to emit, in (w,h), normalized
            whs = [(s_l, s_l), (s_lp, s_lp)]
            for a in aspect_ratios_per_level[l]:
                sr = np.sqrt(a)
                whs.append((s_l * sr, s_l / sr))
                whs.append((s_l / sr, s_l * sr))

            # tile over centers
            for i in range(H):
                cy = (i + 0.5) / H
                for j in range(W):
                    cx = (j + 0.5) / W
                    for (w, h) in whs:
                        priors.append([cx, cy, w, h])

        priors = torch.tensor(priors, dtype=torch.float32)
        if clip:
            # keep centers in [0,1], clip sizes to [eps,1]
            eps = 1e-6
            priors[:, 0:2].clamp_(0.0, 1.0)
            priors[:, 2:4].clamp_(eps, 1.0)
        return priors  # shape [num_priors, 4], (cx,cy,w,h) normalized