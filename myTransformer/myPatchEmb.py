import torch
import torch.nn as nn


class myPatchEmbedding(nn.Module):
    """
    Chop up a batch of 300x300 images into 20x15 patches.

    Inputs
    - img_size: Integer representing the height/width of input image (assumes square image).
    - patch_H: Integer representing height of each patch.
    - patch_W: Integer representing width of each patch.
    - in_channels: Number of input image channels (e.g., 3 for RGB).
    - embed_dim: Dimension of the linear embedding space.

    Output: a tensor of shape (B, N, D), where N = (H / 20) * (W / 15) = 300, D = embed_dim
    """
    def __init__(self,
                 img_size: int = 300,
                 patch_H: int = 20,
                 patch_W: int = 15,
                 in_channels: int = 3,
                 embed_dim: int = 128):
        super().__init__()

        self.img_size = img_size
        self.patch_H = patch_H
        self.patch_W = patch_W
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        assert img_size % patch_H == 0, "Image dimensions must be divisible by the patch size."
        assert img_size % patch_W == 0, "Image dimensions must be divisible by the patch size."

        self.num_tokens = (img_size // patch_H) * (img_size // patch_W)
        self.patch_dim = patch_H * patch_W * in_channels

        self.proj = nn.Linear(in_features = self.patch_dim, out_features = embed_dim)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for myPatchEmb.

        Input
        x: Tensor of shape (B, C, H, W)

        Output
        out: Tensor of shape (B, N, D), where N = self.num_tokens and D = self.embed_dim
        """

        PH = self.patch_H
        PW = self.patch_W
        B = x.shape[0]
        H = self.img_size
        W = self.img_size
        C = self.in_channels

        assert x.shape[1] == C, f"Mismatch between in_channels ({self.in_channels}) and input tensor channels ({x.shape[1]})."

        out = torch.reshape(x, (B, C, H // PH, PH, W // PW, PW))
        out = torch.permute(out, (0, 2, 4, 1, 3, 5))
        out = torch.reshape(out, (B, self.num_tokens, self.patch_dim))
        out = self.proj(out)

        return out