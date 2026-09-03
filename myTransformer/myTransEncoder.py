import torch
import torch.nn as nn

from myMultiHeadAttn import myMultiHeadAttn
from myFFN import myFeedForwardNetwork


class myTransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, to be used with TransformerEncoder.
    """
    def __init__(self,
                 input_dim: int,
                 num_heads: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        """
        Construct a TransformerEncoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads.
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = myMultiHeadAttn(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.ffn = myFeedForwardNetwork(embed_dim=input_dim, ffn_dim=dim_feedforward, dropout=dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)


    def forward(self,
                src: torch.Tensor,
                pos: torch.Tensor,
                src_mask=None,
                ) -> torch.Tensor:
        """
        Pass the inputs (and mask) through the encoder layer.

        Inputs:
        - src: the sequence to the encoder layer, of shape (B, N, D)
        - pos: positional encoding of shape (B, N, D)
        - src_mask: the parts of the source sequence to mask, of shape (N, N)

        Returns:
        - out: the Transformer features, of shape (B, N, D)
        """
        # self-attention
        shortcut = src
        src_norm = self.norm_self(src)
        attn_out = self.self_attn(query = src_norm + pos,
                                  key = src_norm + pos,
                                  value = src_norm,
                                  attn_mask=src_mask,)
        src = shortcut + self.dropout_self(attn_out)
        
        
        # ffn
        shortcut = src
        src_norm = self.norm_ffn(src)
        ffn_out = self.ffn(src_norm)
        src = shortcut + self.dropout_ffn(ffn_out)
        
        return src