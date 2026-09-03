import torch
import torch.nn as nn

from myMultiHeadAttn import myMultiHeadAttn, myMultiHeadSelfAttn
from myFFN import myFeedForwardNetwork


class myTransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, to be used with TransformerDecoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerDecoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = myMultiHeadAttn(input_dim, num_heads, dropout)
        self.cross_attn = myMultiHeadAttn(input_dim, num_heads, dropout)
        self.ffn = myFeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_cross = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)


    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                query_pos: torch.Tensor,
                memory_pos: torch.Tensor,
                ) -> torch.Tensor:
        """
        Pass the inputs through the decoder layer.

        Inputs:
        - tgt: the sequence to the decoder layer, of shape (B, Q, D)
        - memory: the sequence from the last layer of the encoder, of shape (B, N, D)
        - query_pos: shape (B, Q, D)
        - memory_pos: shape (B, N, D)

        Returns:
        - out: the Transformer features, of shape (B, Q, D)
        """

        # decoder self-attention
        shortcut = tgt
        tgt_norm = self.norm_self(tgt)
        attn_out = self.self_attn(query=tgt_norm + query_pos,
                                  key=tgt_norm + query_pos,
                                  value=tgt_norm)
        tgt = shortcut + self.dropout_self(attn_out)

        # encoder-decoder cross-attention
        shortcut = tgt
        tgt_norm = self.norm_cross(tgt)
        cross_out = self.cross_attn(query=tgt_norm + query_pos,
                                    key=memory + memory_pos,
                                    value=memory)
        tgt = shortcut + self.dropout_cross(cross_out)

        # FFN
        shortcut = tgt
        tgt_norm = self.norm_ffn(tgt)
        ffn_out = self.ffn(tgt_norm)
        tgt = shortcut + self.dropout_ffn(ffn_out)

        return tgt