import torch
import torch.nn as nn
import copy



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer,
                 num_layers: int,
                 embed_dim: int):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                query_pos: torch.Tensor,
                memory_pos: torch.Tensor,
                ) -> torch.Tensor:
        intermediate_outputs = []
        output = tgt

        for mod in self.layers:
            output = mod(tgt=output,
                         memory=memory,
                         query_pos=query_pos,
                         memory_pos=memory_pos)
            intermediate_outputs.append(self.norm(output))

        # Shape:
        # list of L tensors of shape (B, Q, D)
        #            ->
        # tensor of shape (L, B, Q, D)
        return torch.stack(intermediate_outputs, dim=0)


class TransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer,
                 num_layers: int,
                 embed_dim: int):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,
                src: torch.Tensor,
                pos: torch.Tensor,
                src_mask = None,
                ) -> torch.Tensor:
        output = src

        for mod in self.layers:
            output = mod(src=output,
                         pos=pos,
                         src_mask=src_mask)

        output = self.norm(output)

        return output