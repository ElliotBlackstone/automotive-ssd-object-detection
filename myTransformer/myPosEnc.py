import torch
import torch.nn as nn
import math


class myPositionEncoding(nn.Module):
    """
    Adds positional encoding to a batch of images after patch embedding has been performed.

    Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        assert embed_dim % 2 == 0, "Embed_dim should be divisible by 2."

        self.dropout = nn.Dropout(p = dropout)

        pe = torch.zeros(1, max_len, embed_dim)
        for i in range(max_len):
          for j in range(embed_dim):
            if j % 2 == 0:
              pe[0, i, j] = torch.sin(torch.tensor(i * (10000 ** (-j/embed_dim))))
            else:
              pe[0, i, j] = torch.cos(torch.tensor(i * (10000 ** (-(j-1)/embed_dim))))


        self.register_buffer('pe', pe)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        
        output = x + self.pe[:,:x.shape[1], :]
        output = self.dropout(output)
        return output





class PositionEncoding2D(nn.Module):
    """
    Fixed 2D sinusoidal positional encoding for image patch tokens.

    Each patch position is identified by:
        - its row in the patch grid
        - its column in the patch grid

    The first half of the embedding encodes the row.
    The second half encodes the column.

    Parameters
    ----------
    embed_dim:
        Dimension of each patch embedding. Must be divisible by 4.

    grid_height:
        Number of patch rows.

    grid_width:
        Number of patch columns.

    dropout:
        Dropout probability applied after adding positional encodings.

    base:
        Base used to generate the range of sinusoidal frequencies.

    Expected input shape
    --------------------
    x: (B, N, D)

    where:
        B = batch size
        N = grid_height * grid_width
        D = embed_dim

    Output shape
    ------------
    (B, N, D)
    """

    def __init__(
        self,
        embed_dim: int,
        grid_height: int,
        grid_width: int,
        dropout: float = 0.1,
        base: float = 10000.0,
    ):
        super().__init__()

        if embed_dim % 4 != 0:
            raise ValueError(
                "embed_dim must be divisible by 4 because the embedding is "
                "split into row and column components, each containing "
                "sine/cosine pairs."
            )

        if grid_height <= 0 or grid_width <= 0:
            raise ValueError("grid_height and grid_width must be positive.")

        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        if base <= 0:
            raise ValueError("base must be positive.")

        self.embed_dim = embed_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_tokens = grid_height * grid_width

        self.dropout = nn.Dropout(p=dropout)

        pe = self._create_positional_encoding(
            embed_dim=embed_dim,
            grid_height=grid_height,
            grid_width=grid_width,
            base=base,
        )

        self.register_buffer("pe", pe)

    @staticmethod
    def _create_positional_encoding(
        embed_dim: int,
        grid_height: int,
        grid_width: int,
        base: float,
    ) -> torch.Tensor:
        """
        Construct a tensor of shape:

            (1, grid_height * grid_width, embed_dim)
        """

        axis_dim = embed_dim // 2

        # Frequencies for one spatial axis.
        #
        # Shape: (axis_dim / 2,)
        frequency_indices = torch.arange(
            0,
            axis_dim,
            2,
            dtype=torch.float32,
        )

        frequencies = torch.exp(
            -math.log(base) * frequency_indices / axis_dim
        )

        # Row positions: (grid_height, 1)
        row_positions = torch.arange(
            grid_height,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Column positions: (grid_width, 1)
        column_positions = torch.arange(
            grid_width,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Each has shape:
        #     (number of positions, axis_dim)
        row_encoding = torch.zeros(grid_height, axis_dim)
        column_encoding = torch.zeros(grid_width, axis_dim)

        row_angles = row_positions * frequencies
        column_angles = column_positions * frequencies

        row_encoding[:, 0::2] = torch.sin(row_angles)
        row_encoding[:, 1::2] = torch.cos(row_angles)

        column_encoding[:, 0::2] = torch.sin(column_angles)
        column_encoding[:, 1::2] = torch.cos(column_angles)

        # Expand row encodings across all columns:
        #     (grid_height, grid_width, axis_dim)
        row_grid = row_encoding[:, None, :].expand(
            grid_height,
            grid_width,
            axis_dim,
        )

        # Expand column encodings across all rows:
        #     (grid_height, grid_width, axis_dim)
        column_grid = column_encoding[None, :, :].expand(
            grid_height,
            grid_width,
            axis_dim,
        )

        # Concatenate row and column encodings:
        #     (grid_height, grid_width, embed_dim)
        pe = torch.cat((row_grid, column_grid), dim=-1)

        # Flatten the patch grid in row-major order:
        #     (1, grid_height * grid_width, embed_dim)
        pe = pe.reshape(1, grid_height * grid_width, embed_dim)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add fixed 2D positional encodings to patch embeddings.
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected x to have shape (B, N, D), but received "
                f"a tensor with {x.ndim} dimensions."
            )

        _, num_tokens, embed_dim = x.shape

        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} tokens from a "
                f"{self.grid_height}x{self.grid_width} patch grid, "
                f"but received {num_tokens} tokens."
            )

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, "
                f"but received {embed_dim}."
            )


        pe = self.pe.to(dtype=x.dtype)

        return self.dropout(x + pe)