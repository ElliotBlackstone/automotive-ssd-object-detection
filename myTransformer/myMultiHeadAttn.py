import torch
import torch.nn as nn



class myMultiHeadAttn(nn.Module):
    """
    Construct a new MultiHeadAttention layer.

    Inputs:
        - embed_dim: Dimension of the token embedding
        - num_heads: Number of attention heads
        - dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        # self.attn_drop = nn.Dropout(p = dropout)



    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attn_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, B is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (B, S, E)
        - key: Input data to be used as the key, of shape (B, T, E)
        - value: Input data to be used as the value, of shape (B, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
            i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (B, S, E) giving the weighted combination of
            data in value according to the attention weights calculated using key
            and query.
        """
        B, S, E = query.shape
        B, T, E = value.shape

        H = self.num_heads
        head_dim = self.embed_dim // self.num_heads
        # H * head_dim = E

        query = self.query(query) # (B, S, E)
        key   = self.key(key)     # (B, T, E)
        value = self.value(value) # (B, T, E)

        q = query.reshape(B, S, H, head_dim).transpose(1, 2)  # (B, H, S, head_dim)
        k = key.reshape(B, T, H, head_dim).transpose(1, 2)    # (B, H, T, head_dim)
        v = value.reshape(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, head_dim)

        # scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, H, S, T)
        # if attn_mask is not None:
        #   scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        # attn = self.attn_drop(torch.softmax(scores, dim=-1))
        # output = torch.matmul(attn, v) # (B, H, S, head_dim)

        attn = nn.functional.scaled_dot_product_attention(query=q,
                                                          key=k,
                                                          value=v,
                                                          attn_mask=attn_mask,
                                                          dropout_p=self.dropout_p if self.training else 0.0)

        attn = attn.transpose(1, 2).reshape(B, S, E)
        output = self.proj(attn)
        return output





class myMultiHeadSelfAttn(nn.Module):
    """
    Multi-head self-attention using one combined QKV projection.

    Instead of computing

        Q = X W_Q
        K = X W_K
        V = X W_V

    using three separate linear layers, compute

        [Q, K, V] = X W_{QKV}

    using one linear layer whose output dimension is 3 * embed_dim.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout

        # One large projection for query, key, and value.
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)

        # Final output projection.
        self.proj = nn.Linear(embed_dim, embed_dim)

        # self.attn_drop = nn.Dropout(p=dropout)

    def forward(self,
                x: torch.Tensor,
                attn_mask: torch.Tensor | None = None,
                ) -> torch.Tensor:
        """
        Inputs:
        - x: Input tensor of shape (B, S, E)
        - attn_mask: Optional attention mask of shape (S, S)

        Returns:
        - output: Tensor of shape (B, S, E)

        Here:
        B = batch size
        S = sequence length
        E = embedding dimension
        H = number of heads
        D = dimension per head = E / H
        """

        B, S, E = x.shape
        H = self.num_heads
        D = self.head_dim

        # ----------------------------------------------------------
        # Combined QKV projection
        #
        # x:   (B, S, E)
        # qkv: (B, S, 3E)
        # ----------------------------------------------------------
        qkv = self.qkv(x)

        # Split the final dimension into three tensors:
        #
        # q: (B, S, E)
        # k: (B, S, E)
        # v: (B, S, E)
        q, k, v = qkv.chunk(3, dim=-1)

        # Separate each embedding into H attention heads.
        #
        # (B, S, E)
        #      ->
        # (B, S, H, D)
        #      ->
        # (B, H, S, D)
        q = q.reshape(B, S, H, D).transpose(1, 2)
        k = k.reshape(B, S, H, D).transpose(1, 2)
        v = v.reshape(B, S, H, D).transpose(1, 2)

        # Attention scores:
        #
        # (B,H,S,D) @ (B,H,D,S)
        #          ->
        #       (B,H,S,S)
        # scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)

        # if attn_mask is not None:
        #     scores = scores.masked_fill(
        #         attn_mask == 0,
        #         float("-inf")
        #     )

        # attn = torch.softmax(scores, dim=-1)
        # attn = self.attn_drop(attn)

        

        # Weighted values:
        #
        # (B,H,S,S) @ (B,H,S,D)
        #          ->
        #       (B,H,S,D)
        # output = torch.matmul(attn, v)

        attn = nn.functional.scaled_dot_product_attention(query=q,
                                                          key=k,
                                                          value=v,
                                                          attn_mask=attn_mask,
                                                          dropout_p=self.dropout_p if self.training else 0.0)

        # Recombine heads:
        #
        # (B,H,S,D)
        #     ->
        # (B,S,H,D)
        #     ->
        # (B,S,E)
        attn = attn.transpose(1, 2).reshape(B, S, E)

        # Final output projection.
        output = self.proj(attn)

        return output