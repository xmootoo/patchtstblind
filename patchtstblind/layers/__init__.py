import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from patchtstblind.utils.utils import *

# Use torch.nn.MultiheadAttention which implements
# https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html. This uses the scaled
# dot-product attention mechanism. You simply forward pass the query, key and values.


class PatchTSTBackbone(nn.Module):
    def __init__(self, num_enc_layers, d_model, d_ff, num_heads, num_channels, num_patches, pred_len, attn_dropout=0.0,
        ff_dropout=0.0, pred_dropout=0.0, batch_first=True, norm_mode="batch1d", return_head=True):
        super(PatchTSTBackbone, self).__init__()

        # Parameters
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.return_head = return_head

        # Encoder
        self.enc = nn.Sequential(*(EncoderBlock(d_model, d_ff, num_heads, num_channels, num_patches, attn_dropout, ff_dropout,
                                                batch_first, norm_mode) for i in range(num_enc_layers)))
        # Prediction head
        self.head = SupervisedHead(num_patches*d_model, pred_len, pred_dropout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size * self.num_channels, self.num_patches, -1) # (batch_size * num_channels, num_patches, d_model)
        x = self.enc(x) # Encode
        x = x.view(batch_size, self.num_channels, self.num_patches, -1) # (batch_size, num_channels, num_patches, d_model)

        if self.return_head:
            x = self.head(x) # Predict forecast window

        return x


class SupervisedHead(nn.Module):
    def __init__(self, linear_dim, pred_len, dropout=0.0):
        super().__init__()
        """
        Flattens and applies a linear layer to each channel independently to form a prediction.
        Args:
            num_channels (int): The number of channels in the input.
            linear_dim (int): The dimension of the linear layer, should be num_patches * d_model.
            pred_len (int): The length of the forecast window.
            dropout (float): The dropout value.
        """

        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(linear_dim, pred_len)

    def forward(self, x) -> torch.Tensor:
        """
        Flattens and applies a linear layer to each channel independently to form a prediction.
        Args:
            x (torch.Tensor): The input of shape (batch_size, num_channels, num_patches, d_model)
        Returns:
            x (torch.Tensor): The output of shape (batch_size, num_channels, pred_len).
        """
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    Args:
        d_model: The embedding dimension.
        num_heads: The number of heads in the multi-head attention models.
        dropout: The dropout value.
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
        norm: The type of normalization to use. Either "batch1d", "batch2d", or "layer".
    """

    def __init__(self, d_model, d_ff, num_heads, num_channels, num_patches, attn_dropout=0.0, ff_dropout=0.0, batch_first=True,
                 norm_mode="batch1d"):
        super(EncoderBlock, self).__init__()

        # Layers
        self.attn = _MultiheadAttention(num_heads=num_heads, d_model=d_model, dropout=attn_dropout,
            batch_first=batch_first)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Dropout(ff_dropout),
                                nn.Linear(d_ff, d_model))

        # Normalization
        self.norm = Norm(norm_mode, num_channels, num_patches, d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_patches, d_model).
        Returns:
            fc_out: Output of the transformer block, a tensor of shape (batch_size, num_patches, d_model).
        """

        # Multihead Attention -> Add & Norm
        attn_out, _ = self.attn(x, x, x)
        attn_norm = self.norm(attn_out + x) # Treat the input as the query, key and value for MHA.

        # Feedforward layer -> Add & Norm
        fc_out = self.ff(attn_norm)
        fc_norm = self.norm(fc_out + attn_out)

        return fc_norm


class _MultiheadAttention(nn.Module):
    """
    Multihead Attention mechanism from the Vanilla Transformer, with some preset parameters for the PatchTST model.
    """

    def __init__(self, num_heads:int, d_model:int, dropout=0.0, batch_first=True):
        super(_MultiheadAttention, self).__init__()

        # Layers
        self.attn = nn.MultiheadAttention(embed_dim=d_model, 
                                          num_heads=num_heads, 
                                          dropout=dropout,
                                          batch_first=batch_first)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q: Query embedding of shape: (batch_size, num_patches, d_model).
            K: Key embedding of shape (batch_size, num_patches, d_model).
            V: Value embedding of shape (batch_size, num_patches, d_model).
            batch_size: The batch size.
            num_patches: The sequence length.
            d_model: The embedding dimension.
        Returns:
            x: The output of the attention layer of shape (batch_size, num_patches, d_model).
        """
        return self.attn(query=Q, key=K, value=V, need_weights=False)

class PositionalEncoding(nn.Module):
    def __init__(self, patch_dim : int=16, d_model : int=128, num_patches : int=64):
        super(PositionalEncoding, self).__init__()
        self.projection = nn.Linear(patch_dim, d_model)  # P x D projection matrix
        self.pos_encoding = nn.Parameter(torch.empty(num_patches, d_model))  # N x D positional encoding matrix

        # Weight initialization
        init.xavier_uniform_(self.projection.weight)
        init.uniform_(self.pos_encoding, -0.02, 0.02)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, M, N, P) where B = batch_size, M = num_channels, N = num_patches,
               P = patch_dim.
        Returns:
            x: tensor of shape (B, M, N, D) where D = d_model.
        """

        B, M, N, P = x.shape
        x = x.view(B*M, N, P) # Reshape the tensor to (B * M, N, P). We process each channel independently.
        x = self.projection(x) + self.pos_encoding.unsqueeze(0)
        x = x.view(B, M, N, -1) # Reshape the tensor to (B, M, N, D).

        return x
    


class RevIN(nn.Module):
    def __init__(self, num_channels: int, eps=1e-5, affine=True):
        """
        Kim et al. (2022): Reversible instance normalization for accurate time-series forecasting against
        distribution shift. Provides a learnable instance normalization layer that is reversible. Code is
        from https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/RevIN.py with modifications,
        which was originally taken from https://github.com/ts-kim/RevIN.

        Args:
            num_channels: The number of features or channels.
            eps: A value added for numerical stability.
            affine: If True, RevIN has learnable affine parameters (e.g., like LayerNorm).
        """

        super(RevIN, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.affine_weight = nn.Parameter(torch.ones(self.num_channels))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))

    def forward(self, x, mode:str):
        """
        Forward pass for normalization or denormalizating the time series with learnable affine transformations.

        Args:
            x: Input tensor of shape (batch_size, num_channels, seq_len).
            mode: 'norm' for normalization and 'denorm' for denormalization.
        Returns:
            Normalized or denormalized tensor of same shape as input tensor.
        """

        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=2, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = (x / self.stdev)
        if self.affine:
            x = x * self.affine_weight.unsqueeze(0).unsqueeze(-1) # Reshape: (1, num_channels, 1)
            x = x + self.affine_bias.unsqueeze(0).unsqueeze(-1) # Reshape: (1, num_channels, 1)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias.unsqueeze(0).unsqueeze(-1)
            x = x / (self.affine_weight.unsqueeze(0).unsqueeze(-1) + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class Patcher(nn.Module):
    """
    Splits the input time series into patches.
    """

    def __init__(self, patch_dim : int=16, stride : int=8):
        super(Patcher, self).__init__()
        self.patch_dim = patch_dim
        self.stride = stride

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, M, L). B: batch_size, M: channels, L: sequence_length.
        Returns:
            patches: tensor of shape (B, M, N, P). N: number of patches, P: patch_dim.
            patches_combined: tensor of shape (B * M, N, P). N: number of patches, P: patch_dim. This is more efficient
                              to input into the Transformer encoder, as we are applying it to channels independently, thus,
                              we can combine the batch and channel dimensions and then reshape it afterwards.
        """
        B, M, L = x.shape

        # Number of patches.
        N = int((L - self.patch_dim) / self.stride) + 2

        # Pad the time series with the last value on each channel repeated S times
        last_column = x[:, :, -1:] # index
        padding = last_column.repeat(1, 1, self.stride)
        x = torch.cat((x, padding), dim=2)

        # Extract patches
        patches = x.unfold(dimension=2, size=self.patch_dim, step=self.stride) # Unfold the input tensor to extract patches.
        patches = patches.contiguous().view(B, M, N, self.patch_dim) # Reshape the tensor to (B, M, N, P).
        patches_combined = patches.view(B * M, N, self.patch_dim) # Reshape the tensor to (B * M, N, P).

        return patches
