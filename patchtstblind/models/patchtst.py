import torch.nn as nn
from patchtstblind.layers import RevIN, \
                                 PositionalEncoding, \
                                 Patcher, \
                                 PatchTSTBackbone


class PatchTSTBlind(nn.Module):
    def __init__(self, num_enc_layers, d_model, d_ff, num_heads, num_channels, seq_len, pred_len, attn_dropout=0.0,
        ff_dropout=0.0, pred_dropout=0.0, batch_first=True, norm_mode="batch1d", revin=True, revout=True, revin_affine=True,
        eps_revin=1e-5, patch_dim=16, stride=1, return_head=True):
        super(PatchTSTBlind, self).__init__()

        # Parameters
        self.num_patches = int((seq_len - patch_dim) / stride) + 2
        self.num_channels = num_channels
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine

        # Initialize layers
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self._revin = None
            self.revout = None

        self.patcher = Patcher(patch_dim, stride)
        self.pos_enc = PositionalEncoding(patch_dim, d_model, self.num_patches)
        self.backbone = PatchTSTBackbone(num_enc_layers, d_model, d_ff, num_heads, num_channels, self.num_patches, pred_len,
                                 attn_dropout,ff_dropout, pred_dropout, batch_first, norm_mode, return_head)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def _init_revin(self, revout:bool, revin_affine:bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

    def forward(self, x):

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        # Patcher
        x = self.patcher(x)

        # Project + Positional Encoding
        x = self.pos_enc(x)

        # Transformer + Linear Head
        x = self.backbone(x)

        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        return x

