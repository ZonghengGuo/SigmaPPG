import torch
import torch.nn as nn
from codebook.modeling_finetune import TemporalConv, Block
from functools import partial
import math


class MaskGenerator(nn.Module):
    def __init__(self, input_size=12000, patch_size=100, in_chans=1, embed_dim=32, depth=1, num_heads=2):
        super().__init__()
        self.patch_size = patch_size

        self.patch_embed = TemporalConv(in_chans=in_chans, out_chans=embed_dim)
        t_conv1 = math.floor((self.patch_size - 15 + 2 * 7) / 8) + 1
        t_conv2 = math.floor((t_conv1 - 3 + 2 * 1) / 1) + 1
        t_conv3 = math.floor((t_conv2 - 3 + 2 * 1) / 1) + 1
        temporal_conv_output_dim = t_conv3 * embed_dim

        if temporal_conv_output_dim != embed_dim:
            self.proj = nn.Linear(temporal_conv_output_dim, embed_dim)
        else:
            self.proj = nn.Identity()

        self.num_patches = input_size // patch_size

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.proj(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x).squeeze(-1)
        return logits