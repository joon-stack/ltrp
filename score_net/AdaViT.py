import torch
import torch.nn as nn
from functools import partial
import argparse
import torch.nn as nn
from score_net.adaViT_utils.ada_vit import AdaStepT2T_ViT

class ada_vit(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.vit = AdaStepT2T_ViT(
            use_t2t=False,
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pretrained=False,
            ada_head=True,
            ada_layer=True,
            ada_token_with_mlp=True,
            ada_block=False,
            ada_head_v2=False,
            dyna_data=False,
            ada_token=True,
            ada_token_nonstep=False,
            ada_head_attn=False, )

    def get_visible_tokens_idx(self, x, len_keep):
        x, head_select, layer_select, token_select, select_logtis = self.vit(x)
        token_select = token_select[:, -6:, 1:]
        token_select = token_select.sum(dim=1)
        ids = torch.argsort(token_select, dim=-1)
        return ids[:, -len_keep:]
