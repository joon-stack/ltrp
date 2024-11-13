import torch
import torch.nn as nn
from score_net.dino_utils.dino_vit import VisionTransformer
from functools import partial


class dino_vit_small(VisionTransformer):
    def __init__(self, head_idx=None, **kwargs):
        self.head_idx = head_idx
        super(dino_vit_small, self).__init__(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0)

    def get_visible_tokens_idx(self, x, len_keep):
        attentions = self.get_last_selfattention(x.float())
        # we keep only the output patch attention
        b, nh, _, _ = attentions.shape  # number of head
        attentions = attentions[:, :, 0, 1:].reshape(b, nh, -1)

        threshold = len_keep / 196
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=2, keepdim=True)
        cumval = torch.cumsum(val, dim=2)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for i in range(b):
            for head in range(nh):
                th_attn[i, head] = th_attn[i, head, idx2[i, head]]

        attentions = th_attn.int()
        if self.head_idx is None:
            z = attentions.sum(dim=1)
        else:
            z = attentions[:, self.head_idx, :]

        ids_shuffle = torch.argsort(z, dim=1)
        ids_keep = ids_shuffle[:, -len_keep:]
        return ids_keep
