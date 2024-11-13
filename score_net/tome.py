import score_net.tome_utils as tome_utils
import torch.nn as nn
import torch
from score_net.tome_utils.tome_vit import vit_small_patch16_224 as vit_small


class tome_small_patch16_224(nn.Module):
    def __init__(self, **kwargs):
        super(tome_small_patch16_224, self).__init__()
        self.vit = vit_small()
        tome_utils.patch.timm(self.vit, trace_source=True)

    def get_visible_tokens_idx(self, x, len_keep):
        r = int((197 - len_keep) / 12)
        self.vit.r = r
        _ = self.vit(x)
        source = self.vit._tome_info["source"]
        source = source[:, 1:, :]
        B, N, L = source.shape
        noise = torch.rand((B, N, L), device=x.device)
        source = source + noise
        _, i = torch.sort(source)
        i = i[:, :, -1]
        i = i - 1  # [1,196] -> [0, 195]
        return i[:, -len_keep:]


