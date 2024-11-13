import torch.multiprocessing
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from score_net.dge_utils.DGE_vit import  deit_dge_s124_small_patch16_256
import os


class dge_small_patch16_224(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.vit = deit_dge_s124_small_patch16_256(num_classes=1000)
    def get_visible_tokens_idx(self, x, len_keep):
        self.vit.train()
        B = x.shape[0]
        padding = nn.ZeroPad2d(16)
        x = padding(x)
        ks = self.vit.get_last_selfattention(x.float())
        k = ks[-1]  # ([6, b, 257, 64])
        k = k.transpose(0, 1)  # [ b, h, 257, 64]
        k = k @ k.transpose(-1, -2)  # [B, H, 257, 257]
        k = k[:, :, 0, 1:]  # [B, H, 256]
        k = k.mean(dim=1)  # [B, 256]
        k = k.reshape(B, 16, 16)

        attentions = F.interpolate(k.unsqueeze(1), size=(14, 14), mode='bicubic')  # mode='bicubic
        attentions = attentions.squeeze(1)  # [B, 14,14]
        score = attentions.reshape(B, -1)
        ids_shuffle = torch.argsort(score, dim=1)  # ascend: small is keep, large is remove
        ids_shuffle = ids_shuffle[:, -len_keep:]
        return ids_shuffle

# test
