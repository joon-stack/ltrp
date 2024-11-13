import torch
import timm.models.vision_transformer
import torch.nn as nn
from functools import partial


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, score_net=None, keep_nums=-1, random_chose=False, finetune_scorenet=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.score_net = score_net
        self.keep_nums = keep_nums
        self.random_chose = random_chose
        self.finetune_scorenet = finetune_scorenet
        if score_net and not finetune_scorenet:
            for p in self.score_net.parameters():
                p.requires_grad = False

        print("random_chose ", random_chose)

    def forward_features(self, x, mask_idx=None, is_train=True, only_feature=False):
        if is_train and self.score_net is not None:
            if self.random_chose:
                N = x.shltrp_base_and_vs_simmimape[0]
                L = self.patch_embed.num_patches
                noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
                len_keep = L if self.keep_nums <= 0 else self.keep_nums
                ids_keep = ids_shuffle[:, -len_keep:]
            else:
                if mask_idx is not None:
                    ids_keep = mask_idx
                else:
                    ids_keep = self.score_net.get_visible_tokens_idx(x, self.keep_nums)
        x = self.patch_embed(x)
        B, L, D = x.shape
        x = x + self.pos_embed[:, 1:, :]

        if is_train and self.score_net is not None:
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = cls_tokens + self.pos_embed[:, 0, :]
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if only_feature:
            return x
        else:
            return x[:, 0]

    def forward(self, x, mask_idx=None, is_train=True, only_feature=False):
        x = self.forward_features(x, mask_idx, is_train, only_feature)
        x = self.head(x)
        return x


def vit_tiny(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
