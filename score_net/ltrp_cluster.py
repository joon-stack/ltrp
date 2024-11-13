from einops import rearrange
import math


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]
    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def cluster_dpc_knn(x, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num


import torch
import torch.nn as nn
from score_net.vision_transformer import Block, PatchEmbed
from utils.pos_embed import get_2d_sincos_pos_embed
import timm.models.vision_transformer


class ltrp_cluster(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, num_classes=196,
                 depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, global_pool=False, ratio=0.7):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.global_pool = global_pool
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.ratio = ratio
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, attn_mask, mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        if mask is not None:
            _, N, C = x.shape
            x = x + self.pos_embed[:, 1:, :]
            x = x[~mask].reshape(B, -1, C)
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x, attn_mask)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, attn_mask=None, mask=None):
        x = self.forward_features(x, attn_mask, mask)
        x = self.head(x)
        return x

    def get_visible_tokens_idx_dpc_knn(self, x, len_keep, patch_size=14):
        x = rearrange(x, 'b c (p1 h) (p2 w) -> b (p1 p2) (h w c)', p1=patch_size, p2=patch_size)
        N, L, _ = x.shape
        idx_cluster, cluster_num = cluster_dpc_knn(x, len_keep, 5)
        return idx_cluster

    def get_visible_tokens_idx(self, x, len_keep):

        with torch.no_grad():
            score = self.forward(x)
            patch_size = int(math.sqrt(score.shape[-1]))
            idx_cluster = self.get_visible_tokens_idx_dpc_knn(x, len_keep, patch_size)

        ####
        N, L = score.shape
        temp = []
        for i in range(0, idx_cluster.shape[0]):
            idx_counts = torch.unique(idx_cluster[i], return_counts=True)[1]
            temp.append(idx_counts.unsqueeze(0))

        ratio = self.ratio
        len_keep_of_ltrp = int(len_keep * ratio)

        ids_shuffle = torch.argsort(score, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, -len_keep_of_ltrp:]

        ####
        idx_counts = torch.cat(temp, dim=0)
        idx_counts = idx_counts.cumsum(dim=-1)
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        idx_cluster_noise = idx_cluster + noise

        v, i = torch.sort(idx_cluster_noise)
        idx = torch.gather(i, dim=1, index=(idx_counts - 1))

        temp = torch.cat(temp, dim=0)
        mask = torch.zeros([N, L], device=x.device)
        mask.scatter_(1, idx, 1)
        mask.scatter_(1, ids_keep, 0)

        p = torch.gather(mask, dim=1, index=idx)
        temp = temp * p
        ids_shuffle_knn = torch.argsort(temp, dim=1, descending=True)

        ids_shuffle_knn = ids_shuffle_knn[:, :(len_keep - len_keep_of_ltrp)]
        ids_keep_knn = torch.gather(idx, dim=1, index=ids_shuffle_knn)

        idx = torch.cat((ids_keep, ids_keep_knn), dim=-1)

        return idx

    def get_last_selfattention(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)


def ltrp_cluster_vit_tiny_patch16(**kwargs):
    model = ltrp_cluster(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, num_classes=256, **kwargs)
    return model


def ltrp_cluster_vit_small_patch16(**kwargs):
    model = ltrp_cluster(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, **kwargs)
    return model


def ltrp_cluster_vit_small_patch14(**kwargs):
    model = ltrp_cluster(
        patch_size=14, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, num_classes=256, **kwargs)
    return model


def ltrp_cluster_vit_tiny_patch14(**kwargs):
    model = ltrp_cluster(
        patch_size=14, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, num_classes=256, **kwargs)
    return model


ltrp_cluster
_vt = ltrp_cluster_vit_tiny_patch16
ltrp_cluster_vs = ltrp_cluster_vit_small_patch16
ltrp_cluster_vs14 = ltrp_cluster_vit_small_patch14
ltrp_cluster_vt14 = ltrp_cluster_vit_tiny_patch14
