import torch
from score_net.dynamicViT_utils.dynamicViT_vit import VisionTransformerDiffPruning
import math


class dynamic_vit_small(VisionTransformerDiffPruning):
    def __init__(self, **kwargs):
        super(dynamic_vit_small, self).__init__(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                                qkv_bias=True,
                                                pruning_loc=[3, 6, 9], token_ratio=[0.1, 0.1, 0.1], viz_mode=True)

    def get_keep_indices(self, decisions):
        keep_indices = []
        for i in range(3):
            if i == 0:
                keep_indices.append(decisions[i])
            else:
                keep_indices.append(keep_indices[-1][decisions[i]])
        return keep_indices

    def get_visible_tokens_idx(self, x, len_keep):
        ratio = len_keep / 196
        base_keep_rate = math.pow(ratio, 1 / 3)
        self.training = False
        self.token_ratio = [base_keep_rate, base_keep_rate ** 2, base_keep_rate ** 3]
        output, decisions = self.forward(x)
        ret = []
        for i in range(0, x.shape[0]):
            d = [decisions[j][0][i] for j in range(3)]
            id = self.get_keep_indices(d)
            id = id[-1].unsqueeze(0)
            ret.append(id)
        ret = torch.cat(ret, dim=0)
        return ret
