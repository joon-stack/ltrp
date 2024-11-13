import torch
import torch.nn as nn
from score_net.ia_red_utils.IA_RED_model import interp_deit_small_patch16_224


class IA_RED(nn.Module):
    def __init__(self, **kwargs):
        super(IA_RED, self).__init__()
        self.model = interp_deit_small_patch16_224(pretrained=False)

    def get_visible_tokens_idx(self, x, len_keep):
        with torch.no_grad():
            _, _, soft_policys = self.model(x)

        score = soft_policys[0][:, 1:]
        ids_shuffle = torch.argsort(score, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, -len_keep:]
        return ids_keep

    # def get_visible_tokens_idx(self, x, len_keep):
    #     with torch.no_grad():
    #         _, _, soft_policys = self.model(x)
    #
    #     #        score = soft_policys[0][:, 1:]
    #     score = None
    #     for i in soft_policys:
    #         if score is None:
    #             score = i[:, 1:]
    #         else:
    #             score = score + i[:, 1:]
    #     score = score / len(soft_policys)
    #     ids_shuffle = torch.argsort(score, dim=1)  # ascend: small is keep, large is remove
    #     ids_keep = ids_shuffle[:, -len_keep:]
    #     return ids_keep

