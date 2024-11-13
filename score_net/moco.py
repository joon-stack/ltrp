from torchvision.models.resnet import ResNet, Bottleneck
from torch import Tensor
import torch
import torch.nn as nn
from score_net.dino_utils.dino_vit import VisionTransformer
from functools import partial


class moco_res50(ResNet):
    def __init__(self, **kwargs):
        super(moco_res50, self).__init__(Bottleneck, [3, 4, 6, 3], **kwargs)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        z = self.layer4(x)

        x = self.avgpool(z)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, z

    def get_visible_tokens_idx(self, x, len_keep):
        y, z = self._forward_impl(x)
        B = x.shape[0]
        z = z.sum(dim=1)  # [B, 7, 7]
        z = (z - z.min()) / (z.max() - z.min())
        score = nn.functional.interpolate(
            z.unsqueeze(1).reshape(B, 1, 7, 7),
            size=[14, 14],
            mode='bicubic',
        ).reshape(B, 1, -1).squeeze(1)
        ids_shuffle = torch.argsort(score, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, -len_keep:]
        return ids_keep


class moco_vit_small(VisionTransformer):
    def __init__(self, **kwargs):
        super(moco_vit_small, self).__init__(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0)

    def get_visible_tokens_idx(self, x, len_keep):
        attentions = self.get_last_selfattention(x.float())
        b, nh, _, _ = attentions.shape
        attentions = attentions[:, :, 0, 1:].reshape(b, nh, -1)
        z = attentions.mean(dim=1)

        ids_shuffle = torch.argsort(z, dim=1)
        ids_keep = ids_shuffle[:, -len_keep:]
        return ids_keep