import torch
import torch.nn as nn
import torchvision.models as models
from utils.grad_cam import GradCAM
from score_net.grad_cam_utils.grad_cam_util import deit_small_patch16_224


class grad_cam_r50(nn.Module):
    def __init__(self):
        super(grad_cam_r50, self).__init__()
        self.res = models.resnet50(pretrained=True)
        target_layers = [self.res.layer4]
        self.cam = GradCAM(model=self.res, target_layers=target_layers, use_cuda=False)

    def get_visible_tokens_idx(self, x, len_keep, target_category=None):
        with torch.enable_grad():
            grayscale_cam = self.cam(input_tensor=x, target_category=target_category, target_size=(14, 14))
            score = torch.tensor(grayscale_cam)
            B, _, _ = score.shape
            score = torch.reshape(score, [B, -1])  # [B, N]
            ids_shuffle = torch.argsort(score, dim=1)  # ascend: small is keep, large is remove
            ids_keep = ids_shuffle[:, -len_keep:]
            ids_keep = ids_keep.to(x.device)
        return ids_keep


class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


class grad_cam_vit(nn.Module):
    def __init__(self):
        super(grad_cam_vit, self).__init__()
        self.vit = deit_small_patch16_224(pretrained=True, num_classes=1000)
        target_layers = [self.vit.blocks[-1].norm1]
        self.cam = GradCAM(model=self.vit,
                           target_layers=target_layers,
                           use_cuda=False,
                           reshape_transform=ReshapeTransform(self.vit))

    def get_visible_tokens_idx(self, x, len_keep, target_category=None):
        with torch.enable_grad():
            grayscale_cam = self.cam(input_tensor=x, target_category=target_category, target_size=(14, 14))
            score = torch.tensor(grayscale_cam)
            B, _, _ = score.shape
            score = torch.reshape(score, [B, -1])  # [B, N]
            ids_shuffle = torch.argsort(score, dim=1)  # ascend: small is keep, large is remove
            ids_keep = ids_shuffle[:, -len_keep:]
            ids_keep = ids_keep.to(x.device)
        return ids_keep
