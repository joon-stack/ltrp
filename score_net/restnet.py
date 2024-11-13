from torchvision.models.resnet import resnet34, resnet18
from torchvision.models.resnet import Bottleneck
import torch
import torch.nn as nn

class resnet(nn.Module):
    def __init__(self, model=None):
        super(resnet, self).__init__()
        self.model = model

    def get_visible_tokens_idx(self, x, len_keep):
        score = self.model.forward(x)
        ids_shuffle = torch.argsort(score, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, -len_keep:]
        return ids_keep



def r34():
    model = resnet34(num_classes=196)
    return resnet(model=model)

def r18():
    model = resnet18(num_classes=196)
    return resnet(model=model)



