import torch
import PIL
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import utils

path = '/home/luoyang/ltrp/other/output/'
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def _export_img(image, i, j):
    image = image.cpu()
    if j == 0:
        img_name = 'anchor_' + str(i)+'.png'
    else:
        img_name = 'anchor_' + str(i) + '_' + str(j)+'.png'

    image = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    #plt.imsave(path+img_name, image)
    print(image.shape)
    image = torch.einsum('hwc->chw', image)
    image = image.to(torch.float64)
    utils.save_image(image, img_name)


def export_imgs(anchor, rm_preds, mask, mim):
    N, _, _ = anchor.shape
    #for i in range()
    anchor = mim.unpatchify(anchor)
    rm_preds = [mim.unpatchify(i) for i in rm_preds]
    anchor = torch.einsum('nchw->nhwc', anchor).detach()
    rm_preds = [torch.einsum('nchw->nhwc', i).detach() for i in rm_preds]
    for i in range(0, N):
        _export_img(anchor[i],i, 0)
        for j in range(0, rm_preds[i].shape[0]):
            _export_img(rm_preds[i][j], i, j+1)
    print(anchor.shape)


