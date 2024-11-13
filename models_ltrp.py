import torch
import torch.nn as nn
from models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
from factory import get_score_net, get_loss, get_img_metric


class LearnToRankPatchMIM(nn.Module):
    def __init__(self, mim, score_net, criterion, mask_all=False, asymmetric=False, img_metric=None):
        super().__init__()
        self.mim = mim
        self.score_net = score_net
        self.criterion = criterion
        self.mask_all = mask_all
        self.asymmetric = asymmetric
        self.img_metric = img_metric
        for p in self.mim.parameters():
            p.requires_grad = False

    def forward_mask_all(self, imgs, mask_ratio):
        # latent [N, Lv+1, D]
        # anchor [N, L + 1, D]
        # mask [N, L]
        # ids_restore [N, L]
        latent, anchor, mask, ids_restore = self.mim(imgs, mask_ratio)
        N, L = mask.shape
        mask = mask.to(torch.bool)
        y_pred = self.score_net(imgs)
        y_pred = y_pred[~mask].reshape(N, -1)

        lv = int(L * (1 - mask_ratio))
        rm_preds = []
        d_attn_mask = torch.zeros([N, lv + 1], requires_grad=False, device=imgs.device)  # [N, L + 1]
        for i in range(0, lv):
            d_attn_mask.fill_(0)
            d_attn_mask[:, i + 1] = 1
            _latent = self.mim.forward_encoderEx(imgs, mask, d_attn_mask)
            each = self.mim.forward_decoder(_latent, ids_restore)
            rm_preds.append(each)

        y_true = self.img_metric(anchor, rm_preds, mask)
        return y_pred, y_true

    def forward_mask_decoder(self, imgs, mask_ratio):
        # latent [N, Lv+1, D]
        # anchor [N, L + 1, D]
        # mask [N, L]
        # ids_restore [N, L]

        latent, anchor, mask, ids_restore = self.mim(imgs, mask_ratio)
        N, L, D = anchor.shape
        mask = mask.to(torch.bool)
        if self.asymmetric:
            y_pred = self.score_net(imgs, mask=mask)
        else:
            y_pred = self.score_net(imgs)

        y_pred = y_pred[~mask].reshape(N, -1)

        lv = latent.shape[1] - 1
        rm_preds = []
        d_attn_mask = torch.zeros([N, L + 1], requires_grad=False, device=imgs.device)  # [N, L + 1]
        ids_restore_ = torch.cat((torch.zeros([N, 1], device=imgs.device, dtype=torch.int64), ids_restore + 1), dim=-1)
        for i in range(0, lv):
            d_attn_mask.fill_(0)
            d_attn_mask[:, i + 1] = 1
            d_attn_mask = torch.gather(d_attn_mask, dim=1, index=ids_restore_)  # unshuff
            each = self.mim.forward_decoder(latent, ids_restore, d_attn_mask)
            rm_preds.append(each)
        y_true = self.img_metric(anchor, rm_preds, mask)

        return y_pred, y_true

    def forward(self, imgs, mask_ratio=0.75):
        if self.mask_all:
            y_pred, y_true = self.forward_mask_all(imgs, mask_ratio)
        else:
            y_pred, y_true = self.forward_mask_decoder(imgs, mask_ratio)

        loss = self.criterion(y_pred, y_true)
        return loss


def ltrp_mae_base_and_vit_small(args, **kwargs):
    mim = mae_vit_base_patch16(norm_pix_loss=True)
    score_net = get_score_net(score_net='vit_small', args=args)
    criterion = get_loss(args)
    img_metric = get_img_metric(args)
    model = LearnToRankPatchMIM(mim, score_net, criterion, args.mask_all, args.asymmetric, img_metric)
    return model


def ltrp_mae_base_and_vit_base(args, **kwargs):
    mim = mae_vit_base_patch16(norm_pix_loss=True)
    score_net = get_score_net(score_net='vit_base', args=args)
    criterion = get_loss(args)
    img_metric = get_img_metric(args)
    model = LearnToRankPatchMIM(mim, score_net, criterion, args.mask_all, args.asymmetric, img_metric)
    return model



def ltrp_mae_base_and_vit_tiny(args, **kwargs):
    mim = mae_vit_base_patch16(norm_pix_loss=True)
    score_net = get_score_net(score_net='vit_tiny')
    criterion = get_loss(args)
    img_metric = get_img_metric(args)
    model = LearnToRankPatchMIM(mim, score_net, criterion, args.mask_all, args.asymmetric, img_metric)
    return model


def ltrp_mae_huge_and_vit_tiny(args, **kwargs):
    mim = mae_vit_huge_patch14(norm_pix_loss=True)
    score_net = get_score_net(score_net='vit_tiny14', num_classes=256)
    criterion = get_loss(args)
    img_metric = get_img_metric(args)
    model = LearnToRankPatchMIM(mim, score_net, criterion, args.mask_all, args.asymmetric, img_metric)
    return model


def ltrp_mae_base_and_resnet50(args, **kwargs):
    mim = mae_vit_base_patch16(norm_pix_loss=True)
    score_net = get_score_net(score_net='res')
    criterion = get_loss(args)
    img_metric = get_img_metric(args)
    model = LearnToRankPatchMIM(mim, score_net, criterion, args.mask_all, args.asymmetric, img_metric)
    return model


def ltrp_mae_large_and_vit_small(args, **kwargs):
    mim = mae_vit_large_patch16(norm_pix_loss=True)
    score_net = get_score_net(score_net='vit_small')
    criterion = get_loss(args)
    img_metric = get_img_metric(args)
    model = LearnToRankPatchMIM(mim, score_net, criterion, args.mask_all, args.asymmetric,
                                img_metric)
    return model


def ltrp_mae_huge_and_vit_small(args, **kwargs):
    mim = mae_vit_huge_patch14(norm_pix_loss=True)
    score_net = get_score_net(score_net='vit_small14', num_classes=256)
    criterion = get_loss(args)
    img_metric = get_img_metric(args)
    model = LearnToRankPatchMIM(mim, score_net, criterion, args.mask_all, args.asymmetric,
                                img_metric)
    return model

def ltrp_mae_base_and_mobile_former_26m(args, **kwargs):
    mim = mae_vit_base_patch16(norm_pix_loss=True)
    score_net = get_score_net(score_net='mobile_former_26m')
    criterion = get_loss(args)
    img_metric = get_img_metric(args)
    model = LearnToRankPatchMIM(mim, score_net, criterion, args.mask_all, args.asymmetric, img_metric)
    return model

ltrp_base_and_vs = ltrp_mae_base_and_vit_small
ltrp_base_and_vt = ltrp_mae_base_and_vit_tiny
ltrp_huge_and_vt = ltrp_mae_huge_and_vit_tiny
ltrp_base_and_r50 = ltrp_mae_base_and_resnet50
ltrp_large_and_vs = ltrp_mae_large_and_vit_small
ltrp_huge_and_vs = ltrp_mae_huge_and_vit_small
ltrp_base_and_vb = ltrp_mae_base_and_vit_base
ltrp_base_and_mf26 = ltrp_mae_base_and_mobile_former_26m