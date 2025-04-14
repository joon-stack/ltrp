import torch
import torch.nn as nn
from models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
from factory import get_score_net, get_loss, get_img_metric
from models_dino_wm import load_model
import random



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

        print("INFO: self.score_net", self.score_net)

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
    


class LearnToRankPatchWM(nn.Module):
    def __init__(self, dino_wm, score_net, criterion, mask_all=False, asymmetric=False, img_metric=None):
        super().__init__()
        self.dino_wm = dino_wm
        self.score_net = score_net        # 점수 예측 네트워크
        self.criterion = criterion        # Learn-to-Rank 손실 함수
        self.mask_all = mask_all          # 전체 마스크 옵션
        self.asymmetric = asymmetric      # 비대칭 옵션
        self.img_metric = img_metric      # 이미지 메트릭 (사용되지 않음)
        self.sample_patch_num = 1
        
        # DINO 인코더와 World Model의 파라미터는 학습되지 않도록 고정
        for p in self.dino_wm.parameters():
            p.requires_grad = False


        print("INFO: self.score_net", self.score_net)

    def forward_mask_all(self, imgs, acts, mask_ratio):
        # DINO 인코더를 사용하여 특징 추출
        latent, _, mask, ids_restore = self.dino_wm(imgs, acts)  # [N, L + 1, D]
        N, L = mask.shape 
        # print("INFO: latent.shape, anchor.shape, mask.shape, ids_restore.shape", latent.shape, anchor.shape, mask.shape, ids_restore.shape)
        mask = mask.to(torch.bool)
        y_pred = self.score_net(imgs["visual"][:, 0])
        y_pred = y_pred[~mask].reshape(N, -1)

        lv = int(L * (1 - mask_ratio))
        rm_preds = []
        anchor_preds = []
        d_attn_mask = torch.zeros([N, lv + 1], requires_grad=False, device=imgs['visual'].device)  # [N, L + 1]
        for i in range(0, lv):
            d_attn_mask.fill_(0)
            d_attn_mask[:, i + 1] = 1
            _latent = self.dino_wm.encodeEx(imgs, acts) # [N, T, L_m, D]
            N, T, L_m, D = _latent.shape
            
            # i번째 패치를 제외한 랜덤 패치 선택
            # L_m 차원에서 i와 다른 패치를 중복 없이 효율적으로 선택
            valid_indices = torch.arange(L_m, device=_latent.device)
            valid_indices = valid_indices[valid_indices != i]  # i번째 인덱스 제외
            
            # torch.randperm로 중복 없이 랜덤하게 인덱스 생성 후 필요한 만큼만 사용
            num_samples = min(self.sample_patch_num, len(valid_indices))
            perm = torch.randperm(len(valid_indices), device=_latent.device)
            random_indices = valid_indices[perm[:num_samples]]
            
            anchor = _latent[:, -1]
            for k in random_indices:
                _latent_clone = _latent.clone()
                _latent_clone[:, :, i] = _latent[:, :, k]
                print(f"INFO: {i}th patch is substituted by {k}th patch")

                _, each = self.dino_wm.forwardEx(_latent_clone)
                rm_preds.append(each)
                anchor_preds.append(anchor)
        
        y_true = self.img_metric(anchor_preds, rm_preds, mask)
        return y_pred, y_true

    def forward_mask_decoder(self, imgs, acts, mask_ratio):
        # DINO 인코더를 사용하여 특징 추출
        latent, _, mask, ids_restore = self.dino_wm(imgs, acts)  # [N, L + 1, D]
        N, L = mask.shape 
        # print("INFO: latent.shape,  mask.shape, ids_restore.shape", latent.shape,  mask.shape, ids_restore.shape)
        mask = mask.to(torch.bool)
        if self.asymmetric:
            # only history
            y_pred = self.score_net(imgs["visual"][:, 0], mask=mask)
        else:
            y_pred = self.score_net(imgs["visual"][:, 0])
        y_pred = y_pred[~mask].reshape(N, -1)

        lv = int(L * (1 - mask_ratio))
        rm_preds = []
        anchor_preds = []
        d_attn_mask = torch.zeros([N, lv + 1], requires_grad=False, device=imgs['visual'].device)  # [N, L + 1]
        for i in range(0, lv):
            _latent = self.dino_wm.encodeEx(imgs, acts) # [N, T, L_m, D]
            N, T, L_m, D = _latent.shape
            # i번째 패치를 제외한 랜덤 패치 선택
            # L_m 차원에서 i와 다른 패치를 중복 없이 효율적으로 선택
            valid_indices = torch.arange(L_m, device=_latent.device)
            valid_indices = valid_indices[valid_indices != i]  # i번째 인덱스 제외
            
            # torch.randperm로 중복 없이 랜덤하게 인덱스 생성 후 필요한 만큼만 사용
            num_samples = min(self.sample_patch_num, len(valid_indices))
            perm = torch.randperm(len(valid_indices), device=_latent.device)
            random_indices = valid_indices[perm[:num_samples]]
            assert len(random_indices) == num_samples
            
            anchor = _latent[:, -1:]
            for k in random_indices:
                _latent_clone = _latent.clone()
                _latent_clone[:, :, i] = _latent[:, :, k]
                # print(f"INFO: {i}th patch is substituted by {k}th patch")

                _, each = self.dino_wm.forwardEx(_latent_clone)
                # print("INFO: anchor.shape, each.shape: in forward_mask_decoder  ", anchor.shape, each.shape)
                rm_preds.append(each)
                anchor_preds.append(anchor)
        
        y_true = self.img_metric(anchor_preds, rm_preds, mask)
        return y_pred, y_true

        # World Model의 prediction error로 y_true 계산
        y_true = self.calculate_prediction_error(latent, rm_preds, mask)
        return y_pred, y_true

    def forward(self, imgs, acts, mask_ratio=0.75):
        if self.mask_all:
            y_pred, y_true = self.forward_mask_all(imgs, acts, mask_ratio)
        else:
            y_pred, y_true = self.forward_mask_decoder(imgs, acts, mask_ratio)

        # 손실 계산
        loss = self.criterion(y_pred, y_true)
        return loss


    def calculate_prediction_error(self, latent, rm_preds, mask):
        """World Model의 prediction error 계산"""
        errors = []
        for pred in rm_preds:
            # 예시로 MSE 사용
            error = (latent - pred).pow(2).mean(dim=-1)  # [N, L]
            errors.append(error)
        y_true = torch.stack(errors, dim=1)  # [N, lv]
        return y_true



def ltrp_mae_base_and_vit_small(args, **kwargs):
    mim = mae_vit_base_patch16(norm_pix_loss=True)
    score_net = get_score_net(score_net='ltrp_cluster_vs', args=args)
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

def ltrp_wm_dino_small_and_vit_small(args, **kwargs):
    dino_wm = load_model(args)
    score_net = get_score_net(score_net='ltrp_cluster_vs', args=args)
    criterion = get_loss(args)
    img_metric = get_img_metric(args)
    model = LearnToRankPatchWM(dino_wm, score_net, criterion, args.mask_all, args.asymmetric, img_metric)
    return model

ltrp_base_and_vs = ltrp_mae_base_and_vit_small
ltrp_base_and_vt = ltrp_mae_base_and_vit_tiny
ltrp_huge_and_vt = ltrp_mae_huge_and_vit_tiny
ltrp_base_and_r50 = ltrp_mae_base_and_resnet50
ltrp_large_and_vs = ltrp_mae_large_and_vit_small
ltrp_huge_and_vs = ltrp_mae_huge_and_vit_small
ltrp_base_and_vb = ltrp_mae_base_and_vit_base
ltrp_base_and_mf26 = ltrp_mae_base_and_mobile_former_26m

ltrp_dinowm_and_vs = ltrp_wm_dino_small_and_vit_small