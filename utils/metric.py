import torch
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import PeakSignalNoiseRatio
from torchvision.transforms import Normalize


@torch.no_grad()
def l1(anchor, rm_preds, mask):

    if isinstance(anchor, list):
        scores = []
        for i, rm_pred in enumerate(rm_preds):
            anchor_each = anchor[i].squeeze() # [N, L_m, D]
            rm_pred_each = rm_pred.squeeze() # [N, L_m, D]

            # print("INFO: anchor_each.shape, rm_pred_each.shape: ", anchor_each.shape, rm_pred_each.shape)
            
            score = (anchor_each - rm_pred_each).abs()
            # print("INFO: score.shape: ", score.shape)
            score = score.mean(dim=(-1, -2)).unsqueeze(-1)  # [N, 1]
            # print("INFO: score_after.shape: ", score.shape)

            assert score.shape == (anchor_each.shape[0], 1)
            scores.append(score)
    else:
    
        N, _, D = anchor.shape
        scores = []
        print("INFO: mask: ", mask)
        anchor_masked = anchor[mask].reshape(N, -1, D)

        for rm_pred in rm_preds:
            print("INFO: anchor.shape, rm_pred.shape, mask.shape: ", anchor.shape, rm_pred.shape, mask.shape)
            each = rm_pred[mask].reshape(N, -1, D)  # [N, L_m, D]
            score = (anchor_masked - each).abs()
            score = score.mean(dim=(-1, -2)).unsqueeze(-1)  # [N, 1]
            scores.append(score)

    scores = torch.cat(scores, dim=-1)
    return scores





def unpatchify(x):
    h, w, p = 14, 14, 16
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


@torch.no_grad()
def ssim(anchor, rm_preds, mask):
    anchor = unpatchify(anchor)
    ssim_loss = StructuralSimilarityIndexMeasure().to(anchor.device)
    scores = torch.zeros([anchor.shape[0], len(rm_preds)], device=anchor.device)
    for i in range(0, len(rm_preds)):
        rm_pred = rm_preds[i]
        each = unpatchify(rm_pred)
        for j in range(0, anchor.shape[0]):
            # n image
            score = ssim_loss(anchor[j].unsqueeze(0), each[j].unsqueeze(0))
            scores[j, i] = score
    scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
    return scores


@torch.no_grad()
def psnr(anchor, rm_preds, mask):
    anchor = unpatchify(anchor)
    psnr = PeakSignalNoiseRatio().to(anchor.device)
    scores = torch.zeros([anchor.shape[0], len(rm_preds)], device=anchor.device)
    for i in range(0, len(rm_preds)):
        rm_pred = rm_preds[i]
        each = unpatchify(rm_pred)
        for j in range(0, anchor.shape[0]):
            score = psnr(anchor[j].unsqueeze(0), each[j].unsqueeze(0))
            scores[j, i] = score

    scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())

    return scores


@torch.no_grad()
def lpips(anchor, rm_preds, mask):
    img_norm = Normalize(
        mean=torch.tensor((0.485, 0.456, 0.406)),
        std=torch.tensor((0.229, 0.224, 0.225)))
    anchor = img_norm(anchor)
    lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(anchor.device)
    scores = torch.zeros([anchor.shape[0], len(rm_preds)])
    for i in range(0, len(rm_preds)):
        rm_pred = rm_preds[i]
        each = unpatchify(rm_pred)
        each = img_norm(each)
        for j in range(0, anchor.shape[0]):
            # n image
            score = lpips_loss(anchor[j].unsqueeze(0), each[j].unsqueeze(0))
            scores[j, i] = score
    scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
    return scores
