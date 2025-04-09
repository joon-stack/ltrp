
from score_net.dino_utils.dino_mlp import DINOHead
import utils.ltrp_loss as ltrp_loss
import score_net.evit as evit
import score_net.grad_cam as grad_cam
import score_net.moco as moco
from score_net.dino import dino_vit_small
from score_net.dpc_knn import dpc_knn
from score_net.gfnet import gf_net
from score_net.IA_RED import IA_RED
import utils.metric as metric
import score_net.restnet as restnet
import score_net.ltrp_cluster as ltrp_cluster
from score_net.dynamicViT import dynamic_vit_small

from score_net.DGE import  dge_small_patch16_224
from score_net.tome import tome_small_patch16_224
from score_net.AdaViT import ada_vit
from score_net.mobile_former import mobile_former_26m

def get_score_net(args=None, score_net='', **kwargs):
    model = args.score_net if score_net == '' else score_net
    print("get_score_net ", model)
    if model.startswith('ltrp_cluster'):
        # model = ltrp_cluster.__dict__[model](ratio=args.ltrp_cluster_ratio)
        model = ltrp_cluster.__dict__[model]()
        print(f"score net is {model}")

    elif model.startswith('dino_mlp'):
        model = DINOHead(**kwargs)
    elif model.startswith('gf_net'):
        model = gf_net()
    elif model.startswith('AdaViT'):
        model = ada_vit()
    elif model.startswith('IA_RED'):
        model = IA_RED()
    elif model.startswith('grad_cam'):
        model = grad_cam.grad_cam_vit(**kwargs)
    elif model.startswith('moco'):
        model = moco.__dict__[model](**kwargs)
    elif model.startswith('dino_vit_small'):
        model = dino_vit_small(head_idx=args.dino_head_idx)
    elif model.startswith('dynamic_vit_small'):
        model = dynamic_vit_small()
    elif model.startswith('dpc_knn'):
        model = dpc_knn()
    elif model.startswith('dge_small'):
        model = dge_small_patch16_224()
    elif model.startswith('tcformer'):
        pass
        #model = tcformer_small_patch16_224()
    elif model.startswith('tome'):
        model = tome_small_patch16_224(nb_classes=args.nb_classes)
    elif model.startswith('r'):
        model = restnet.__dict__[model]()
    elif model.startswith('evit'):
        model = evit.__dict__[model](
            num_classes=1000,
            drop_rate=0,
            base_keep_rate=0.25,
            drop_path_rate=0.1,
            fuse_token=True,
            img_size=(224, 224))
    elif model.startswith('mobile_former'):
        model = mobile_former_26m(num_classes=196)
    else:
        print("score net is None")
        return None

    return model


def get_loss(args):
    if args.ltr_loss == 'list_mle':
        criterion = ltrp_loss.list_mle(k=args.list_mle_k)
    elif args.ltr_loss == 'list_mleEx':
        criterion = ltrp_loss.list_mleEx(k=args.list_mle_k)
    elif args.ltr_loss == 'rank_net':
        criterion = ltrp_loss.rank_net(thread=args.rank_net_t, sigma=args.rank_net_sigma)
    elif args.ltr_loss == 'list_net':
        criterion = ltrp_loss.list_net()
    elif args.ltr_loss == 'point_wise':
        criterion = ltrp_loss.point_wise()
    elif args.ltr_loss == 'focused_rank':
        criterion = ltrp_loss.focused_rank(k=args.focused_rank_k)
    else:
        print("please set ltr loss function")
        exit(0)

    return criterion


def get_img_metric(args, _img_metric=''):
    img_metric = args.img_metric if _img_metric == '' else _img_metric
    if img_metric == '':
        img_metric = 'l1'
    if img_metric is not None and img_metric in metric.__dict__:
        func = metric.__dict__[img_metric]
    else:
        func = None
    return func

