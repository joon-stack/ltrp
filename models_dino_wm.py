import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat

import numpy as np

import hydra 
import os
from omegaconf import OmegaConf
from pathlib import Path
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

import sys
sys.path.insert(0, '/home/s2/youngjoonjeong/github/dino_wm/models')
sys.path.insert(0, "/shared/s2/lab01/youngjoonjeong/dino_wm_oc")



ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i+1) + [zeros] * (nwindow - i-1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

def latin_hypercube_sampling(n, grid_size=14):
    """
    Latin Hypercube Sampling을 이용해 grid_size x grid_size 격자에서 n개의 점을 선택합니다.
    반환되는 좌표는 0부터 grid_size-1까지의 정수 (행, 열) 쌍입니다.
    """
    # 각 차원(행, 열)에 대해 n개의 샘플 생성 (0~1 범위)
    samples = np.zeros((n, 2))
    for dim in range(2):
        # [0, 1]을 n개의 구간으로 나누고, 각 구간에서 임의의 값을 선택
        intervals = np.linspace(0, 1, n+1)
        points = intervals[:-1] + np.random.rand(n) * (1.0 / n)
        # 각 차원의 순서를 무작위로 섞음
        np.random.shuffle(points)
        samples[:, dim] = points

    # [0,1]에서 구한 값을 격자 크기에 맞게 스케일하고, 0-indexed 정수 좌표로 변환
    grid_samples = np.floor(samples * grid_size).astype(int)
    # 혹시 값이 grid_size 이상인 경우를 방지 (예: 14가 나오지 않도록)
    grid_samples = np.clip(grid_samples, 0, grid_size-1)
    res = []
    for g in grid_samples:
        res.append(grid_size*g[0] + g[1])
    return res

def fixed_grid_sampling(grid_size, percentage):
    """
    2차원 그리드에서 특정 퍼센티지에 따라 샘플링된 패치 인덱스를 반환합니다.
    
    Args:
        grid_size: 그리드의 크기 (N x N)
        percentage: 샘플링 비율 (25 또는 50만 지원)
    
    Returns:
        샘플링된 패치 인덱스 리스트
    """
    if percentage not in [25, 50]:
        raise ValueError("Only 25% and 50% sampling are supported.")
    
    sampled_indices = []
    
    if percentage == 50:
        # 체스보드 패턴
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0:
                    sampled_indices.append(i * grid_size + j)
    
    elif percentage == 25:
        # 2x2 블록에서 하나의 셀만 선택
        for i in range(0, grid_size, 2):
            for j in range(0, grid_size, 2):
                sampled_indices.append(i * grid_size + j)
    
    return sampled_indices

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        drop_rate_ub=None,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used
        self.drop_rate_ub = drop_rate_ub

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in self.encoder.name:
            if type(image_size) == int:
                decoder_scale = 16  # from vqvae
                num_side_patches = image_size // decoder_scale
                self.encoder_image_size = num_side_patches * encoder.patch_size
                assert self.encoder_image_size == 224, f"self.encoder_image_size {self.encoder_image_size} not supported."
                self.encoder_transform = transforms.Compose(
                    [transforms.Resize(self.encoder_image_size)]
                )
            elif len(image_size) == 2:
                decoder_scale = 14
                num_side_patches = (image_size[0] // decoder_scale, image_size[1] // decoder_scale)
                self.encoder_image_size = (num_side_patches[0] * encoder.patch_size, num_side_patches[1] * encoder.patch_size)
                self.encoder_transform = transforms.Compose(
                    [transforms.Resize(self.encoder_image_size)]
                )
            print("INFO: self.encoder_image_size: ", self.encoder_image_size)
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z
    
    def encode_act(self, act):
        
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        # print("INFO: visual.shape: ", visual.shape)
        visual = self.encoder_transform(visual)
        # print("INFO: visual.shape: "    , visual.shape)
        # print("INFO: visual.shape: ", visual.shape)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)
        # print("INFO: visusal_embs.shape: ", visual_embs.shape)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)

        # print("INFO: visual_embs.shape: ", visual_embs.shape, visual_embs.min(), visual_embs.max(), visual_embs.mean())

        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        if num_patches == 1:
            z_obs['visual'] = z_obs['visual'].repeat(1, 1, int(self.encoder_image_size/self.encoder.patch_size)**2, 1)
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)
        
        if self.drop_rate_ub is not None:
            drop_rate_ub = self.drop_rate_ub
            _, num_hist, num_patches, _ = z_src.shape
            assert num_patches == 196, f"num_patches {num_patches} not supported."
            drop_patches_num_ub = int(num_patches * drop_rate_ub)
            drop_patches_num = torch.randint(1, drop_patches_num_ub + 1, (1,)).item()
            drop_patches_idx = torch.randperm(num_patches)[:drop_patches_num]
            if num_hist > 1:
                drop_patches_idx_frames = torch.cat([drop_patches_idx + i * num_patches for i in range(num_hist)])
            else:
                drop_patches_idx_frames = drop_patches_idx
            # print("INFO: drop_patches_num, drop_patches_idx: ", drop_patches_num, drop_patches_idx)
            keep_patches_idx_frames = torch.tensor([i for i in range(num_patches * num_hist) if i not in drop_patches_idx_frames])
            # print("INFO: keep_patches_idx_frames: ", keep_patches_idx_frames)
            tmp_mask_matrix = generate_mask_matrix(num_patches, num_hist).cuda()
            tmp_mask_matrix[:, :, drop_patches_idx_frames.unsqueeze(1), keep_patches_idx_frames] = 0
            tmp_mask_matrix[:, :, keep_patches_idx_frames.unsqueeze(1), drop_patches_idx_frames] = 0
            # print("INFO: drop_patches_idx_frames: ", drop_patches_idx_frames)
            # print("INFO: keep_patches_idx_frames: ", keep_patches_idx_frames)
            # print("INFO: tmp_mask_matrix: ", tmp_mask_matrix)
            # print("INFO: tmp_mask_matrix: ", tmp_mask_matrix)
            for layer in self.predictor.module.transformer.layers:
                # print("INFO: bias before: ", layer[0].bias, layer[0].bias.dtype)
                layer[0].bias = tmp_mask_matrix
                # print("INFO: bias after: ", layer[0].bias, layer[0].bias.dtype)


                
        if self.predictor is not None:

            z_pred = self.predict(z_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )
            # if drop_rate_ub is None:
            #     if self.concat_dim == 0:
            #         z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
            #         z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
            #         z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            #     elif self.concat_dim == 1:
            #         z_visual_loss = self.emb_criterion(
            #             z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
            #             z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
            #         )
            #         z_proprio_loss = self.emb_criterion(
            #             z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
            #             z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
            #         )
            #         z_loss = self.emb_criterion(
            #             z_pred[:, :, :, :-self.action_dim], 
            #             z_tgt[:, :, :, :-self.action_dim].detach()
            #         )
            # else:
            #     if self.concat_dim == 0:
            #         # TODO
            #         z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
            #         z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
            #         z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            #     elif self.concat_dim == 1:
            #         z_visual_loss = self.emb_criterion(
            #             z_pred[:, :, keep_patches_idx_frames, :-(self.proprio_dim + self.action_dim)], \
            #             z_tgt[:, :, keep_patches_idx_frames, :-(self.proprio_dim + self.action_dim)].detach()
            #         )
            #         z_proprio_loss = self.emb_criterion(
            #             z_pred[:, :, keep_patches_idx_frames, -(self.proprio_dim + self.action_dim): -self.action_dim], 
            #             z_tgt[:, :, keep_patches_idx_frames, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
            #         )
            #         z_loss = self.emb_criterion(
            #             z_pred[:, :, keep_patches_idx_frames, :-self.action_dim], 
            #             z_tgt[:, :, keep_patches_idx_frames, :-self.action_dim].detach()
            #         )


            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def rollout(self, obs_0, act, eval_actions=False):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z

    def get_vit_patch_coordinates(self, num_patches=196, coord_min=-1, coord_max=1):
        grid_size = int(np.sqrt(num_patches))

        patch_size = (coord_max - coord_min) / grid_size  # 패치 하나의 크기

        # 14개의 x, y 좌표 만들기 (중심점 기준)
        x_coords = torch.linspace(coord_min + patch_size / 2, coord_max - patch_size / 2, grid_size)
        y_coords = torch.linspace(coord_min + patch_size / 2, coord_max - patch_size / 2, grid_size)

        # 좌표 meshgrid 생성 (14x14)
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')

        # 좌표를 (196, 2) 형태로 변환
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        coords = coords.numpy()

        return coords


class VWorldModelDrop(VWorldModel):
    """
    Only used in planning!!!!
    """
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        plan_num_clusters,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        fixed_drop=False,
        grid_drop=False,
    ):
        super().__init__(
            image_size,
            num_hist,
            num_pred,
            encoder,
            proprio_encoder,
            action_encoder,
            decoder,
            predictor,
            proprio_dim,
            action_dim,
            concat_dim,
            num_action_repeat,
            num_proprio_repeat,
            train_encoder,
            train_predictor,
            train_decoder,
        )
        self.num_clusters = plan_num_clusters
        print("INFO: VWorldModelDrop initialized with plan_num_clusters: ", self.num_clusters)
        self.centroids = None
        self.prev_object_group = None
        self.num_features = 1
        self.patch_num = 196
        if grid_drop:
            if not fixed_drop:
                print("INFO: Grid Drop (LHS)")
                self.keep_patches_idx = torch.tensor(latin_hypercube_sampling(self.num_clusters, grid_size=np.sqrt(self.patch_num))).int()
                self.drop_patches_idx = torch.tensor([i for i in range(self.patch_num) if i not in keep_patches_idx])
            else:
                print("INFO: fixed grid drop")
                percentage = int(self.num_clusters / self.patch_num * 100)
                assert self.num_clusters in [49, 98], "Only 25 and 50 sampling are supported."
                keep_patches_idx = torch.tensor(fixed_grid_sampling(14, percentage)).int()
                drop_patches_idx = torch.tensor([i for i in range(self.patch_num) if i not in keep_patches_idx])
                self.keep_patches_idx = keep_patches_idx
                self.drop_patches_idx = drop_patches_idx
        else:
            self.keep_patches_idx = torch.randperm(self.patch_num)[:self.num_clusters]
            self.drop_patches_idx = torch.tensor([i for i in range(self.patch_num) if i not in self.keep_patches_idx])


        
        print("INFO: drop_patches_idx: ", self.drop_patches_idx)
        tmp_mask_matrix = generate_mask_matrix(self.patch_num, num_hist).cuda()
        try:
            for layer in self.predictor.module.transformer.layers:
                # print("INFO: bias before: ", layer[0].bias, layer[0].bias.dtype)
                layer[0].bias = tmp_mask_matrix
        except:
            for layer in self.predictor.transformer.layers:
                layer[0].bias = tmp_mask_matrix
        # self.keep_patches_idx = torch.expand(self.keep_patches_idx, (1000, len(self.keep_patches_idx)))
        # self.drop_patches_idx = torch.expand(self.drop_patches_idx, (1000, len(self.drop_patches_idx)))
    
    def encode_obs(self, obs):
        video_norm = obs['visual'] # (B, T, 3, H, W)
        # print("INFO: video_norm.shape: ", video_norm.shape)
        b = video_norm.shape[0]

        # frames_bt = visual.permute([0, 1, 3, 4, 2]) # (B, T, H, W, C)
        B, T, C, H, W = video_norm.shape
        
        video_norm = rearrange(video_norm, "b t c h w -> (b t) c h w")
        
        z = self.encoder(video_norm) # (B, T, num_patches, encoder_emb_dim)
        z = rearrange(z, "(b t) n d -> b t n d", b=b)
        # print("INFO: z.shape: ", z.shape)
        # patch_num = z.shape[2] # patch number

        # z, mask, ids_restore, ids_keep = self.random_masking(z, masking_ratio)
        # for visualization
        # np.random.seed(72)
        # color_map = [np.concatenate([np.random.random(3), [1.0]]) for _ in range(500)]
        # grid_size = H // self.encoder.patch_size
        # step_y = H // grid_size
        # step_x = W // grid_size
        # patch_h = int(step_y)
        # patch_w = int(step_x)
        # token_positions = np.array([
        #     (int(y * step_y + step_y//2), int(x * step_x + step_x//2))
        #     for y in range(grid_size) for x in range(grid_size)
        # ])
        # overlay = np.zeros((B, T, H, W, 4))
        # for token_idx in ids_keep:
        #     y, x = token_positions[token_idx]
        #     color_bgr = [0., 1., 0., 1.,]
            
        #     overlay[
        #         :,
        #         :,
        #         max(0, y-patch_h//2):min(H, y+patch_h//2),
        #         max(0, x-patch_w//2):min(W, x+patch_w//2)
        #     ] = color_bgr

        
        

        # assert z.ndim == 4, f"visual shape should be (b, t, num_objects, encoder_emb, dim), {visual.shape} not supported."
        # assert z.shape[2] == self.num_clusters, f"z.shape[2]: {z.shape[2]} should be equal to self.num_clusters: {self.num_clusters}"

        proprio = obs['proprio']
        # print("INFO: z.shape: ", z.shape)
        proprio_emb = self.encode_proprio(proprio)
        # print(overlay)


        # res = {"visual": z, "proprio": proprio_emb, 'overlay': torch.tensor(overlay)}
        # return res
        return {"visual": z, "proprio": proprio_emb}


    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        # z = self.predictor(z, cluster_labels)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def encode(self, obs, act, masking_ratio=0.0):
        """
        input : obs (dict): "visual" (b, num_frames, num_objects, emb_dim), "proprio"
        output: z (tensor): (b, num_frames, num_objects, emb_dim)
        """
        z_dct = self.encode_obs(obs, masking_ratio)
        act_emb = self.encode_act(act)
        z_dct['action'] = act_emb
        z_dct, mask, ids_restore, ids_keep = self.random_masking(z_dct, masking_ratio)
        act_emb = z_dct['action']
        del z_dct['action']
        assert 'action' not in z_dct, f"z_dct should not have action: {z_dct.keys()}"

        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)

             
        for k in z_dct.keys():
            print(f"INFO: z_dct[{k}].shape: ", z_dct[k].shape)
        return z, mask, ids_restore, ids_keep

    def encodeEx(self, obs, act, mask, attn_mask):
        """
        input : obs (dict): "visual" (b, num_frames, num_objects, emb_dim), "proprio"
        output: z (tensor): (b, num_frames, num_objects, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        N, L, D = z_dct['visual'].shape
        
        z_dct['action'] = act_emb
        # z_dct, mask, ids_restore, ids_keep = self.random_masking(z_dct, masking_ratio)
        for k in z_dct.keys():
            z_dct[k] = z_dct[k][~mask].reshape(N, -1, D)
        act_emb = z_dct['action']
        del z_dct['action']
        assert 'action' not in z_dct, f"z_dct should not have action: {z_dct.keys()}"

        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)

             
            print("INFO: z.shape: ", z.shape)
        return z
    
    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        z, mask, ids_restore, ids_keep = self.encode(obs, act, masking_ratio=0.0)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)
    
    
        if self.predictor is not None:

            z_pred = self.predict(z_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )
          


            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        # return z_pred, visual_pred, visual_reconstructed, loss, loss_components
        return z_src, z_pred, mask, ids_restore

    def forwardEx(self, z):
        """
        forward with given encoded obses
        """
        loss = 0
        loss_components = {}
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)

        print("INFO: z_src.shape in forwardEx: ", z_src.shape)
        
    
        if self.predictor is not None:

            z_pred = self.predict(z_src)

        else:
            z_pred = None



        return z_tgt, z_pred

    def rollout(self, obs_0, act, eval_actions=False):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            # z_pred = self.predict(z[:, -self.num_hist :], cluster_labels[:, -self.num_hist :])
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        # z_pred = self.predict(z[:, -self.num_hist :], cluster_labels[:, -self.num_hist :])
        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x['visual'].shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        x_masked = {}

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # keep the visible patches to be sorted
        temp, _ = torch.sort(ids_shuffle[:, :len_keep], dim=-1)
        ids_shuffle[:, :len_keep] = temp

        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        for k in x.keys():
            # shuffle the sequence
            x_masked[k] = torch.gather(x[k], dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

def load_ckpt(snapshot_path, device):
    print("INFO: snapshot_path: ", snapshot_path)
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    print(f"Checkpoint keys: {payload.keys()}")
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


# def load_model(model_ckpt, train_cfg, num_action_repeat, device, plan_num_clusters=None, drop=False, sample_random_drop=False, fixed_drop=False, grid_drop=False):
def load_model(args):
    model_ckpt = args.model_ckpt
    num_action_repeat = 1
    plan_num_clusters = -1
    drop = True
    fixed_drop = False
    grid_drop = False
    train_cfg_path = os.path.join(os.path.dirname(os.path.dirname(args.model_ckpt)), 'hydra.yaml')
    with open(train_cfg_path, 'r') as f:
        train_cfg = OmegaConf.load(f)
    model_ckpt = Path(model_ckpt)
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, 'cpu')
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    print("INFO: train_cfg.encoder: ", train_cfg.encoder)
    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    print("INFO: result.keys(): ", result.keys())
    print("INFO: result.action_encoder: ", result['action_encoder'])

    if plan_num_clusters is not None:
        if 'VOWorldModel' in train_cfg.model._target_:
            result['predictor'].num_patches = plan_num_clusters
            result['predictor'].for_planning = True
            for layer in result['predictor'].transformer.layers:
                layer[0].bias = generate_mask_matrix(plan_num_clusters, train_cfg.num_hist).cuda()
        elif 'VWorldModel' in train_cfg.model._target_:
            result['predictor'].plan_num_clusters = plan_num_clusters
            result['predictor'].num_frames = train_cfg.num_hist

    train_cfg_cpy = train_cfg.copy()
    train_cfg_cpy.model._target_ = train_cfg_cpy.model._target_.replace('VWorldModel', 'VWorldModelDrop')
    train_cfg_cpy.has_decoder = False

    print("INFO: result.keys(): ", result.keys())
    model = hydra.utils.instantiate(
        train_cfg_cpy.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=None,
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,  
        plan_num_clusters=plan_num_clusters,
        grid_drop=grid_drop,
        fixed_drop=fixed_drop,
    )

    return model

dino_wm = load_model