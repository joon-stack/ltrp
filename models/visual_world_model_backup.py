import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from scipy.optimize import linear_sum_assignment # for Hungarian algorithm
import numpy as np

from .SOLV import SOLV

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
        # print("INFO: visual.shape: ", visual.shape)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

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


    def rollout(self, obs_0, act):
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



class VOWorldModel(VWorldModel):
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
        num_clusters,
        num_features,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        pos_dim=0,
        pos_scale=10,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        use_coord=False,
        use_patch_info=True,
        
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
        self.num_clusters = num_clusters
        self.pos_dim = pos_dim
        self.pos_scale = pos_scale
        self.num_features = num_features
        self.prev_object_groups = None
        self.use_coord = use_coord
        self.use_patch_info = use_patch_info
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) + self.pos_dim
        print(f"emb_dim of object-centric model: {self.emb_dim}")
    
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

    def encode_pos(self, pos):
        pos = self.pos_encoder(pos)
        return pos

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio", "z" (b, t, num_objects, encoder_emb_dim)
        output:   z (dict): "proprio", "z" (b, t, num_objects, encoder_emb_dim)
        """
        if 'z' in obs:
            z = obs['z']
            pos_emb = obs['pos'] * self.pos_scale
            overlays_z = obs['pos'] # dummy
            object_groups = obs['pos'] # dummy
        else:
            visual = obs['visual'] # (B, T, 3, H, W)
            video_norm = visual
            frames_bt = visual.permute([0, 1, 3, 4, 2]) # (B, T, H, W, C)
            B, T, C, H, W = video_norm.shape
            
            video_norm = rearrange(video_norm, "b t c h w -> (b t) c h w")
            
            z = self.encoder(video_norm) # (B, T, num_objects, encoder_emb_dim)
            z = rearrange(z, "(b t) n d -> b t n d", b=visual.shape[0])
            patch_num = z.shape[2] # patch number
            coords = self.get_vit_patch_coordinates(num_patches=patch_num)

            # for visualization
            np.random.seed(72)
            color_map = [np.concatenate([np.random.random(3), [1.0]]) for _ in range(100)]

            object_z = []
            coords_z = []
            overlays_bt = []
            patch_z = []
            for x, b in enumerate(z):
                object_b = []
                coords_b = []
                patch_b = []

                # for visualization
                frames_t = frames_bt[x]
                overlays_t = []

                for i, features in enumerate(b):
                    # for visualization
                    frame = frames_t[i]
                    grid_size = H // self.encoder.patch_size
                    step_y = H // grid_size
                    step_x = W // grid_size
                    token_positions = np.array([
                        (int(y * step_y + step_y//2), int(x * step_x + step_x//2))
                        for y in range(grid_size) for x in range(grid_size)
                    ])
        

                    overlay = np.ones((frame.shape[0], frame.shape[1], 4))
                    overlay[:, :, 3] = 0
                    patch_h = int(step_y)
                    patch_w = int(step_x)

                    # for clustering
                    features = features.detach().cpu().numpy()  # (num_objects, feature_dim)    
                    distance_matrix = cosine_distances(features)
                    clustering = AgglomerativeClustering(
                        n_clusters=self.num_clusters, 
                        metric='precomputed', 
                        linkage='average',
                        distance_threshold=None,
                    )
                    cluster_labels = clustering.fit_predict(distance_matrix)
                    object_groups = []
                    for cluster_id in np.unique(cluster_labels):
                        group = np.where(cluster_labels == cluster_id)[0].tolist()
                        object_groups.append(group)

                    
                    if self.prev_object_groups is None:
                        object_groups.sort(key=len, reverse=True)
                        if self.num_clusters == 6:
                            tmp = object_groups[3]
                            object_groups[3] = object_groups[4]
                            object_groups[4] = tmp
                            if 0 not in object_groups[0]:
                                tmp = object_groups[0]
                                object_groups[0] = object_groups[1]
                                object_groups[1] = tmp
                            
                        # object_groups.sort()
                        # tmp = object_groups[0]
                        # object_groups[0] = object_groups[1]
                        # object_groups[1] = tmp
                        # print("INFO: object_groups: ", object_groups)
                        # print("INFO: object_groups sorted!!")

                    # 프레임 간 클러스터 순서 정렬
                    if self.prev_object_groups is not None:
                        object_groups.sort(key=len, reverse=True)
                        reordered_groups = []
                        used_indices = set()

                        for prev_group in self.prev_object_groups:
                            # 현재 프레임의 그룹과 이전 프레임의 그룹 간의 유사도를 비교
                            best_match_idx = -1
                            best_overlap = 0

                            for i, curr_group in enumerate(object_groups):
                                if i in used_indices:
                                    continue  # 이미 매칭된 그룹은 건너뜀
                                
                                # overlap = len(set(prev_group) & set(curr_group)) / min(len(prev_group), len(curr_group)) # 교집합 크기 계산 
                                overlap = len(set(prev_group) & set(curr_group))

                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_match_idx = i

                            if best_match_idx != -1:
                                reordered_groups.append(object_groups[best_match_idx])
                                used_indices.add(best_match_idx)

                        # 아직 매칭되지 않은 그룹들은 그대로 추가
                        for i, group in enumerate(object_groups):
                            if i not in used_indices:
                                reordered_groups.append(group)

                        object_groups = reordered_groups  # 업데이트

                    self.prev_object_groups = object_groups

                    frame_features = []
                    frame_coords = []
                    frame_patch_info = []
                    for group_idx, group in enumerate(object_groups):
                        group = sorted(group)

                        # for visualizations
                        for token_idx in group:
                            y, x = token_positions[token_idx]
                            color_bgr = color_map[group_idx] 
                            
                            overlay[
                                max(0, y-patch_h//2):min(H, y+patch_h//2),
                                max(0, x-patch_w//2):min(W, x+patch_w//2)
                            ] = color_map[group_idx]

                        # for clustering features
                        cluster_features = features[group] # (num_patches_in_cluster, feature_dim)
                        cluster_coords = coords[group]

                        cluster_feature_chunks = np.array_split(cluster_features, self.num_features)
                        cluster_coord_chunks = np.array_split(cluster_coords, self.num_features)
                        patch_chunks = np.array_split(np.array(group), self.num_features)

                        for j, feature_chunk in enumerate(cluster_feature_chunks):
                            k = j
                            feature_chunk = cluster_feature_chunks[k]
                            coord_chunk = cluster_coord_chunks[k]
                            patch_chunk = patch_chunks[k]

                            while feature_chunk.shape[0] == 0:
                                # print("group : ", group)
                                k -= 1
                                feature_chunk = cluster_feature_chunks[k]
                                coord_chunk = cluster_coord_chunks[k]
                                patch_chunk = patch_chunks[k]
                            
                            avg_feature = np.mean(feature_chunk, axis=0)
                            frame_features.append(avg_feature)

                            avg_coord = np.mean(coord_chunk, axis=0)
                            std_coord = np.std(coord_chunk, axis=0)
                            avg_coord = np.concatenate([avg_coord, std_coord])
                            frame_coords.append(avg_coord)

                            patch_info_chunk = np.zeros(patch_num).astype(float)
                            patch_info_chunk[patch_chunk] = 1.0
                            frame_patch_info.append(patch_info_chunk)

                    frame_features = np.array(frame_features) # (num_objects, feature_dim)
                    frame_coords = np.array(frame_coords) # (num_objects, 4)
                    frame_patch_info = np.array(frame_patch_info) # (num_objects, num_patches)
                    object_b.append(frame_features)
                    coords_b.append(frame_coords)
                    overlays_t.append(overlay)
                    patch_b.append(frame_patch_info)
                   
                object_b = np.array(object_b) # (T, num_objects, feature_dim)
                coords_b = np.array(coords_b) # (T, num_objects, 4)
                patch_b = np.array(patch_b) # (T, num_objects, num_patches) 
                object_z.append(object_b)
                coords_z.append(coords_b)
                overlays_t = np.array(overlays_t) # (T, H, W, 4)
                overlays_bt.append(overlays_t)
                patch_z.append(patch_b)

            object_z = np.array(object_z) # (B, T, num_objects, feature_dim)    
            coords_z = np.array(coords_z) # (B, T, num_objects, 4) or (B, T, num_objects, num_patches)
            patch_z = np.array(patch_z) # (B, T, num_objects, num_patches)
            z = torch.tensor(object_z, device=visual.device)

            if self.use_coord:
                coords = torch.tensor(coords_z, device=visual.device)
            elif self.use_patch_info:
                coords = torch.tensor(patch_z, device=visual.device)

            pos_emb = coords * self.pos_scale
            pos_emb = pos_emb.to(torch.float32)

            overlays_z = np.array(overlays_bt) # (B, T, H, W, 4)
            overlays_z = torch.tensor(overlays_z, device=visual.device)


        assert z.ndim == 4, f"visual shape should be (b, t, num_objects, encoder_emb, dim), {visual.shape} not supported."

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)

        return {"z": z, "proprio": proprio_emb, 'pos': pos_emb, 'overlay': overlays_z}

    def encode(self, obs, act):
        """
        input : obs (dict): "visual" (b, num_frames, num_objects, emb_dim), "proprio"
        output: z (tensor): (b, num_frames, num_objects, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['z'], z_dct['proprio'].unsqueeze(2), z_dct['pos'], act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_objects + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['z'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['z'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            # print("INFO: z_dct['z'].dtype, proprio_repeated.dtype, z_dct['pos'].dtype, act_repeated.dtype: ", z_dct['z'].dtype, proprio_repeated.dtype, z_dct['pos'].dtype, act_repeated.dtype)
            z = torch.cat(
                [z_dct['z'], proprio_repeated, z_dct['pos'], act_repeated], dim=3
            )  # (b, num_frames, num_objects, dim + 10 + 10 + 4) or (b, num_frames, num_objects, dim + 10 + 10 + 196)
            print("INFO: z.shape: ", z.shape)
            print("INFO: self.num_proprio_repeat: ", self.num_proprio_repeat)
            print("INFO: self.num_action_repeat: ", self.num_action_repeat)
            print("INFO: z_dct['z'].shape: ", z_dct['z'].shape)
            print("INFO: proprio_repeated.shape: ", proprio_repeated.shape)
            print("INFO: z_dct['pos'].shape: ", z_dct['pos'].shape)
            print("INFO: act_repeated.shape: ", act_repeated.shape)
        return z
    
    # only work with visual embeddings
    def match_objects(self, tgt, pred):
        """
        input : tgt, pred : (b, num_frames, num_objects, emb_dim)
        output: pred      : (b, num_frames, num_objects, emb_dim)
        """
        B, T, N, D_ = pred.shape
        pred_full = pred.clone()
        tgt_full = tgt.clone()
        
        pred = pred[:, :, :, :self.encoder.emb_dim]  # remove proprio, action, and pos dims
        tgt = tgt[:, :, :, :self.encoder.emb_dim]  # remove proprio, action, and pos dims
        
        B, T, N, D = pred.shape
        assert pred.shape[-1] == tgt.shape[-1] == 384, f"pred.shape[-1]: {pred.shape[-1]}, tgt.shape[-1]: {tgt.shape[-1]}"
        # 배치와 시간 차원 병합
        pred_flat = pred.reshape(B*T, N, D)
        tgt_flat = tgt.reshape(B*T, N, D)

        pred_flat_full = pred_full.reshape(B*T, N, -1)
        tgt_flat_full = tgt_full.reshape(B*T, N, -1)

        cost_matrices = torch.cdist(tgt_flat, pred_flat)  # (B*T, N, N)
        matched_pred_flat_full = torch.zeros_like(pred_flat_full)

        # 디버그용 샘플 선택 (첫 2개 배치 & 첫 시간대만 출력)
        debug_samples = min(0, B*T)  # 최대 2개 샘플만 출력


        for i in range(B*T):
            cost_matrix = cost_matrices[i].detach().cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            matched_pred_flat_full[i] = pred_flat_full[i, col_idx]

            # 디버그 정보 출력 -------------------------------------------------
            if i < debug_samples:
                batch_idx = i // T
                time_idx = i % T
                original_order = torch.arange(N).numpy()
                
                # 매칭 전후 Cost 계산
                original_cost = cost_matrix[original_order, original_order].sum()
                matched_cost = cost_matrix[row_idx, col_idx].sum()

                if original_cost > matched_cost:
                    print(f"\n=== Sample [Batch {batch_idx}, Time {time_idx}] ===")
                    print(f"Object Count: {N}")
                    print(f"Original Object Order: {original_order, original_order}")
                    print(f"Matched Object Order: {row_idx, col_idx}")
                    print(f"Cost Matrix:\n{np.round(cost_matrix, 4)}")
                    print(f"Original Cost (Diagonal): {original_cost:.4f}")
                    print(f"Matched Cost (Hungarian): {matched_cost:.4f}")
                    print(f"Cost Improvement: {original_cost - matched_cost:.4f}\n")

        res = matched_pred_flat_full.reshape(B, T, N, -1)
        return res
    
    def decode(self, z):
        """
        input :   z: ( b, num_frames, num_patches, emb_dim)
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
        b, num_frames, num_objects, emb_dim = z_obs["z"].shape
        num_patches = int((self.encoder_image_size / self.encoder.patch_size) ** 2)
        patch_info = z_obs['pos'] / self.pos_scale # (b, num_frames, num_objects, num_patches)
        assert patch_info.min().item() >= 0 and patch_info.max().item() <= 1, f"patch_info.min(): {patch_info.min().item()}, patch_info.max(): {patch_info.max().item()}"
        patch_info = torch.argmax(patch_info, dim=2) # (b, num_frames, num_patches)
        patch_info = patch_info.unsqueeze(-1).expand(-1, -1, -1, emb_dim) # (b, num_frames, num_patches, emb_dim)
        
        assembled_features = torch.gather(z_obs['z'], 2, patch_info) # (b, num_frames, num_patches, emb_dim)
        visual, diff = self.decoder(assembled_features)  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        
        obs = {
            "visual": visual,  # (b, t, 3, H, W)
            "proprio": z_obs["proprio"]
        }
        return obs, diff
        
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        # TODO: not for pos
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_pos, z_act = z[..., :-(self.proprio_dim + self.action_dim + self.pos_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim + self.pos_dim) :-(self.action_dim + self.pos_dim)],  \
                                         z[..., -(self.action_dim+self.pos_dim):-self.action_dim],  \
                                         z[..., -self.action_dim:]

            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"z": z_visual, "proprio": z_proprio, "pos": z_pos}
        return z_obs, z_act

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio", "gt" (b, num_frames, 3, img_size, img_size)
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

        if visual_src.shape[-1] == 3:
            visual_src = visual_src.permute(0, 1, 4, 2, 3)
        
        if visual_tgt.shape[-1] == 3:
            visual_tgt = visual_tgt.permute(0, 1, 4, 2, 3)


        if self.predictor is not None:
            z_pred = self.predict(z_src)
            
            # z_pred = self.match_objects(z_tgt, z_pred) # Hungarian algorithm
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual'] # (b, num_hist, num_objects, emb_dim)
                # print("INFO: visual_pred.shape, visual_tgt.shape: ", visual_pred.shape, visual_tgt.shape)
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # print(z_pred[:, :, :, -(self.action_dim + self.pos_dim): -(self.action_dim)], 
            #     z_tgt[:, :, :, -(self.action_dim + self.pos_dim): -(self.action_dim)].detach())
            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim + self.pos_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim + self.pos_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim + self.pos_dim): -(self.action_dim + self.pos_dim)], 
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim + self.pos_dim): -(self.action_dim + self.pos_dim)].detach()
                )
                z_pos_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.action_dim + self.pos_dim): -(self.action_dim)], 
                    z_tgt[:, :, :, -(self.action_dim + self.pos_dim): -(self.action_dim)].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
            loss_components["z_pos_loss"] = z_pos_loss
            # print("INFO: z_loss: ", z_loss.item())
            # print("INFO: z_visual_loss: ", z_visual_loss.item())
            # print("INFO: z_proprio_loss: ", z_proprio_loss.item())
            # print("INFO: z_pos_loss: ", z_pos_loss.item())
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


    def rollout(self, obs_0, act):
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
            # z_new = self.match_objects(z[:, -self.num_hist :], z_new) # Hungarian algorithm
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        # z_new = self.match_objects(z[:, -self.num_hist :], z_new) # Hungarian algorithm
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z
    
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
        z[:, :, :, -(self.action_dim + self.pos_dim): -(self.action_dim)] = torch.softmax(z[:, :, :, -(self.action_dim + self.pos_dim): -(self.action_dim)], dim=-2)
        return z


class VSlotWorldModel(VWorldModel):
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
        num_clusters,
        slot_ckpt,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,     
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
        self.num_clusters = num_clusters
        self.checkpoint_path = slot_ckpt
        self.encoder_image_size = image_size

        args = Args(num_clusters=self.num_clusters, checkpoint_path=self.checkpoint_path, image_size=self.encoder_image_size)
        slot_encoder = SOLV(args)
        slot_ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        new_slot_ckpt = {}
        for k, v in slot_ckpt['model'].items():
            new_slot_ckpt[k.replace('module.', '')] = v
        slot_encoder.load_state_dict(new_slot_ckpt)
        slot_encoder.cuda()
        slot_encoder.eval()
        print("INFO: SOLV loaded from checkpoint: ", self.checkpoint_path)
        self.slot_encoder = slot_encoder
        self.args = args

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio", "z" (b, t, num_objects, encoder_emb_dim)
        output:   z (dict): "proprio", "z" (b, t, num_objects, encoder_emb_dim)
        """
        if 'z' in obs:
            z = obs['z']
        else:
            # TODO for planning
            visual = obs['visual'] # (B, T, 3, H, W)
            video_norm = visual
            frames_bt = visual.permute([0, 1, 3, 4, 2]) # (B, T, H, W, C)
            B, T, C, H, W = video_norm.shape
            
            video_norm = rearrange(video_norm, "b t c h w -> (b t) c h w")
            video_norm = self.encoder_transform(video_norm)
            
            z = self.encoder(video_norm) # (B*T, num_patches, encoder_emb_dim)

            # print("INFO: num_patches: ", z.shape[1])
            # z = rearrange(z, "(b t) n d -> b t n d", b=visual.shape[0])
            num_frames = z.shape[0]
            patch_num = z.shape[1] # patch number
            
            # input z to SOLV to get object embeddings
            
            z = z.unsqueeze(1)
            z = z.expand(-1, 2*self.args.N+1, -1, -1) # (B*T, 2N+1, num_patches, encoder_emb_dim)
            z = torch.flatten(z, 0, 1) # (B*T*(2N+1), num_patches, encoder_emb_dim)
            input_masks = torch.ones(num_frames, 2*self.args.N+1).to(visual.device) # (B*T, 2N+1)
            token_indices = torch.arange(patch_num).unsqueeze(0).expand(num_frames, patch_num).unsqueeze(1).expand(-1, 2*self.args.N+1, -1).to(visual.device) # (B*T, 2N+1, token_num)
            token_indices = torch.flatten(token_indices, 0, 1) # (B*T*(2N+1), patch_num)
            # print("INFO: z.shape, input_masks.shape, token_indices.shape: ", z.shape, input_masks.shape, token_indices.shape)
            with torch.no_grad():
                reconstruction = self.slot_encoder(z, input_masks, token_indices)
            slots = reconstruction['slots'] # (B*T, slot_num, D_slot)
            z = rearrange(slots, '(b t) n d -> b t n d', b=B)
            # print("INFO: z: ", z.shape, z.min().item(), z.max().item(), z.mean().item())



        assert z.ndim == 4, f"visual shape should be (b, t, num_objects, encoder_emb, dim), {visual.shape} not supported."

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)

        return {"z": z, "proprio": proprio_emb}

    def encode(self, obs, act):
        """
        input : obs (dict): "visual" (b, num_frames, num_objects, emb_dim), "proprio"
        output: z (tensor): (b, num_frames, num_objects, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['z'], z_dct['proprio'].unsqueeze(2), z_dct['pos'], act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_objects + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['z'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['z'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            # print("INFO: z_dct['z'].dtype, proprio_repeated.dtype, z_dct['pos'].dtype, act_repeated.dtype: ", z_dct['z'].dtype, proprio_repeated.dtype, z_dct['pos'].dtype, act_repeated.dtype)
            z = torch.cat(
                [z_dct['z'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_objects, dim + 10 + 10 + 4) or (b, num_frames, num_objects, dim + 10 + 10 + 196)
        return z
    
    def decode(self, z):
        """
        input :   z: ( b, num_frames, num_patches, emb_dim)
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
        b, num_frames, num_objects, emb_dim = z_obs["z"].shape
        num_patches = int((self.encoder_image_size / self.encoder.patch_size) ** 2)
        patch_info = z_obs['pos'] / self.pos_scale # (b, num_frames, num_objects, num_patches)
        assert patch_info.min().item() >= 0 and patch_info.max().item() <= 1, f"patch_info.min(): {patch_info.min().item()}, patch_info.max(): {patch_info.max().item()}"
        patch_info = torch.argmax(patch_info, dim=2) # (b, num_frames, num_patches)
        patch_info = patch_info.unsqueeze(-1).expand(-1, -1, -1, emb_dim) # (b, num_frames, num_patches, emb_dim)
        
        assembled_features = torch.gather(z_obs['z'], 2, patch_info) # (b, num_frames, num_patches, emb_dim)
        visual, diff = self.decoder(assembled_features)  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        
        obs = {
            "visual": visual,  # (b, t, 3, H, W)
            "proprio": z_obs["proprio"]
        }
        return obs, diff
        
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        # TODO: not for pos
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-(self.action_dim)],  \
                                         z[..., -self.action_dim:]

            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"z": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio", "gt" (b, num_frames, 3, img_size, img_size)
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
        


        if self.predictor is not None:
            z_pred = self.predict(z_src)
            
            # z_pred = self.match_objects(z_tgt, z_pred) # Hungarian algorithm
            if self.decoder is not None:
                visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
                visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

                if visual_src.shape[-1] == 3:
                    visual_src = visual_src.permute(0, 1, 4, 2, 3)
                
                if visual_tgt.shape[-1] == 3:
                    visual_tgt = visual_tgt.permute(0, 1, 4, 2, 3)
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual'] # (b, num_hist, num_objects, emb_dim)
                # print("INFO: visual_pred.shape, visual_tgt.shape: ", visual_pred.shape, visual_tgt.shape)
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # print(z_pred[:, :, :, -(self.action_dim + self.pos_dim): -(self.action_dim)], 
            #     z_tgt[:, :, :, -(self.action_dim + self.pos_dim): -(self.action_dim)].detach())
            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim )], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim )].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim ): -(self.action_dim )], 
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim ): -(self.action_dim )].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
            # print("INFO: z_loss: ", z_loss.item())
            # print("INFO: z_visual_loss: ", z_visual_loss.item())
            # print("INFO: z_proprio_loss: ", z_proprio_loss.item())
            # print("INFO: z_pos_loss: ", z_pos_loss.item())
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


    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['proprio'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            # z_new = self.match_objects(z[:, -self.num_hist :], z_new) # Hungarian algorithm
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z
    
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
        # z[:, :, :, -(self.action_dim + self.pos_dim): -(self.action_dim)] = torch.softmax(z[:, :, :, -(self.action_dim + self.pos_dim): -(self.action_dim)], dim=-2)
        return z

class Args:
    def __init__(self, num_clusters, checkpoint_path, image_size):
        """
        for SOLV
        """

        self.slot_dim = 128
        self.num_slots = num_clusters
        self.merge_slots = False

        self.slot_att_iter = 3
        self.slot_merge_coeff = 0.12
        self.N = 2
        self.token_drop_ratio = 0.5
        self.encoder = 'dinov2-vits-14'
        self.use_checkpoint = True
        self.checkpoint_path = checkpoint_path 
        self.resize_to = image_size
        self.patch_size = 14
        if type(image_size) == int:
            self.token_num = (image_size // self.patch_size) ** 2
            self.resize_to = (image_size, image_size)
        elif len(image_size) == 2:
            self.token_num = (image_size[0] // self.patch_size) * (image_size[1] // self.patch_size)
        

        print("INFO: Args for SOLV: ", self.__dict__)

