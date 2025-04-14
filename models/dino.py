import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        # print("INFO: x.shape: ", x.shape)
        # print("INFO: emb.shape: ", emb.shape)
        return emb


class DinoEncoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load('facebookresearch/dino:main', name)  # 16은 patch_size
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        # self.patch_size = self.base_model.patch_size
        self.patch_size = 16

    def forward(self, x):
        emb = self.base_model.get_intermediate_layers(x, n=1)[0]
        # print("INFO: x.shape: ", x.shape)
        # print("INFO: emb.shape: ", emb.shape)
        if self.latent_ndim == 1:
            emb = emb[:, 0]
            emb = emb.unsqueeze(1) # dummy patch dim
        else:
            emb = emb[:, 1:]
        return emb

class DinoV2EncoderReg(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size


    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        # print("INFO: x.shape: ", x.shape)
        # print("INFO: emb.shape: ", emb.shape)
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb