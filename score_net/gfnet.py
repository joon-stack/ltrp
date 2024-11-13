import PIL.Image as Image
import score_net.gfnet.gfnet_resnet as resnet
from score_net.gfnet.gfnet_utils import *
import torch.multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F


class gf_net(nn.Module):
    def __init__(self):
        super(gf_net, self).__init__()

    def load(self, checkpoint):
        model = resnet.resnet50(pretrained=False)
        model_prime = resnet.resnet50(pretrained=False)

        model_configuration = {
            'feature_num': 2048,
            'feature_map_channels': 2048,
            'policy_conv': False,
            'policy_hidden_dim': 1024,
            'fc_rnn': True,
            'fc_hidden_dim': 1024,
            'image_size': 224,
            'crop_pct': 0.875,
            'dataset_interpolation': Image.BILINEAR,
            'prime_interpolation': 'bicubic'
        }

        model_arch = checkpoint['model_name']
        patch_size = checkpoint['patch_size']
        prime_size = checkpoint['patch_size']
        maximum_length = len(checkpoint['flops'])
        state_dim = model_configuration['feature_map_channels'] * math.ceil(patch_size / 32) * math.ceil(
            patch_size / 32)
        policy = ActorCritic(model_configuration['feature_map_channels'], state_dim,
                             model_configuration['policy_hidden_dim'],
                             model_configuration['policy_conv'])
        fc = Full_layer(model_configuration['feature_num'], model_configuration['fc_hidden_dim'],
                        model_configuration['fc_rnn'])

        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_prime_state_dict'].items()}
        model_prime.load_state_dict(state_dict)
        fc.load_state_dict(checkpoint['fc'])
        policy.load_state_dict(checkpoint['policy'])
        memory = Memory()

        self.memory = memory
        self.fc = fc
        self.policy = policy
        self.model = model
        self.model_prime = model_prime
        self.patch_size = patch_size
        self.prime_size = prime_size
        self.maximum_length = maximum_length
        self.model_configuration = model_configuration
        self.model_arch = model_arch
        temp = torch.tensor(range(0, 196)).cuda()
        temp = torch.reshape(temp, [14, 14])
        self.idx = temp

    def get_prime(self, images, patch_size, interpolation='bicubic'):
        """Get down-sampled original image"""
        prime = F.interpolate(images, size=[patch_size, patch_size], mode=interpolation, align_corners=True)
        return prime

    def get_patch(self, images, action_sequence, patch_size):
        """Get small patch of the original image"""
        batch_size = images.size(0)
        image_size = images.size(2)

        patch_coordinate = torch.floor(action_sequence * (image_size - patch_size)).int()
        patches = []
        for i in range(batch_size):
            per_patch = images[i, :,
                        (patch_coordinate[i, 0].item()): ((patch_coordinate[i, 0] + patch_size).item()),
                        (patch_coordinate[i, 1].item()): ((patch_coordinate[i, 1] + patch_size).item())]

            patches.append(per_patch.view(1, per_patch.size(0), per_patch.size(1), per_patch.size(2)))

        return torch.cat(patches, 0)

    def get_patchEx(self, images, action_sequence):
        """Get small patch of the original image"""
        batch_size = images.size(0)
        patch_coordinate = torch.floor(action_sequence * (14 - 8)).int()
        patches = []

        for i in range(batch_size):
            per_patch = self.idx[
                        (patch_coordinate[i, 0].item()): ((patch_coordinate[i, 0] + 8).item()),
                        (patch_coordinate[i, 1].item()): ((patch_coordinate[i, 1] + 8).item())]
            per_patch = per_patch.flatten()
            patches.append(per_patch.view(1, per_patch.size(0)))

        return torch.cat(patches, 0)

    def get_visible_tokens_idx(self, imgs, len_keep):
        self.memory.clear_memory()
        self.fc.hidden = None
        input_prime = self.get_prime(imgs, self.prime_size, self.model_configuration['prime_interpolation'])
        output, state = self.model_prime(input_prime)
        for patch_step in range(1, self.maximum_length):
            with torch.no_grad():
                if patch_step == 1:
                    action = self.policy.act(state, self.memory, restart_batch=True)
                else:
                    action = self.policy.act(state, self.memory)
                if patch_step == (self.maximum_length - 1):
                    patches = self.get_patchEx(imgs, action)
                else:
                    patches = self.get_patch(imgs, action, self.patch_size)
                    output, state = self.model(patches)
                    output = self.fc(output, restart=False)
        ids_keep = patches[:, -len_keep:]

        return ids_keep
