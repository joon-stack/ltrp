import torch
import decord
import numpy as np
from pathlib import Path
from typing import Callable, Optional
decord.bridge.set_bridge("torch")

import numpy as np

import abc
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Sequence, List
from torch.utils.data import Dataset, Subset
from torch import default_generator, randperm
from einops import rearrange

# precomputed dataset stats
ACTION_MEAN = torch.tensor([0.0006, 0.0015])
ACTION_STD = torch.tensor([0.4395, 0.4684])
STATE_MEAN = torch.tensor([0.7518, 0.9239, -3.9702e-05, 3.1550e-04])
STATE_STD = torch.tensor([1.0964, 1.2390, 1.3819, 1.5407])

from torchvision import transforms
import torch

def default_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )

def imagenet_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

# def normalize(x: torch.Tensor) -> torch.Tensor:
#     if x.max() > 1.0:
#         x = x.float() / 255.0
#     # print("INFO: x.min, x.max: ", x.min(), x.max())
#     if x.shape[-1] == 3:
#         x = x.permute(2, 0, 1)
#     # print("INFO: x.shape: ", x.shape)
#     normalize = transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     return normalize(x)

# https://github.com/JaidedAI/EasyOCR/issues/1243
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

class TrajDataset(Dataset, abc.ABC):
    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

class TrajSubset(TrajDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset: TrajDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class TrajSlicerDataset(TrajDataset):
    def __init__(
        self,
        dataset: TrajDataset,
        num_frames: int,
        frameskip: int = 1,
        process_actions: str = "concat",
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.slices = []
        for i in range(len(self.dataset)): 
            T = self.dataset.get_seq_length(i)
            if T - num_frames < 0:
                print(f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}")
            else:
                self.slices += [
                    (i, start, start + num_frames * self.frameskip)
                    for start in range(T - num_frames * frameskip + 1)
                ]  # slice indices follow convention [start, end)
        # randomly permute the slices
        self.slices = np.random.permutation(self.slices)
        
        self.proprio_dim = self.dataset.proprio_dim
        if process_actions == "concat":
            self.action_dim = self.dataset.action_dim * self.frameskip
        else:
            self.action_dim = self.dataset.action_dim

        self.state_dim = self.dataset.state_dim


    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        obs, act, state, _ = self.dataset[i]
        for k, v in obs.items():
            obs[k] = v[start:end:self.frameskip]
        state = state[start:end:self.frameskip]
        act = act[start:end]
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)  # concat actions
        return tuple([obs, act, state])


def random_split_traj(
    dataset: TrajDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajSubset]:
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    print(
        [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]
    )
    return [
        TrajSubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def get_train_val_sliced(
    traj_dataset: TrajDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    num_frames: int = 10,
    frameskip: int = 1,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    train_slices = TrajSlicerDataset(train, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val, num_frames, frameskip)
    return train, val, train_slices, val_slices

class WallDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/wall_single",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        print("Loading wall dataset from self.data_path")
        states = torch.load(self.data_path / "states.pth")
        self.states = states
        self.proprios = self.states.clone()
        self.actions = torch.load(self.data_path / "actions.pth")
        self.actions = self.actions / action_scale
        self.door_locations = torch.load(self.data_path / "door_locations.pth")
        self.wall_locations = torch.load(self.data_path / "wall_locations.pth")

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)
            print(f"Loaded {n} rollouts")

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.proprios = self.proprios[:n]
        self.door_locations = self.door_locations[:n]
        self.wall_locations = self.wall_locations[:n]

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]
        self.traj_len = self.actions.shape[1]
        if normalize_action:
            self.action_mean = self.actions.mean(dim=(0, 1))
            self.action_std = self.actions.std(dim=(0, 1))
            self.state_mean = self.states.mean(dim=(0, 1))
            self.state_std = self.states.std(dim=(0, 1))
            self.proprio_mean = self.proprios.mean(dim=(0, 1))
            self.proprio_std = self.proprios.std(dim=(0, 1))
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.traj_len

    def get_all_actions(self):
        result = []
        for i in range(len(self.states)):
            T = self.get_seq_length(i)
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        obs_dir = self.data_path / "obses"
        image = torch.load(obs_dir / f"episode_{idx:03d}.pth")
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        door_location = self.door_locations[idx, frames]
        wall_location = self.wall_locations[idx, frames]

        image = image[frames] / 255 
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image,"proprio": proprio}
        return obs, act, state, {'fix_door_location': door_location[0], 'fix_wall_location': wall_location[0]}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return self.states.shape[0] if not self.n_rollout else self.n_rollout
    
    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return imgs


def load_wall_slice_train_val(
    transform,
    n_rollout=50,
    data_path='data/wall_single',
    normalize_action=False,
    split_ratio=0.8,
    split_mode="random",
    num_hist=0,
    num_pred=0,
    frameskip=0,
    object=None,
    encoder=None,
    num_clusters=4,
    num_features=3,
    use_coord=False,
    use_patch_info=False,
    folder_name='ft',
):  
    if split_mode == "random":

        dset = WallDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path,
            normalize_action=normalize_action,
        )
        dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
            traj_dataset=dset, 
            train_fraction=split_ratio, 
            num_frames=num_hist + num_pred, 
            frameskip=frameskip
        )
    elif split_mode == "folder":
        
        dset_train = WallDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path + "/train",
            normalize_action=normalize_action,
        )
        dset_val = WallDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path + "/val",
            normalize_action=normalize_action,
        )
        num_frames = num_hist + num_pred
        train_slices = TrajSlicerDataset(dset_train, num_frames, frameskip)
        val_slices = TrajSlicerDataset(dset_val, num_frames, frameskip)

    datasets = {}
    datasets['train'] = train_slices # for compatibility with other datasets
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset
