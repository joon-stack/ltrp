import torch
import decord
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Union
import os  # os 모듈 추가
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

    @abc.abstractmethod
    def get_trajectory(self, idx, start, end, frameskip):
        """
        Returns a slice of the idx-th trajectory.
        """
        raise NotImplementedError

    # 기존 __getitem__은 제거하거나 에러를 발생시키도록 변경
    def __getitem__(self, idx):
        raise NotImplementedError("Use get_trajectory for TrajDataset subclasses")


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
            # 각 trajectory의 실제 길이를 고려하여 가능한 슬라이스 생성
            effective_length = (self.num_frames - 1) * self.frameskip + 1
            if T < effective_length:
                print(f"Ignored short sequence #{i}: len={T}, required_len={effective_length}")
            else:
                # 슬라이스 인덱스 생성: 마지막 프레임이 T를 넘지 않도록 함
                for start in range(T - effective_length + 1):
                    self.slices.append((i, start)) # trajectory index와 start index만 저장

        # 슬라이스 인덱스를 메모리 효율적인 numpy 배열로 저장
        self.slices = np.array(self.slices, dtype=np.int32)
        np.random.shuffle(self.slices) # numpy 배열로 섞기

        # 데이터셋 속성은 dataset 객체에서 가져오도록 변경
        self.proprio_dim = self.dataset.proprio_dim
        if process_actions == "concat":
            self.action_dim = self.dataset.action_dim * self.frameskip
        else:
            self.action_dim = self.dataset.action_dim
        self.state_dim = self.dataset.state_dim


    def get_seq_length(self, idx: int) -> int:
        # SlicerDataset의 길이는 항상 num_frames
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        # 저장된 trajectory index와 start index 가져오기
        traj_idx, start = self.slices[idx]
        # 실제 종료 프레임 계산 (frameskip 고려)
        end = start + self.num_frames * self.frameskip
        # 수정된 get_trajectory 메소드 호출
        obs, act, state, meta = self.dataset.get_trajectory(traj_idx, start, end, self.frameskip)

        # 액션 처리 (concat)
        if hasattr(self, 'action_dim') and self.action_dim == self.dataset.action_dim * self.frameskip:
             # 프레임 스킵이 적용된 길이에 맞춰 reshape
             act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames, f=self.frameskip)

        return tuple([obs, act, state]) # meta는 필요시 반환


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
        load_meta_only: bool = True # 메타데이터만 로드할지 여부 결정 플래그
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.action_scale = action_scale

        print(f"Initializing WallDataset from {self.data_path}")

        # 파일 경로 저장
        self.state_path = self.data_path / "states.pth"
        self.action_path = self.data_path / "actions.pth"
        self.proprio_path = self.data_path / "states.pth" # proprio는 state와 같음
        self.door_loc_path = self.data_path / "door_locations.pth"
        self.wall_loc_path = self.data_path / "wall_locations.pth"
        self.obs_dir = self.data_path / "obses"
        self.episode_paths = sorted([p for p in self.obs_dir.glob("episode_*.pth")])

        # 메타데이터 로드 (데이터 차원, 시퀀스 길이 등)
        if os.path.exists(self.state_path):
            # 임시로 첫 번째 데이터 로드하여 shape 확인 (또는 메타데이터 파일 사용)
            _temp_state = torch.load(self.state_path, map_location='cpu') # CPU에 로드하여 GPU 메모리 절약
            self.state_dim = _temp_state.shape[-1]
            self.proprio_dim = self.state_dim # proprio는 state와 같음
            self.traj_len = _temp_state.shape[1]
            self.num_total_rollouts = _temp_state.shape[0]
            del _temp_state # 메모리 해제
        else:
            raise FileNotFoundError(f"State file not found: {self.state_path}")

        if os.path.exists(self.action_path):
            _temp_action = torch.load(self.action_path, map_location='cpu')
            self.action_dim = _temp_action.shape[-1]
            # traj_len은 state에서 이미 얻었으므로 확인만 (선택적)
            assert self.traj_len == _temp_action.shape[1], "Mismatch in trajectory length between states and actions"
            del _temp_action
        else:
            raise FileNotFoundError(f"Action file not found: {self.action_path}")

        self.n_rollout = n_rollout if n_rollout is not None else self.num_total_rollouts
        if self.n_rollout > self.num_total_rollouts:
             print(f"Warning: n_rollout ({self.n_rollout}) > total available rollouts ({self.num_total_rollouts}). Using {self.num_total_rollouts}.")
             self.n_rollout = self.num_total_rollouts
        self.episode_paths = self.episode_paths[:self.n_rollout]


        # 정규화 파라미터 계산 또는 로드 (필요시)
        # 전체 데이터를 로드하지 않고 평균/표준편차를 계산하려면 스트리밍 방식 필요
        # 여기서는 미리 계산된 값 사용하거나, 계산 과정 생략 (예시)
        if normalize_action:
             # 전체 데이터를 메모리에 로드하지 않고 평균/표준편차 계산하기 어려움
             # 미리 계산된 값 사용 (예: ACTION_MEAN, ACTION_STD) 또는 구현 필요
             print("Warning: Calculating mean/std without loading full data requires implementation. Using predefined or zero/one values.")
             # 예시: 미리 정의된 값 사용
             self.action_mean = ACTION_MEAN
             self.action_std = ACTION_STD
             self.proprio_mean = STATE_MEAN # proprio는 state와 같음
             self.proprio_std = STATE_STD
             # self.state_mean = STATE_MEAN # state는 정규화 안 함 (원본 반환)
             # self.state_std = STATE_STD
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
            # self.state_mean = torch.zeros(self.state_dim)
            # self.state_std = torch.ones(self.state_dim)

        # 실제 데이터 로딩은 __getitem__에서 수행

    def get_seq_length(self, idx):
        # 모든 trajectory 길이가 동일하다고 가정
        return self.traj_len

    def get_trajectory(self, idx, start, end, frameskip):
        """ 특정 trajectory의 지정된 범위만 로드 """
        if idx >= self.n_rollout:
             raise IndexError(f"Index {idx} out of range for {self.n_rollout} rollouts.")

        # 필요한 프레임 인덱스 계산
        frame_indices = list(range(start, end, frameskip))
        if not frame_indices: # 프레임 스킵으로 인해 빈 리스트가 될 경우 방지
             return None, None, None, None

        # 행동(Action) 로드 및 슬라이싱
        # 전체 action 파일을 로드하지 않고 필요한 부분만 읽기 (예시: torch.load 사용)
        # 주의: torch.load는 전체 파일을 읽으므로 매우 큰 파일에는 비효율적일 수 있음
        # -> 대안: NumPy memmap 또는 HDF5 사용 고려
        full_actions = torch.load(self.action_path, map_location='cpu')
        act = full_actions[idx, start:end] # frameskip 적용 전의 모든 액션 로드
        del full_actions # 메모리 해제
        act = act / self.action_scale
        if self.normalize_action:
             act = (act - self.action_mean) / self.action_std

        # 상태(State) 및 Proprio 로드 및 슬라이싱 (정규화는 proprio에만 적용)
        full_states = torch.load(self.state_path, map_location='cpu')
        state = full_states[idx, frame_indices] # frameskip 적용된 인덱스로 로드
        proprio = full_states[idx, frame_indices].clone() # 복사본 생성
        del full_states
        proprio = (proprio - self.proprio_mean) / self.proprio_std

        # 이미지(Observation) 로드 및 슬라이싱
        image_path = self.episode_paths[idx]
        full_images = torch.load(image_path, map_location='cpu') # CPU 로드 권장
        image = full_images[frame_indices]
        del full_images
        image = image / 255.0 # 정규화 (0~1 범위로)
        if self.transform:
             # 개별 이미지에 transform 적용 위해 반복문 사용 또는 vmap 고려
             image = torch.stack([self.transform(img) for img in image])

        obs = {"visual": image,"proprio": proprio}

        # 메타데이터 로드 (예: door/wall location) - 필요한 경우
        full_door_loc = torch.load(self.door_loc_path, map_location='cpu')
        door_location = full_door_loc[idx, frame_indices]
        del full_door_loc
        full_wall_loc = torch.load(self.wall_loc_path, map_location='cpu')
        wall_location = full_wall_loc[idx, frame_indices]
        del full_wall_loc
        # meta 정보는 첫 번째 프레임만 사용한다고 가정
        meta = {'fix_door_location': door_location[0], 'fix_wall_location': wall_location[0]}

        return obs, act, state, meta

    def __len__(self):
        return self.n_rollout

def load_wall_slice_train_val(
    transform,
    n_rollout=None, # None으로 변경하여 모든 롤아웃 사용 가능하도록 함
    data_path='data/wall_single',
    normalize_action=False,
    split_ratio=0.8,
    split_mode="random",
    num_hist=0,
    num_pred=0,
    frameskip=1, # frameskip 기본값 1로 설정
    # object, encoder, num_clusters, num_features, use_coord, use_patch_info, folder_name 등 불필요 인자 제거 가능
):
    num_frames = num_hist + num_pred
    if num_frames <= 0:
         raise ValueError("num_hist + num_pred must be positive")

    if split_mode == "random":
        dset = WallDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path,
            normalize_action=normalize_action,
        )
        # random_split_traj 사용
        train_set, val_set = split_traj_datasets(
            dset,
            train_fraction=split_ratio,
            # random_seed=42 # 필요시 random seed 설정
        )
        train_slices = TrajSlicerDataset(train_set, num_frames, frameskip)
        val_slices = TrajSlicerDataset(val_set, num_frames, frameskip)

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
        train_slices = TrajSlicerDataset(dset_train, num_frames, frameskip)
        val_slices = TrajSlicerDataset(dset_val, num_frames, frameskip)
        # 폴더 분할 시 train_set, val_set 정의 (필요시)
        train_set = dset_train
        val_set = dset_val

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {} # 원본 Trajectory 데이터셋 (필요시 사용)
    traj_dset['train'] = train_set
    traj_dset['valid'] = val_set
    return datasets, traj_dset
