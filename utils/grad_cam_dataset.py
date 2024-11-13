import torch
from torchvision import datasets, transforms

class grad_cam_dataset(datasets.ImageFolder):
    def __init__(self, root, transform, mask_root):
        super(grad_cam_dataset, self).__init__(root, transform=transform)
        self.mask_root = mask_root

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        #cls_name = path.split('train/')[-1] if self.is_train else path.split('val/')[-1]
        cls = path.split('/')[-2]
        name = path.split('/')[-1]
        mask_idx = torch.load(self.mask_root + cls + '_' + name + '.pt', map_location='cpu')
        return sample, target, mask_idx