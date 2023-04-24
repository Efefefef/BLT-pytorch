import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import h5py


def get_dataloaders(dataset, train_batch_size, val_test_batch_size, transforms_list, num_mean_channels, 
                    num_std_channels, train_num_workers, val_test_num_workers, prefetch_factor, device):

    train_transform = compose_transforms(transforms_list, num_mean_channels, num_std_channels)
    val_test_transform = compose_transforms(['normalize'], num_mean_channels, num_std_channels)

    train_dataloader = DataLoader(
        MiniEcoset(os.path.join(dataset, 'train'), transform=train_transform, device=device),
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        prefetch_factor=prefetch_factor,
    )
    val_dataloader = DataLoader(
        MiniEcoset(os.path.join(dataset, 'val'), transform=val_test_transform, device=device),
        batch_size=val_test_batch_size,
        num_workers=val_test_num_workers,
        prefetch_factor=prefetch_factor,
    )
    test_dataloader = DataLoader(
        MiniEcoset(os.path.join(dataset, 'test'), transform=val_test_transform, device=device),
        batch_size=val_test_batch_size,
        num_workers=val_test_num_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_dataloader, val_dataloader, test_dataloader

def compose_transforms(transforms_list, num_mean_channels, num_std_channels):
    transform_list = []

    if 'trivialaug' in transforms_list:
        transform_list.append(transforms.TrivialAugmentWide())
        
    if 'randaug' in transforms_list:
        transform_list.append(transforms.RandAugment())

    transform_list.append(transforms.ConvertImageDtype(torch.float))

    if 'normalize' in transforms_list:
        mean = num_mean_channels / 255.
        std = num_std_channels / 255.
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)

class MiniEcoset(Dataset):
    def __init__(self, split_path, transform, device):
        self.transform = transform
        self.device = device

        with h5py.File(self.split_path, "r") as f:
            self.images = torch.from_numpy(f[split_path]['data'][()]).permute(0, 3, 1, 2)  # to match the CHW expectation of pytorch
            self.labels = torch.from_numpy(f[split_path]['labels'][()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            print('IS THIS EVER HAPPENING?')
            idx = idx.tolist()

        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image).to(self.device)
        return image, label
