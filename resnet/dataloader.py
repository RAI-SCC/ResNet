import torch
import torchvision as tv
from perun import monitor
import numpy as np


@monitor()
def dataloader(batch_size: int = 32, num_workers: int = 8, use_subset: bool = False, path_to_data: str = "./"):
    """
    Get distributed ImageNet dataloaders for training and validation in a DDP setting.

    Parameters
    __________
    batch_size : int
        Batch size.
    num_workers : int
        How many workers to use for data loading.
    """

    # Define Paths
    path_to_train = path_to_data + "/train"
    path_to_valid = path_to_data + "/val"

    image_size = 224  # Image size required for ResNet

    # Define transforms
    train_transform = tv.transforms.Compose(
        [
            tv.transforms.Resize((image_size, image_size)),
            tv.transforms.RandomCrop(64),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Define datasets
    train_dataset = tv.datasets.ImageFolder(root=path_to_train, transform=train_transform)
    valid_dataset = tv.datasets.ImageFolder(root=path_to_valid, transform=train_transform)

    # subset for fast debugging - to be deleted
    if use_subset:
        subset_ratio = 0.01
        # Get subset indices
        train_size_sub = int(len(train_dataset) * subset_ratio)
        valid_size_sub = int(len(valid_dataset) * subset_ratio)
        train_indices = np.random.choice(len(train_dataset), train_size_sub, replace=False)
        valid_indices = np.random.choice(len(valid_dataset), valid_size_sub, replace=False)
        # Create subset datasets
        train_dataset_sub = torch.utils.data.Subset(train_dataset, train_indices)
        valid_dataset_sub = torch.utils.data.Subset(valid_dataset, valid_indices)
        train_dataset = train_dataset_sub
        valid_dataset = valid_dataset_sub

    # Define sampler that restricts data loading to a subset of the dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,
        drop_last=True,
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,
        drop_last=True,
    )

    # Define loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    return train_loader, valid_loader
