import torch
import torchvision as tv
from perun import monitor
import numpy as np
import random
from collections import defaultdict


def worker_init_seed_fn(worker_id):
    """
    Function to set random seed for workers.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@monitor()
def dataloader(batch_size: int = 32,
               num_workers: int = 8,
               subset_size: int = None,
               path_to_data: str = "./",
               seed_training: bool = False,
               seed: int = None):
    """
    Get distributed ImageNet dataloaders for training and validation in a DDP setting.

    Parameters
    __________
    batch_size : int
        Batch size.
    num_workers : int
        How many workers to use for data loading.
    subset_size : int
        Number of samples used.
    path_to_data : str
        Path to data.
    seed_training : bool
        Use deterministic training if True.
    seed : int
        Seed for deterministic training.
    """

    # Define Paths
    path_to_train = path_to_data + "/train"
    path_to_valid = path_to_data + "/val"

    image_size = 224  # Image size required for ResNet

    # Define transforms
    train_transform = tv.transforms.Compose(
        [
            tv.transforms.RandomResizedCrop(size=image_size, scale=(0.5, 1.0)),
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

    # Use a subset
    if subset_size is not None:
        if seed_training is True:
            np.random.seed(seed)
        targets = np.array(train_dataset.targets)
        class_indices = defaultdict(list)
        total_size = len(targets)
        # Get indices for each label
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
        # Get fraction for correct distribution
        class_share_ints = {}
        class_share_diff = {}
        intermediate_size = 0
        for label in class_indices:
            label_share = len(class_indices[label]) / total_size
            fraction = label_share * subset_size
            class_share_ints[label] = int(fraction)
            class_share_diff[label] = fraction - int(fraction)
            intermediate_size += int(fraction)
        # Get indices for subset
        train_indices = []
        diff = subset_size - intermediate_size
        top_n = sorted(class_share_diff.items(), key=lambda x: x[1], reverse=True)[:diff]
        for label in class_share_ints:
            fraction = class_share_ints[label]
            if label in [lbl for lbl, _ in top_n]:
                fraction += 1
            train_indices.append(np.random.choice(class_indices[label], fraction, replace=False))
        train_indices = np.concatenate(train_indices).tolist()
        # Create subset
        train_dataset_sub = torch.utils.data.Subset(train_dataset, train_indices)
        train_dataset = train_dataset_sub

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

    # Set random seeds
    generator = None
    worker_init_fn = None
    if seed_training is True:
        worker_init_fn = worker_init_seed_fn
        generator = torch.Generator()
        generator.manual_seed(0)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size // torch.distributed.get_world_size(),
        sampler=train_sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size // torch.distributed.get_world_size(),
        sampler=valid_sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator
    )
    return train_loader, valid_loader
