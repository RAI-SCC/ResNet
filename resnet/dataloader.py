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
def create_subset(valid_dataset=None,
                  train_dataset=None,
                  subset_size: int = None,
                  subset_factor: int = 0,
                  seed_training: bool = False,
                  seed: int = None):
    """
    Create a subset of the data.

    Parameters
    __________
    valid_dataset :
        Validation set.
    train_dataset :
        Training set.
    subset_size : int
        Number of samples used for train.
    subset_factor : int
        factor for train and validation subsets.
    seed_training : bool
        Use deterministic training if True.
    seed : int
        Seed for deterministic training.
    """

    if seed_training is True:
        np.random.seed(seed)

    train_targets = np.array(train_dataset.targets)
    valid_targets = np.array(train_dataset.targets)
    train_class_indices = defaultdict(list)
    valid_class_indices = defaultdict(list)
    total_size_train = len(train_targets)  # Total number of sampler in train
    total_size_valid = len(valid_targets)  # Total nNumber of sampler in valid

    if subset_factor != 0:
        subset_size_train = total_size_train / subset_factor
        subset_size_valid = total_size_valid / subset_factor
    else:
        subset_size_train = subset_size
        subset_size_valid = total_size_valid

    # Get indices for each train label
    for idx, label in enumerate(train_targets):
        train_class_indices[label].append(idx)
    # Get indices for each valid label
    for idx, label in enumerate(valid_targets):
        valid_class_indices[label].append(idx)

    # Get train fraction for correct distribution
    class_share_ints_train = {}  # fractions (number of samples) for each label rounded off to int
    class_share_diff_train = {}  # difference lost by rounding
    intermediate_size_train = 0
    for label in train_class_indices:
        label_share = len(train_class_indices[label]) / total_size_train
        fraction = label_share * subset_size_train
        class_share_ints_train[label] = int(fraction)
        class_share_diff_train[label] = fraction - int(fraction)
        intermediate_size_train += int(fraction)

    # Get valid fraction for correct distribution
    class_share_ints_valid = {}  # fractions (number of samples) for each label rounded off to int
    class_share_diff_valid = {}  # difference lost by rounding
    intermediate_size_valid = 0
    for label in valid_class_indices:
        label_share = len(valid_class_indices[label]) / total_size_valid
        fraction = label_share * subset_size_valid
        class_share_ints_valid[label] = int(fraction)
        class_share_diff_valid[label] = fraction - int(fraction)
        intermediate_size_valid += int(fraction)

    # Get indices for train subset
    train_indices = []
    diff = subset_size_train - intermediate_size_train
    top_n = sorted(class_share_diff_train.items(), key=lambda x: x[1], reverse=True)[:diff]
    for label in class_share_ints_train:
        fraction = class_share_ints_train[label]
        if label in [lbl for lbl, _ in top_n]:
            fraction += 1
        train_indices.append(np.random.choice(train_class_indices[label], fraction, replace=False))
    train_indices = np.concatenate(train_indices).tolist()
    train_dataset_sub = torch.utils.data.Subset(train_dataset, train_indices)

    # Get indices for valid subset
    valid_indices = []
    diff = subset_size_train - intermediate_size_valid
    top_n = sorted(class_share_diff_valid.items(), key=lambda x: x[1], reverse=True)[:diff]
    for label in class_share_ints_valid:
        fraction = class_share_ints_valid[label]
        if label in [lbl for lbl, _ in top_n]:
            fraction += 1
        valid_indices.append(np.random.choice(valid_class_indices[label], fraction, replace=False))
    valid_indices = np.concatenate(valid_indices).tolist()
    valid_dataset_sub = torch.utils.data.Subset(valid_dataset, valid_indices)

    return train_dataset_sub, valid_dataset_sub


@monitor()
def dataloader(batch_size: int = 32,
               num_workers: int = 8,
               subset_size: int = None,
               subset_factor: int = 0,
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
        Number of samples used for train.
    subset_size : int
        Fraction for train and validation sets.
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
    if subset_size is not None or subset_factor != 0:
        train_dataset, valid_dataset = create_subset(valid_dataset=valid_dataset,
                                                     train_dataset=train_dataset,
                                                     subset_size=subset_size,
                                                     subset_factor=subset_factor,
                                                     seed_training=seed_training,
                                                     seed=seed)

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
