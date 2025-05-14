import torch
import torchvision as tv
from perun import monitor
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
import random

class DummyImageNetDataset(Dataset):
    """
    A dummy dataset that generates random image-like tensors and labels
    to mimic the output of ImageFolder for debugging/testing.
    """
    def __init__(self, num_samples=100000, image_size=224, num_classes=1000):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        # The shape matches the expected output after ImageFolder + transforms
        # (Channels, Height, Width) and dtype float32 after ToTensor and Normalize
        self.image_shape = (3, image_size, image_size)

        print(f"Initialized DummyImageNetDataset with {num_samples} samples.")
        print(f"  Image shape: {self.image_shape}")
        print(f"  Number of classes: {self.num_classes}")


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random image tensor
        # Use dtype float32 as transforms like ToTensor() and Normalize() would produce
        dummy_image = torch.randn(self.image_shape, dtype=torch.float32)

        # Generate a random class label
        # Use dtype long as is standard for classification targets in PyTorch
        dummy_label = torch.tensor(random.randint(0, self.num_classes - 1), dtype=torch.long)

        return dummy_image, dummy_label

@monitor()
def dataloader(batch_size: int = 32, num_workers: int = 8, use_subset: bool = False, path_to_data: str = "./", seed_training: bool = False):
    """
    Get distributed ImageNet dataloaders for training and validation in a DDP setting.

    Parameters
    __________
    batch_size : int
        Batch size.
    num_workers : int
        How many workers to use for data loading.
    """

    print("Using dummy dataloaders.")

    # --- Dummy Dataset Creation ---
    image_size = 224
    num_classes = 1000 # Standard ImageNet number of classes

    # Define sizes for full and subset dummy datasets
    full_dataset_size = 100000 # Arbitrary large number
    subset_dataset_size = full_dataset_size * 0.01 # Arbitrary smaller number

    if use_subset:
        train_dataset_size = subset_dataset_size
        valid_dataset_size = subset_dataset_size
    else:
        train_dataset_size = full_dataset_size
        valid_dataset_size = full_dataset_size

    # Create dummy datasets
    train_dataset = DummyImageNetDataset(num_samples=train_dataset_size, image_size=image_size, num_classes=num_classes)
    valid_dataset = DummyImageNetDataset(num_samples=valid_dataset_size, image_size=image_size, num_classes=num_classes)

    # --- Sampler and DataLoader Setup (Mimics Original) ---

    # Ensure distributed training is initialized for DistributedSampler
    if not torch.distributed.is_initialized():
         print("Warning: Distributed training not initialized. DistributedSampler may fail or behave unexpectedly.")
         # Optionally raise an error or handle non-DDP case if needed
         # raise RuntimeError("Distributed training must be initialized.")
         # For a simple dummy test without DDP, you might return standard DataLoaders without Samplers
         # but the original code structure relies on DDP being initialized.
         # Let's proceed assuming DDP is initialized for now.
         world_size = 1
         rank = 0
    else:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()


    # Define sampler that restricts data loading to a subset of the dataset
    # These use the __len__ method of the DummyDataset
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True, # Original code shuffled validation sampler too
        drop_last=True, # Original code dropped last for validation too
    )

    # Worker init function mimicking original
    generator = None
    worker_init_fn = None
    if seed_training is True:
        def worker_init_fn(worker_id):
            # This seeds numpy and random based on torch's initial seed + worker_id
            # torch.initial_seed() returns the seed torch is using for the current worker
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            # Note: Seeding torch within worker_init_fn for GPU operations is more complex
            # torch.cuda.manual_seed_all(worker_seed) might be needed for GPU tensors
            # but for simply generating random data on CPU before pin_memory, this is usually fine.

        # Setting the generator for the main process DataLoader is needed for
        # reproducible shuffling when using Samplers with shuffle=True
        # It also influences the seeds seen by worker_init_fn
        generator = torch.Generator()
        # Consider using a seed based on rank + a fixed seed for DDP reproducibility across runs
        base_seed = 42 # Or some other fixed seed
        generator.manual_seed(base_seed + rank)


    # Define dataloaders
    # The batch_size is divided by world_size as in the original DDP setup
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size // world_size, # Local batch size per process
        sampler=train_sampler, # Use the distributed sampler
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator, # Pass the generator for reproducible shuffling
        pin_memory=True # Typically True for GPU training
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size // world_size, # Local batch size per process
        sampler=valid_sampler, # Use the distributed sampler
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator, # Pass the same generator (or a derived one)
        pin_memory=True
    )

    return train_loader, valid_loader