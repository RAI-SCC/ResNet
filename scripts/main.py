import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from perun import monitor

import argparse

from resnet.model import ResNet
from resnet.train import train_model
from resnet.dataloader import dataloader


@monitor()
def main():
    #  Adjust hyperparameters:
    #  num_worker, batch size, epochs, ResNet size
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_subset",action="store_true") # a tag to use data subset for debugging
    parser.add_argument("--data_path",default="./",type=str)
    parser.add_argument("--batchsize",default=1,type=int)
    parser.add_argument("--num_epochs",default=2,type=int)
    args = parser.parse_args()
    
    start_time = time.perf_counter()

    # Distributed set up.
    world_size = int(os.getenv("SLURM_NPROCS"))  # Get overall number of processes.
    rank = int(os.getenv("SLURM_PROCID"))  # Get individual process ID.
    slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")  # Get GPU IDs.
    slurm_localid = int(os.getenv("SLURM_LOCALID"))  # Get local process ID.
    gpus_per_node = torch.cuda.device_count()
    gpu = rank % gpus_per_node
    assert gpu == slurm_localid
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(device)

    # Initialize DDP.
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method="env://"
    )

    if rank == 0:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {world_size}")
        print(f"Global Batch Size: {args.batchsize}")
        print(f"Local Batch Size: {int(args.batchsize / world_size)}")
        print(f"Max Epoch: {args.num_epochs}")
        print(f"Use Data Subset: {args.use_subset}")
        print(40*"-")
    if dist.is_initialized():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Slurm rank / world size: {rank} / {world_size}")
        print(f"Torch rank / world size: {torch.distributed.get_rank()} / {torch.distributed.get_world_size()}")
        print(40*"-")
    else: 
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Batch Size: {args.batchsize}")
        print(f"Max Epoch: {args.num_epochs}")
        print(f"Use Data Subset: {args.use_subset}")
        print(40*"-")

    # Get distributed dataloaders on all ranks.
    train_loader, valid_loader = dataloader(batch_size=args.batchsize, num_workers=2, use_subset=args.use_subset, path_to_data=args.data_path)

    model = ResNet().to(device)  # Create model and move it to GPU with id rank.
    model = DDP(model, device_ids=[slurm_localid], output_device=slurm_localid)  # Wrap model with DDP.
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.1, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Train model.
    valid_loss_history, train_acc_history, valid_acc_history, time_history = train_model(
        model=model,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        start_time=start_time,
        lr_scheduler=lr_scheduler
    )

    dist.destroy_process_group()


# MAIN STARTS HERE.
if __name__ == "__main__":
    main()
