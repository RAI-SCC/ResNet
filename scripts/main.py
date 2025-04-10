import os
import time
import random
import argparse

from perun import monitor
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR

from resnet.model import ResNet
from resnet.train import train_model
from resnet.dataloader import dataloader


@monitor()
def main():
    #  Adjust hyperparameters:
    #  num_worker, batch size, epochs, ResNet size
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_subset", action="store_true")  # a tag to use data subset for debugging
    parser.add_argument("--data_path", default="./", type=str)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--lr_scheduler", default="multistep", type=str, choices=["cosine", "plateau", "multistep"], help="Choose learning rate scheduler (cosine, plateau, multistep)")
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)  
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
        torch.backends.cudnn.benchmark = False 
    
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
        if args.seed is not None:
            print(f"Deterministic training is enabled")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {world_size}")
        print(f"Global Batch Size: {args.batchsize}")
        print(f"Local Batch Size: {int(args.batchsize / world_size)}")
        print(f"Max Epoch: {args.num_epochs}")
        print(f"Use Data Subset: {args.use_subset}")
        print(f"Number of Workers: {args.num_workers}")  
        print(f"LR Scheduler: {args.lr_scheduler}")
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
    if args.seed is not None:
        train_loader, valid_loader = dataloader(
            batch_size=args.batchsize, 
            num_workers=args.num_workers, 
            use_subset=args.use_subset, 
            path_to_data=args.data_path,
            seed_training=True
        )
    else: 
        train_loader, valid_loader = dataloader(
            batch_size=args.batchsize, 
            num_workers=args.num_workers, 
            use_subset=args.use_subset, 
            path_to_data=args.data_path
        )

    model = ResNet().to(device)  # Create model and move it to GPU with id rank.
    model = DDP(model, device_ids=[slurm_localid], output_device=slurm_localid)  # Wrap model with DDP.
    reference_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=reference_lr, weight_decay=0.0001)

    # Define schedulers
    warmup_epochs = 5
    if args.batchsize > 256:
        linear_scaling_factor = args.batchsize / 256  # lr factor to resolve large batch effect with batch size 256 as baseline: https://arxiv.org/pdf/1706.02677
    else: 
        linear_scaling_factor = 1
    max_lr = reference_lr * linear_scaling_factor  # Final learning rate after warmup

    def warmup_fn(epoch):
        if epoch == 0:
            return 1
        else:
            return (epoch+1) / warmup_epochs * max_lr * 1 / reference_lr
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
    if args.lr_scheduler == "plateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif args.lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.lr_scheduler == "multistep":
        milestones = [int(.3 * args.num_epochs), int(.6 * args.num_epochs), int(.8 * args.num_epochs)]
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        raise ValueError(f"Unknown lr scheduler: {args.lr_scheduler}")

    # Train model.
    valid_loss_history, train_acc_history, valid_acc_history, lr_history, time_history = train_model(
        model=model,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        start_time=start_time,
        warmup_scheduler=warmup_scheduler,
        lr_scheduler=lr_scheduler,
        warmup_epochs=warmup_epochs
    )

    dist.destroy_process_group()


# MAIN STARTS HERE.
if __name__ == "__main__":
    main()
