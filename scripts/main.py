import os
import time
import random
import argparse
import socket

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR

from resnet.model import ResNet
from resnet.train_with_monitoring import train_model, warmup_goyal_fn
from resnet.dataloader import dataloader


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", default=None, type=int,
                        help='Size of Subset, i.e. number of Samples. If None, the full dataset is used')
    parser.add_argument("--subset_factor", default=0, type=int,
                        help='Factor (devisor) of Subset. If 0, the full dataset is used')
    parser.add_argument("--data_path", default="./", type=str,
                        help='Path to data.')
    parser.add_argument("--batchsize", default=1, type=int,
                        help='Global batch size.')
    parser.add_argument("--num_epochs", default=2, type=int,
                        help='Number of epochs to be trained.')
    parser.add_argument("--num_workers", default=2, type=int,
                        help='Number of workers used in dataloader.')
    parser.add_argument("--lr_scheduler", default="plateau", type=str, choices=["cosine", "plateau", "multistep"],
                        help="Choose learning rate scheduler (cosine, plateau, multistep).")
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training')
    parser.add_argument('--batch_iter', default=0, type=int,
                        help='Maximum number of batch iterations per epoch. If 0, this value is ignored.')
    args = parser.parse_args()

    seed_training = False
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_training = True

    # Distributed set up
    world_size = int(os.getenv("SLURM_NPROCS"))  
    rank = int(os.getenv("SLURM_PROCID")) 
    hostname = socket.gethostname()
    slurm_localid = int(os.getenv("SLURM_LOCALID")) 
    gpus_per_node = torch.cuda.device_count()
    gpu = rank % gpus_per_node
    assert gpu == slurm_localid
    device = f"cuda:{slurm_localid}"
    torch.cuda.set_device(device)
    device_id = torch.device(f"cuda:{slurm_localid}")
    gpu_id = torch.cuda.current_device()

    # Initialize DDP
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device_id, init_method="env://"
    )

    start_time = time.perf_counter()

    if rank == 0:
        print(f"{30 * '-'} \n")
        if args.seed is not None:
            print(f"TRAINING INFO || Deterministic training is enabled")
        if args.subset_size is not None:
            print(f"TRAINING INFO || A data subset of {args.subset_size} samples in train is used")
        if args.subset_factor != 0:
            print(f"TRAINING INFO || A data subset with a fraction of 1/{args.subset_factor} in train and validation is used")
        if args.batch_iter != 0:
            print(f"TRAINING INFO || A maximum number of {args.batch_iter} batch iterations applied")
        print(f"TRAINING INFO || CUDA Available: {torch.cuda.is_available()} \n"
              f"TRAINING INFO || Number of GPUs: {world_size} \n"
              f"TRAINING INFO || Global Batch Size: {args.batchsize} \n"
              f"TRAINING INFO || Local Batch Size: {int(args.batchsize / world_size)} \n"
              f"TRAINING INFO || Max Epoch: {args.num_epochs} \n"
              f"TRAINING INFO || Number of Workers: {args.num_workers} \n"
              f"TRAINING INFO || LR Scheduler: {args.lr_scheduler} \n"
              f"{30 * '-'}")
    torch.cuda.synchronize()
    dist.barrier()
    if dist.is_initialized():
        print(f"| Hostname: {hostname} "
              f"| GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())} "
              f"| Device: {gpu_id} "
              f"| Slurm rank / world size: {rank} / {world_size} |")
    else:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Batch Size: {args.batchsize}")
        print(f"Max Epoch: {args.num_epochs}")

    # Get distributed dataloaders on all ranks
    train_loader, valid_loader = dataloader(
        batch_size=args.batchsize,
        num_workers=args.num_workers,
        subset_size=args.subset_size,
        subset_factor=args.subset_factor,
        path_to_data=args.data_path,
        seed_training=seed_training,
        seed=args.seed
    )

    model = ResNet().to(device) 
    model = DDP(model, device_ids=[slurm_localid], output_device=slurm_localid)
    reference_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=reference_lr, weight_decay=0.0001)

    # Define warmup https://arxiv.org/pdf/1706.02677
    warmup_epochs = 5
    warmup_scheduler = LambdaLR(optimizer,
                                lr_lambda=lambda epoch: warmup_goyal_fn(epoch,
                                                                        batchsize=args.batchsize,
                                                                        warmup_epochs=warmup_epochs,
                                                                        reference_lr=reference_lr
                                                                        )
                                )

    # Define schedulers
    if args.lr_scheduler == "plateau":
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif args.lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.lr_scheduler == "multistep":
        milestones = [int(.3 * args.num_epochs), int(.6 * args.num_epochs), int(.8 * args.num_epochs)]
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        raise ValueError(f"Unknown lr scheduler: {args.lr_scheduler}")

    # Train model
    valid_loss_history, train_acc_history, valid_acc_history, lr_history, time_history = train_model(
        model=model,
        num_epochs=args.num_epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        start_time=start_time,
        warmup_scheduler=warmup_scheduler,
        lr_scheduler=lr_scheduler,
        warmup_epochs=warmup_epochs,
        batch_iter=args.batch_iter
    )

    dist.destroy_process_group()


# Main starts here
if __name__ == "__main__":
    main()
