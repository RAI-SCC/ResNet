import time
import pickle
import itertools

import torch
import h5py


def train_model(
    model,
    num_epochs,
    train_loader,
    valid_loader,
    optimizer,
    start_time,
    warmup_scheduler,
    lr_scheduler,
    warmup_epochs=0
):
    """
    Train model in DDP fashion.

    Parameters
    __________
    model : torch.nn.Module
        model to train
    num_epochs : int
        number of epochs to train
    train_loader : torch.utils.data.Dataloader
        training dataloader
    valid_loader : torch.utils.data.Dataloader
        validation dataloader
    optimizer : torch.optim.Optimizer
        optimizer to use
    start_time : float
        Start time of main
    warmup_scheduler :
        For gradually increasing the lr
    lr_scheduler :
        LR scheduler
    warmup_epochs : int
        num of epochs for lr get reach target value

    Returns
    _______
    loss_history : list
        History of loss.
    train_acc_history : list
        History of training accuracy.
    valid_acc_history : list
        History of validation accuracy.
    lr : list
        History of learning rate
    time_history : list
        History of elapsed time corresponding to lists above.
    """

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    (valid_loss_history, train_loss_history, time_history, lr_history) = [], [], [], []
    (top1_acc_train_history, top5_acc_train_history) = [], []
    (top1_acc_valid_history, top5_acc_valid_history) = [], []

    if rank == 0:
        print("Start Training")
        print(40*"-")

    epoch_times = {}

    for epoch in range(num_epochs):  # Loop over epochs.
        train_loader.sampler.set_epoch(epoch)
        model.train()  # Set model to training mode.

        batch_times = {"batch_time_dataloading": [], "batch_time_data_to_device": [], "batch_time_forward": [],
                       "batch_time_backward": [], "batch_time_total": []}

        n_train_batches = 75
        btimer_0 = time.perf_counter()
        for batch_idx, (features, targets) in enumerate(itertools.islice(train_loader, n_train_batches)):  # Loop over mini batches.
            # Data to GPUs
            btimer_1 = time.perf_counter()
            features = features.cuda()
            targets = targets.cuda()
            btimer_2 = time.perf_counter()
            # Forward and backward pass.
            output = model(features)
            btimer_3 = time.perf_counter()
            loss = torch.nn.functional.cross_entropy(output, targets)
            optimizer.zero_grad()
            btimer_4 = time.perf_counter()
            loss.backward()
            btimer_5 = time.perf_counter()
            optimizer.step()
            btimer_6 = time.perf_counter()
            batch_times["batch_time_dataloading"].append(btimer_1 - btimer_0)
            batch_times["batch_time_data_to_device"].append(btimer_2 - btimer_1)
            batch_times["batch_time_forward"].append(btimer_3 - btimer_2)
            batch_times["batch_time_backward"].append(btimer_5 - btimer_4)
            batch_times["batch_time_total"].append(btimer_6 - btimer_0)
            btimer_0 = time.perf_counter()

        epoch_times[f"batch_times_e{epoch + 1}"] = batch_times


    local_dict = epoch_times
    gathered_times = {}
    torch.distributed.all_gather_object(gathered_times, local_dict)

    if rank == 0:
        with h5py.File('times.h5', 'w') as h5f:
            for key1, val1 in gathered_times.items():
                # Create node groups
                h5f.create_group(f"{key1}")
                for key2, val2 in gathered_times[key1].items():
                    # Create batch groups
                    if isinstance(val2, dict):
                        h5f.create_group(f"{key1}/{key2}")
                        for key3, val3 in gathered_times[key1][key2].items():
                            h5f[f"{key1}/{key2}/{key3}"] = val3
                    # write epoch items
                    else:
                        h5f[f"{key1}/{key2}"] = val2

    return valid_loss_history, top1_acc_train_history, top1_acc_valid_history, lr_history, time_history
