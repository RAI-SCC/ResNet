import time

import torch
from perun import monitor


@monitor()
def compute_accuracy(model, data_loader):
    """
    Compute accuracy of model predictions on given labeled data.

    Parameters
    __________
    model : torch.nn.Module
        Model.
    data_loader : torch.utils.data.Dataloader
        Dataloader.
    device : torch.device
        device to use

    Returns
    _______
    float : The model's accuracy on the given dataset in percent.
    """
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.cuda()
            targets = targets.float().cuda()
            output = model(features)
            _, predicted_labels = torch.max(output, 1)  # Get class with highest score.
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def get_right(model, data_loader):
    """
    Compute the number of correctly predicted samples and the overall number of samples in a given dataset.

    Parameters
    __________
    model : torch.nn.Module
        Model.
    data_loader : torch.utils.data.Dataloader
        Dataloader.

    Returns
    _______
    int : correct_pred
        The number of correctly predicted samples.
    int : num_examples
        The overall number of samples in the dataset.
    """
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.cuda()
            targets = targets.float().cuda()
            output = model(features)
            _, predicted_labels = torch.max(output, 1)  # Get class with highest score.
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    num_examples = torch.Tensor([num_examples]).cuda()
    return correct_pred, num_examples


@monitor()
def train_model(
    model,
    num_epochs,
    train_loader,
    valid_loader,
    optimizer,
    start_time,
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

    Returns
    _______
    loss_history : list
        History of loss.
    train_acc_history : list
        History of training accuracy.
    valid_acc_history : list
        History of validation accuracy.
    time_history : list
        History of elapsed time corresponding to lists above.
    """

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    loss_history, train_acc_history, valid_acc_history, time_history = [], [], [], []  # Initialize history lists.

    for epoch in range(num_epochs):  # Loop over epochs.
        train_loader.sampler.set_epoch(epoch)
        model.train()  # Set model to training mode.

        for batch_idx, (features, targets) in enumerate(train_loader):  # Loop over mini batches.
            # Data to GPUs
            features = features.cuda()
            targets = targets.cuda()
            # Forward and backward pass.
            output = model(features)
            loss = torch.nn.functional.cross_entropy(output, targets)
            optimizer.zero_grad()
            # lr scheduler?
            loss.backward()
            optimizer.step()
            # Logging.
            torch.distributed.all_reduce(loss)
            loss /= world_size

        model.eval()  # Set model to evaluation mode.

        with torch.no_grad():  # Disable gradient calculation.
            # Get rank-local numbers of correctly classified and overall samples in training and validation set.
            right_train, num_train = get_right(model, train_loader)
            right_valid, num_valid = get_right(model, valid_loader)
            # Allreduce rank-local numbers of correctly classified and overall training and validation samples.
            torch.distributed.all_reduce(right_train)
            torch.distributed.all_reduce(right_valid)
            torch.distributed.all_reduce(num_train)
            torch.distributed.all_reduce(num_valid)
            torch.distributed.all_reduce(loss)
            loss /= world_size
            time_elapsed = (time.perf_counter() - start_time) / 60
            train_acc = right_train.item() / num_train.item() * 100
            valid_acc = right_valid.item() / num_valid.item() * 100
            loss_history.append(loss.item())
            train_acc_history.append(train_acc)
            valid_acc_history.append(valid_acc)
            time_history.append(time_elapsed)

            if rank == 0:
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Loss: {loss:.4f} '
                      f'| Train: {train_acc :.2f}% '
                      f'| Validation: {valid_acc :.2f}% '
                      f'| Time: {time_elapsed :.2f} min')

                torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, "ckpt.tar")

    if rank == 0:
        torch.save(loss_history, f'loss.pt')
        torch.save(train_acc_history, f'train_acc.pt')
        torch.save(valid_acc_history, f'valid_acc.pt')
        torch.save(time_history, f'time.pt')

    return loss_history, train_acc_history, valid_acc_history, time_history
