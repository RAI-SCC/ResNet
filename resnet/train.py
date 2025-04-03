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
    correct_pred : int
        The number of correctly predicted samples.
    num_examples : int
        The overall number of samples in the dataset.
    loss : float
        Loss
    """
    with torch.no_grad():
        top1_pred, top5_pred, total_num_examples, loss = 0, 0, 0
        for i, (features, targets) in enumerate(data_loader):
            features = features.cuda()
            targets = targets.float().cuda()
            output = model(features)

            num_examples = targets.size(0)
            total_num_examples += num_examples

            loss += torch.nn.functional.cross_entropy(output, targets.long())
            top1_labels = torch.topk(output, 1, dim=1).indices  # Top-1 prediction
            top1_labels = top1_labels.reshape(top1_labels.shape[0])
            top5_labels = torch.topk(output, 5, dim=1).indices  # Top-5 predictions
            top1_correct = (top1_labels == targets).sum()
            top5_correct = sum([targets[j] in top5_labels[j] for j in range(num_examples)])
            top1_pred += top1_correct
            top5_pred += top5_correct
        total_num_examples = torch.Tensor([total_num_examples]).cuda()
        top1_pred = torch.Tensor([top1_pred]).cuda()
        top5_pred = torch.Tensor([top5_pred]).cuda()
        loss = torch.Tensor([loss]).cuda()
    return total_num_examples, loss, top1_pred, top5_pred


@monitor()
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
            loss.backward()
            optimizer.step()

        model.eval()  # Set model to evaluation mode.

        with torch.no_grad():  # Disable gradient calculation.
            # Get rank-local numbers of correctly classified and overall samples in training and validation set.
            num_train, train_loss, top1_pred_train, top5_pred_train = get_right(model, train_loader)
            num_valid, valid_loss, top1_pred_valid, top5_pred_valid = get_right(model, valid_loader)
            # Allreduce rank-local numbers of correctly classified and overall training and validation samples.
            torch.distributed.all_reduce(top1_pred_train)
            torch.distributed.all_reduce(top5_pred_train)
            torch.distributed.all_reduce(top1_pred_valid)
            torch.distributed.all_reduce(top5_pred_valid)
            torch.distributed.all_reduce(num_train)
            torch.distributed.all_reduce(num_valid)
            torch.distributed.all_reduce(valid_loss)
            torch.distributed.all_reduce(train_loss)
            # Calculate correct values
            time_elapsed = (time.perf_counter() - start_time) / 60
            top1_acc_train = top1_pred_train.item() / num_train.item() * 100
            top5_acc_train = top5_pred_train.item() / num_train.item() * 100
            top1_acc_valid = top1_pred_valid.item() / num_valid.item() * 100
            top5_acc_valid = top5_pred_valid.item() / num_valid.item() * 100
            valid_loss /= num_valid.item()
            train_loss /= num_train.item()
            # append to history
            valid_loss_history.append(valid_loss.item())
            train_loss_history.append(train_loss.item())
            top1_acc_train_history.append(top1_acc_train)
            top5_acc_train_history.append(top5_acc_train)
            top1_acc_valid_history.append(top1_acc_valid)
            top5_acc_valid_history.append(top5_acc_valid)
            time_history.append(time_elapsed)
            lr_history.append(optimizer.state_dict()['param_groups'][0]['lr'])
            # Scheduler Step
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                lr_scheduler.step(valid_loss)
            
            if rank == 0:
                print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                      f'| Validation Loss: {valid_loss:.4f} '
                      f'| Training Loss: {train_loss:.4f} '
                      f'| Time: {time_elapsed :.2f} min '
                      f'| Top1-Train: {top1_acc_train :.2f}% '
                      f'| Top1-Validation: {top1_acc_valid :.2f}% '
                      f'| Top5-Train: {top5_acc_train :.2f}% '
                      f'| Top5-Validation: {top5_acc_valid :.2f}% '
                      f'| LR: {optimizer.state_dict()["param_groups"][0]["lr"] :.6f} '
                      f'| Time: {time_elapsed :.2f} min')

                torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, "ckpt.tar")

    if rank == 0:
        torch.save(train_loss_history, f'train_loss.pt')
        torch.save(valid_loss_history, f'valid_loss.pt')
        torch.save(top1_acc_train_history, f'train_top1.pt')
        torch.save(top1_acc_valid_history, f'valid_top1.pt')
        torch.save(top5_acc_train_history, f'train_top5.pt')
        torch.save(top5_acc_valid_history, f'valid_top5.pt')
        torch.save(time_history, f'time.pt')
        torch.save(lr_history, f'lr.pt')

    return valid_loss_history, top1_acc_train_history, top1_acc_valid_history, lr_history, time_history
