import time
import itertools

import torch
import torch.distributed as dist
from perun import monitor


def warmup_goyal_fn(epoch, batchsize, warmup_epochs, reference_lr):
    """
    Function for warm-up as defined in https://arxiv.org/pdf/1706.02677.

    Parameters
    ----------
    epoch : int
        Current epoch.
    batchsize : int
        Batchsize.
    warmup_epochs : int
        Number of epochs used for warm-up.
    reference_lr : float
        Initial reference learning rate.

    Returns
    -------
    lr : float
        Learning rate.
    """
    linear_scaling_factor = batchsize / 256
    max_lr = reference_lr * linear_scaling_factor
    diff_lr = max_lr - reference_lr
    if epoch == 0:
        lr = 1
    else:
        lr = (reference_lr + (epoch / warmup_epochs) * diff_lr) / reference_lr
    return lr


def get_right(model, data_loader):
    """
    Compute the number of correctly predicted samples and the overall number of samples in a given dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Model.
    data_loader : torch.utils.data.Dataloader
        Dataloader.

    Returns
    -------
    num_examples : int
        The overall number of samples in the dataset.
    loss : float
        Loss.
    top1_pred : fload
        Top1 error.
    top5_pred : float
        Top5 error.
    """
    with torch.no_grad():
        top1_pred, top5_pred, total_num_examples, loss = 0, 0, 0, 0
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
        loss /= (i + 1)
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
    warmup_epochs=0,
    batch_iter=0
):
    """
    Train model in DDP fashion.

    Parameters
    __________
    model : torch.nn.Module
        Model to be trained.
    num_epochs : int
        Number of epochs to be trained.
    train_loader : torch.utils.data.Dataloader
        Training data loader.
    valid_loader : torch.utils.data.Dataloader
        Validation data loader.
    optimizer : torch.optim.Optimizer
        Optimizer to used.
    start_time : float
        Start time in main.
    warmup_scheduler :
        Warm-up scheduler for learning rate.
    lr_scheduler :
        LR scheduler.
    warmup_epochs : int
        Number of epochs used for the warm-up scheduler.
    batch_iter: int
        Maximum number of batch iterations per epoch. If 0, no limit is applied.

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

    (valid_loss_history, train_loss_history, epoch_time_history, lr_history) = [], [], [], []
    (top1_acc_train_history, top5_acc_train_history) = [], []
    (top1_acc_valid_history, top5_acc_valid_history) = [], []

    if rank == 0:
        print("Start Training")
        print(40*"-")

    for epoch in range(num_epochs):  # Loop over epochs.
        train_loader.sampler.set_epoch(epoch)
        model.train()

        if batch_iter == 0:
            n_train_batches = len(train_loader)
        else:
            n_train_batches = batch_iter

        for batch_idx, (features, targets) in enumerate(itertools.islice(train_loader, n_train_batches)):  # Loop over mini batches.

            # Data to GPUs
            torch.cuda.synchronize()
            dist.barrier()
            features, targets = data_to_device(features, targets)
            # Forward
            torch.cuda.synchronize()
            dist.barrier()
            output = single_forward_step(features)
            # Loss
            torch.cuda.synchronize()
            dist.barrier()
            loss = single_loss_step(output, targets)
            optimizer.zero_grad()
            # Backward
            torch.cuda.synchronize()
            dist.barrier()
            single_backward_step(loss)
            # Weight step
            torch.cuda.synchronize()
            dist.barrier()
            single_update_step()

        # Evaluation
        if batch_iter == 0:
            model.eval()
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
                valid_loss = valid_loss.item() / world_size
                train_loss = train_loss.item() / world_size
                # append to history
                valid_loss_history.append(valid_loss)
                train_loss_history.append(train_loss)
                top1_acc_train_history.append(top1_acc_train)
                top5_acc_train_history.append(top5_acc_train)
                top1_acc_valid_history.append(top1_acc_valid)
                top5_acc_valid_history.append(top5_acc_valid)
                epoch_time_history.append(time_elapsed)
                lr_history.append(optimizer.state_dict()['param_groups'][0]['lr'])

                if rank == 0:
                    print(f'Epoch: {epoch + 1:03d}/{num_epochs:03d} '
                          f'| Validation Loss: {valid_loss:.4f} '
                          f'| Training Loss: {train_loss:.4f} '
                          f'| Top1-Train: {top1_acc_train :.2f}% '
                          f'| Top1-Validation: {top1_acc_valid :.2f}% '
                          f'| Top5-Train: {top5_acc_train :.2f}% '
                          f'| Top5-Validation: {top5_acc_valid :.2f}% '
                          f'| LR: {optimizer.state_dict()["param_groups"][0]["lr"] :.6f} '
                          f'| Time: {time_elapsed :.2f} min')

                # Scheduler Step
                if epoch < warmup_epochs:
                    warmup_scheduler.step()
                else:
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        lr_scheduler.step(valid_loss)
                    else:
                        lr_scheduler.step()

    if rank == 0:
        if batch_iter == 0:
            torch.save(train_loss_history, f'train_loss.pt')
            torch.save(valid_loss_history, f'valid_loss.pt')
            torch.save(top1_acc_train_history, f'train_top1.pt')
            torch.save(top1_acc_valid_history, f'valid_top1.pt')
            torch.save(top5_acc_train_history, f'train_top5.pt')
            torch.save(top5_acc_valid_history, f'valid_top5.pt')
            torch.save(epoch_time_history, f'epoch_times.pt')
            torch.save(lr_history, f'lr.pt')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, "ckpt.tar")

    return valid_loss_history, top1_acc_train_history, top1_acc_valid_history, lr_history, epoch_time_history


@monitor()
def single_forward_step(inp, model):
    output = model(inp)
    return output


@monitor()
def single_loss_step(output, targets):
    loss = torch.nn.functional.cross_entropy(output, targets)
    return loss


@monitor()
def single_backward_step(loss):
    loss.backward()


@monitor()
def data_to_device(features, targets):
    features = features.cuda()
    targets = targets.cuda()
    return features, targets


@monitor()
def single_update_step(optimizer):
    optimizer.step()
