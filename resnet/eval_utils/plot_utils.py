import matplotlib.pyplot as plt

def plot_loss(loss, time):
    """
    Plot the loss versus time and epochs.
    Parameters
    __________
    loss : list
        Loss
    time : list
        Corresponding time.

    """
    fig, ax1 = plt.subplots()
    ax1.plot(time, loss,  marker='o', linestyle='-', color='b', label="Epoch")
    ax1.set_xlabel("Time / min")
    ax1.set_ylabel("Loss")

    ax2 = ax1.secondary_xaxis('top')
    ax2.set_xticks(time)
    ax2.set_xticklabels(range(len(time)))
    ax2.set_xlabel("Epochs")

    plt.show()