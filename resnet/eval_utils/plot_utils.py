from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

from resnet.eval_utils.read_utils import get_perun_data


def plot_scaling_time(result_path, name, gpus, times, total_energies):
    """
    Plots strong and weak scaling with total run times and consumed energy.
    Parameters
    __________
    result_path : Path
        Path to results
    name : str
        Name for the plot to be saves.
    gpus : list
        Number of gpus for each experiment
    time : list
        Total run time for each experiment
    total_energies : list
        Consumed energy for each experiment
    """
    fs = 7
    ms = 3
    lw = 1
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    name = name + "_with_times"
    target_path = Path(result_path, name)
    ax1.plot(range(len(gpus)), total_energies, marker='o', ms=ms, linestyle='-', color='C0', label="Energy", lw=lw)
    ax1.set_xlabel("GPUs", fontsize=fs)
    ax1.set_xticks(range(len(gpus)))
    ax1.set_xticklabels(gpus, fontsize=fs)
    ax1.set_ylabel("Energy / MJ", fontsize=fs, color="C0")
    ax1.tick_params(axis='y', labelsize=fs)
    ax2 = ax1.twinx()
    ax2.plot(range(len(gpus)), times, marker='o', ms=ms, linestyle='-', color='C1', label="Time", lw=lw)
    ax2.set_ylabel("Time / min", fontsize=fs, color="C1")
    ax2.tick_params(axis='y', labelsize=fs)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')


def plot_scaling_top1(result_path, name, gpus, top1, total_energies):
    """
    Plots strong and weak scaling with top1 error and consumed energy.
    Parameters
    __________
    result_path : Path
        Path to results
    name : str
        Name for the plot to be saves.
    gpus : list
        Number of gpus for each experiment
    top1 : list
        Top1 error for each experiment
    total_energies : list
        Consumed energy for each experiment
    """
    fs = 7
    ms = 3
    lw = 1
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    name = name + "_with_top1"
    target_path = Path(result_path, name)
    ax1.plot(range(len(gpus)), total_energies, marker='o', ms=ms, linestyle='-', color='C0', label="Energy", lw=lw)
    ax1.set_xlabel("GPUs", fontsize=fs)
    ax1.set_xticks(range(len(gpus)))
    ax1.set_xticklabels(gpus, fontsize=fs)
    ax1.set_ylabel("Energy / MJ", fontsize=fs, color="C0")
    ax1.tick_params(axis='y', labelsize=fs)
    ax2 = ax1.twinx()
    ax2.plot(range(len(gpus)), top1, marker='o', ms=ms, linestyle='-', color='C1', label="Top1 Error", lw=lw)
    ax2.set_ylabel("Top1 Error / %", fontsize=fs, color="C1")
    ax2.tick_params(axis='y', labelsize=fs)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')


def plot_top1(result_path, scaling_list, name):
    """
    Plot the Top1 errors versus epochs.
    Parameters
    __________
    result_path : Path
        Path to results
    scaling_list : list
        Names of specific folders within result path. Saved as strings.
    name : str
        Name for the plot to be saves.
    """
    fs = 7
    lw = 1
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    name = name + "_top1error"
    target_path = Path(result_path, name)
    gpus = []
    top1_valid_error = []
    for folder in scaling_list:
        # Get setup
        num_gpus = int(folder.split("g")[0])
        lbs = int((folder.split("b")[0]).split("g")[-1])
        gbs = lbs * num_gpus
        gpus.append(num_gpus)

        # Get Top1
        path = Path(result_path, folder, "valid_top1.pt")
        top1_valid_acc = torch.load(path)
        top1_valid_error = [100 - val for val in top1_valid_acc]

        ax1.plot(range(len(top1_valid_error)), top1_valid_error, linestyle='-', label=f"GPUs: {num_gpus}, gbs: {gbs}, lbs: {lbs}", lw=lw)

    ax1.set_xlabel("Epochs", fontsize=fs)
    ax1.set_xlim(0, len(top1_valid_error)-1)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    xticks = np.arange(9, len(top1_valid_error), 10)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(int(tick+1)) for tick in xticks], fontsize=fs)
    ax1.set_ylabel("Top1 error / %", fontsize=fs)
    ax1.tick_params(axis='y', labelsize=fs)
    ax1.legend(loc='upper right', fontsize=fs)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')


def plot_scaling(result_path, scaling_list, name):
    """
    Plots strong and weak scaling.
    Parameters
    __________
    result_path : Path
        Path to results
    scaling_list : list
        Names of specific folders within result path. Saved as strings.
    name : str
        Name for the plot to be saves.
    """
    times = []
    total_energies = []
    gpus = []
    top1_valid_errors = []

    for folder in scaling_list:
        print(folder)
        # Get setup
        num_gpus = int(folder.split("g")[0])
        lbs = int((folder.split("b")[0]).split("g")[-1])
        gbs = lbs * num_gpus
        gpus.append(num_gpus)

        # Get perun data
        perun_h5_file = Path(result_path, folder, "perun", "perun.hdf5")
        h5val = h5py.File(perun_h5_file, 'r')
        perun_data = get_perun_data(h5val)
        # Get Total energies
        gpu = []
        cpu = []
        ram = []
        for key, _ in perun_data.items():
            for num, _ in perun_data[key]["gpu"].items():
                gpu.append(perun_data[key]["gpu"][num]["energy"][-1] / 10 ** 6)
            for num, _ in perun_data[key]["cpu"].items():
                cpu.append(perun_data[key]["cpu"][num]["energy"][-1] / 10 ** 6)
            for num, _ in perun_data[key]["ram"].items():
                ram.append(perun_data[key]["ram"][num]["energy"][-1] / 10 ** 6)
        gpu_total = np.array(gpu).sum()
        cpu_total = np.array(cpu).sum()
        ram_total = np.array(ram).sum()
        total_energy = gpu_total + cpu_total + ram_total
        total_energies.append(total_energy)
        # Get perun last time:
        time_list = []
        for key, _ in perun_data.items():
            for num, _ in perun_data[key]["gpu"].items():
                time_list.append(perun_data[key]["gpu"][num]["timesteps"][-1])
            for num, _ in perun_data[key]["cpu"].items():
                time_list.append(perun_data[key]["cpu"][num]["timesteps"][-1])
            for num, _ in perun_data[key]["ram"].items():
                time_list.append(perun_data[key]["ram"][num]["timesteps"][-1])
        total_time = (max(time_list)) / 60
        times.append(total_time)

        # Get Top1
        path = Path(result_path, folder, "valid_top1.pt")
        top1_valid_acc = torch.load(path)
        top1_valid_errors.append(100 - top1_valid_acc[-1])

    # Plot it
    plot_scaling_time(result_path, name, gpus, times, total_energies)
    plot_scaling_top1(result_path, name, gpus, top1_valid_errors, total_energies)
