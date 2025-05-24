from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from cycler import cycler


kitcolors = {
    "brown": (167 / 255, 130 / 255, 46 / 255),
    "purple": (163 / 255, 16 / 255, 124 / 255),
    "cyan": (35 / 255, 161 / 255, 224 / 255),
    "peagreen": (140 / 255, 182 / 255, 60 / 255),
    "yellow": (252 / 255, 229 / 255, 0 / 255),
    "orange": (223 / 255, 155 / 255, 27 / 255),
    "red": (162 / 255, 34 / 255, 35 / 255),
    "green": (0 / 255, 150 / 255, 130 / 255),
    "blue": (70 / 255, 100 / 255, 170 / 255),
    "white": (0 / 255, 150 / 255, 130 / 255),
    "black100": (0 / 255, 0 / 255, 0 / 255),
    "black70": (64 / 255, 64 / 255, 64 / 255),
}
plt.rcParams['axes.prop_cycle'] = cycler(color=list(kitcolors.values()))


def plot_single_device(power, time, label):
    fs = 7
    lw = 0.5
    ms = 0.5
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(time, power, linestyle='', markersize=ms, marker=".", label=label, lw=lw)
    ax1.set_xlabel("Epochs", fontsize=fs)
    ax1.grid(True, which='both', linestyle='--', linewidth=lw, color='gray')
    ax1.set_ylabel(label, fontsize=fs)
    ax1.tick_params(axis='y', labelsize=fs)
    plt.show()


def plot_scaling(result_path, data, scaling_list, name):
    """
    Plots strong and weak scaling with total run times and consumed energy.
    Parameters
    __________
    result_path : Path
        Path to results.
    data : dict
        Results saved in nested dictionary.
    scaling_list : dict
        Labels saved in dict.
    name : str
        Name for the plot to be saves.
    """
    fs = 7
    ms = 2
    lw = 1
    elw = 1
    y1_vals = []
    y2_vals = []
    y1_rmse = []
    y2_rmse = []
    num_gpus = []
    for folder in scaling_list:
        y1_vals.append(data[folder]["mean"]["perun_energy"])
        y2_vals.append(data[folder]["mean"]["perun_time"])
        y1_rmse.append(data[folder]["rmse"]["perun_energy"])
        y2_rmse.append(data[folder]["rmse"]["perun_time"])
        num_gpus.append(data[folder]["gpus"])
    fig, ax1 = plt.subplots(figsize=(3.5, 1.5))
    name = name + "_perun_data"
    target_path = Path(result_path, name)
    ax1.errorbar(range(len(num_gpus)), y1_vals, yerr=y1_rmse, marker='o', ms=ms, linestyle='-', color="C0",
                 label="Energy", lw=lw, capsize=2, elinewidth=elw)
    ax1.set_xlabel("Number of GPUs", fontsize=fs)
    ax1.set_xticks(range(len(num_gpus)))
    ax1.set_xticklabels(num_gpus, fontsize=fs)
    ax1.set_ylabel("Energy [kWh]", fontsize=fs, color="C0")
    ax1.tick_params(axis='y', labelsize=fs)
    ax2 = ax1.twinx()
    ax2.errorbar(range(len(num_gpus)), y2_vals, yerr=y2_rmse, marker='o', ms=ms, linestyle='-', color="C1",
                 label="Time", lw=lw, capsize=2, elinewidth=elw)
    ax2.set_ylabel("Time [min]", fontsize=fs, color="C1")
    ax2.tick_params(axis='y', labelsize=fs)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')


def plot_scaling_per_gpu(result_path, data, scaling_list, name):
    """
    Plots strong and weak scaling with total run times and consumed energy.
    Parameters
    __________
    result_path : Path
        Path to results.
    data : dict
        Results saved in nested dictionary.
    scaling_list : dict
        Labels saved in dict.
    name : str
        Name for the plot to be saves.
    """
    fs = 7
    ms = 2
    lw = 1
    elw = 1
    y1_vals = []
    y2_vals = []
    y1_rmse = []
    y2_rmse = []
    num_gpus = []
    for folder in scaling_list:
        y1_vals.append(data[folder]["mean"]["perun_energy_per_gpu"])
        y2_vals.append(data[folder]["mean"]["perun_time_per_gpu"])
        y1_rmse.append(data[folder]["rmse"]["perun_energy_per_gpu"])
        y2_rmse.append(data[folder]["rmse"]["perun_time_per_gpu"])
        num_gpus.append(data[folder]["gpus"])
    fig, ax1 = plt.subplots(figsize=(3.5, 1.5))
    name = name + "_perun_data_per_node"
    target_path = Path(result_path, name)
    ax1.errorbar(range(len(num_gpus)), y1_vals, yerr=y1_rmse, marker='o', ms=ms, linestyle='-', color="C0",
                 label="Energy", lw=lw, capsize=2, elinewidth=elw)
    ax1.set_xlabel("Number of GPUs", fontsize=fs)
    ax1.set_xticks(range(len(num_gpus)))
    ax1.set_xticklabels(num_gpus, fontsize=fs)
    ax1.set_ylabel("Energy / GPUs [kWh]", fontsize=fs, color="C0")
    ax1.tick_params(axis='y', labelsize=fs)
    ax2 = ax1.twinx()
    ax2.errorbar(range(len(num_gpus)), y2_vals, yerr=y2_rmse, marker='o', ms=ms, linestyle='-', color="C1",
                 label="Time", lw=lw, capsize=2, elinewidth=elw)
    ax2.set_ylabel("Time / GPUs [min]", fontsize=fs, color="C1")
    ax2.tick_params(axis='y', labelsize=fs)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')


def plot_scaling_per_workload(result_path, data, scaling_list, name):
    """
    Plots strong and weak scaling with total run times and consumed energy.
    Parameters
    __________
    result_path : Path
        Path to results.
    data : dict
        Results saved in nested dictionary.
    scaling_list : dict
        Labels saved in dict.
    name : str
        Name for the plot to be saves.
    """
    fs = 7
    ms = 2
    lw = 1
    elw = 1
    y1_vals = []
    y2_vals = []
    y1_rmse = []
    y2_rmse = []
    num_gpus = []
    for folder in scaling_list:
        y1_vals.append(data[folder]["mean"]["perun_energy_per_workload"])
        y2_vals.append(data[folder]["mean"]["perun_time_per_workload"])
        y1_rmse.append(data[folder]["rmse"]["perun_energy_per_workload"])
        y2_rmse.append(data[folder]["rmse"]["perun_time_per_workload"])
        num_gpus.append(data[folder]["gpus"])
    fig, ax1 = plt.subplots(figsize=(3.5, 1.5))
    name = name + "_perun_data_per_workload"
    target_path = Path(result_path, name)
    ax1.errorbar(range(len(num_gpus)), y1_vals, yerr=y1_rmse, marker='o', ms=ms, linestyle='-', color="C0",
                 label="Energy", lw=lw, capsize=2, elinewidth=elw)
    ax1.set_xlabel("Number of GPUs", fontsize=fs)
    ax1.set_xticks(range(len(num_gpus)))
    ax1.set_xticklabels(num_gpus, fontsize=fs)
    ax1.set_ylabel("Energy / Workload [kWh]", fontsize=fs, color="C0")
    ax1.tick_params(axis='y', labelsize=fs)
    ax2 = ax1.twinx()
    ax2.errorbar(range(len(num_gpus)), y2_vals, yerr=y2_rmse, marker='o', ms=ms, linestyle='-', color="C1",
                 label="Time", lw=lw, capsize=2, elinewidth=elw)
    ax2.set_ylabel("Time / Workload [min]", fontsize=fs, color="C1")
    ax2.tick_params(axis='y', labelsize=fs)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')


def plot_top1(result_path, scaling_list, name, key):
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
    key : str
        Plot for strong scaling list or weak scaling list.
    """
    fs = 7
    lw = 1
    exp_list = [2**n for n in range(21)]
    k_list = [str(n) for n in exp_list[0:10]] + [str(2**n)+"k" for n in range(12)]
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

        if key == "ws":
            for i, val in enumerate(exp_list):
                if gbs == val:
                    label = f"n: {num_gpus}, g: {k_list[i]}"
        elif key == "ss":
            label = f"n: {num_gpus}, l: {lbs}"
        else:
            label = f"GPUs: {num_gpus}, gbs: {gbs}, lbs: {lbs}"
        ax1.plot(range(len(top1_valid_error)), top1_valid_error, linestyle='-', label=label, lw=lw)

    ax1.set_xlabel("Epochs", fontsize=fs)
    ax1.set_xlim(0, len(top1_valid_error)-1)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    xticks = np.arange(9, len(top1_valid_error), 10)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(int(tick+1)) for tick in xticks], fontsize=fs)
    ax1.set_ylabel("Top1 error / %", fontsize=fs)
    ax1.tick_params(axis='y', labelsize=fs)
    ax1.legend(loc='upper right', fontsize=fs)  # , handlelength=1.0)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')
