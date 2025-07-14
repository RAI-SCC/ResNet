from pathlib import Path
import os

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


def plot_scaling(result_path, data, scaling_list, name):
    """
    Plots scaling with total times/ gpuh and consumed energy.
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
    energy_mean = {"total" : [], "gpu": [], "cpu": [], "ram": []}
    energy_rmse = {"total" : [], "gpu": [], "cpu": [], "ram": []}
    y2_vals = []
    y2_rmse = []
    num_gpus = []

    for folder in scaling_list:
        #y2_vals.append(data[folder]["mean"]["perun_time"])
        y2_vals.append(data[folder]["mean"]["perun_gpu_h"])
        #y2_rmse.append(data[folder]["rmse"]["perun_time"])
        y2_rmse.append(data[folder]["rmse"]["perun_gpu_h"])
        num_gpus.append(data[folder]["gpus"])

        for key in energy_mean:
            lab = key+"_"
            if key == "total":
                lab = ""
            energy_mean[key].append(data[folder]["mean"][f"perun_{lab}energy"])
            energy_rmse[key].append(data[folder]["rmse"][f"perun_{lab}energy"])
    for key in energy_mean:
        lab = key + "_"
        if key == "total":
            lab = ""

        target_path = Path(result_path, "figs", "energy_scaling")
        os.makedirs(target_path, exist_ok=True)
        target_path = Path(result_path, "figs", "energy_scaling", key)
        os.makedirs(target_path, exist_ok=True)
        target_path = Path(result_path, "figs", "energy_scaling", key, f"{name}_{key}_energy_scaling")

        fig, ax1 = plt.subplots(figsize=(3.5, 1.5))
        ax1.errorbar(range(len(num_gpus)), energy_mean[key], yerr=energy_rmse[key], marker='o', ms=ms, linestyle='-', color="C0",
                     label="GPU", lw=lw, capsize=2, elinewidth=elw)
        ax1.set_xlabel("Number of GPUs", fontsize=fs)
        ax1.set_xticks(range(len(num_gpus)))
        ax1.set_xticklabels(num_gpus, fontsize=fs)
        ax1.set_ylabel("Energy [kWh]", fontsize=fs, color="C0")
        ax1.tick_params(axis='y', labelsize=fs)
        ax2 = ax1.twinx()
        ax2.errorbar(range(len(num_gpus)), y2_vals, yerr=y2_rmse, marker='o', ms=ms, linestyle='-', color="C1",
                     label="Time", lw=lw, capsize=2, elinewidth=elw)
        #ax2.set_ylabel("Time [min]", fontsize=fs, color="C1")
        ax2.set_ylabel("GPU h [h]", fontsize=fs, color="C1")
        ax2.tick_params(axis='y', labelsize=fs)
        plt.savefig(target_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


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


def plot_scaling_power(result_path, data, scaling_list, name):
    """
    Plots power scaling.
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
        y2_vals.append(data[folder]["mean"]["gpu_power_mean"])
        y1_rmse.append(data[folder]["rmse"]["perun_energy"])
        y2_rmse.append(data[folder]["rmse"]["gpu_power_mean"])
        num_gpus.append(data[folder]["gpus"])
    fig, ax1 = plt.subplots(figsize=(3.5, 1.5))
    name = name + "power"
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
    ax2.set_ylabel("Power per GPU [Ws]", fontsize=fs, color="C1")
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


def plot_power(result_path, data, scaling_list, name):
    """
    Plots mean cpu/gpu power.

    Parameters
    ----------
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

    time_vals = []
    gpu_power_vals = []
    cpu_power_vals = []
    num_gpus = []

    for folder in scaling_list:
        time_vals.append(data[folder]["mean"]["perun_time"])
        gpu_power_vals.append(data[folder]["mean"]["gpu_power_mean"])
        cpu_power_vals.append(data[folder]["mean"]["cpu_socket_power_mean"])
        num_gpus.append(data[folder]["gpus"])

    target_path = Path(result_path, "figs", "mean_power_scaling")
    os.makedirs(target_path, exist_ok=True)
    target_path = Path(result_path, "figs", "mean_power_scaling", f"{name}_mean_power")

    fig, ax1 = plt.subplots(figsize=(3.5, 2.0))
    ax1.plot(range(len(num_gpus)), gpu_power_vals, marker='o', ms=ms, linestyle='-', color="C0", label="Single GPU", lw=lw)
    ax1.plot(range(len(num_gpus)), cpu_power_vals, marker='s', ms=ms, linestyle=':', color="C0", label="CPU socket", lw=lw)
    ax1.set_xlabel("Number of GPUs", fontsize=fs)
    ax1.set_xticks(range(len(num_gpus)))
    ax1.set_xticklabels(num_gpus, fontsize=fs)
    ax1.set_ylabel("Power [W]", fontsize=fs, color="C0")
    ax1.tick_params(axis='y', labelsize=fs)
    ax1.legend(fontsize=fs)

    ax2 = ax1.twinx()
    ax2.plot(range(len(num_gpus)), time_vals, marker='o', ms=ms, linestyle='-', color="C1", label="Runtime", lw=lw)
    ax2.set_ylabel("Runtime [min]", fontsize=fs, color="C1")
    ax2.tick_params(axis='y', labelsize=fs)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')


def plot_efficiency(result_path, data, scaling_list, name):
    """
    Plots efficiencies.
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
    ms = 3
    lw = 1
    elw = 1

    energy_vals = []
    time_vals = []
    top1_vals = []
    energy_per_node_vals = []
    gpu_h_vals = []
    num_gpus = []

    for folder in scaling_list:
        energy_vals.append(data[folder]["efficiency"]["perun_energy"])
        time_vals.append(data[folder]["efficiency"]["perun_time"])
        top1_vals.append(data[folder]["efficiency"]["top1_error_valid"])
        energy_per_node_vals.append(data[folder]["efficiency"]["perun_energy_per_node"])
        gpu_h_vals.append(data[folder]["efficiency"]["perun_gpu_h"])
        num_gpus.append(data[folder]["gpus"])

    fig, ax1 = plt.subplots(figsize=(3.5, 2.0))
    name = name + "_eff"
    target_path = Path(result_path, name)
    lines = []

    ax1.axhline(y=1, color='C0', linestyle='--', lw=lw*0.5)

    l, = ax1.plot(range(len(num_gpus)), energy_per_node_vals, marker='_', ms=ms*2, linestyle=':', color="C0", label="Energy/Node", lw=lw)
    lines.append(l)
    l, = ax1.plot(range(len(num_gpus)), time_vals, marker='|', ms=ms*2, linestyle='--', color="C0", label="Runtime", lw=lw)
    lines.append(l)
    l, = ax1.plot(range(len(num_gpus)), top1_vals, marker='d', ms=ms, linestyle='-.', color="C0", label="Top1", lw=lw)
    lines.append(l)

    ax1.set_xlabel("Number of GPUs", fontsize=fs)
    ax1.set_xticks(range(len(num_gpus)))
    ax1.set_xticklabels(num_gpus, fontsize=fs)
    ax1.set_ylabel("Efficiency", fontsize=fs, color="C0")
    ax1.tick_params(axis='y', labelsize=fs)

    ax2 = ax1.twinx()
    ax2.axhline(y=1, color='C1', linestyle='--', lw=lw * 0.5)

    l, = ax2.plot(range(len(num_gpus)), gpu_h_vals, marker='o', ms=ms, linestyle=':', color="C1", label="GPU h", lw=lw)
    lines.append(l)
    l, = ax2.plot(range(len(num_gpus)), energy_vals, marker='s', ms=ms, linestyle='--', color="C1", label="Energy", lw=lw)
    lines.append(l)

    ax2.set_ylabel("Efficiency", fontsize=fs, color="C1")
    ax2.tick_params(axis='y', labelsize=fs)

    labels = [line.get_label() for line in lines]

    ax2.legend(lines, labels, frameon=True, fontsize=fs)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')


def plot_timings(result_path, data, scaling_list, name):
    """
    Plots timings.
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

    epoch_time_total = []

    dataloading_abs = []
    data_to_device_abs = []
    forward_abs = []
    backward_abs = []

    dataloading_sum = []
    data_to_device_sum = []
    forward_sum = []
    backward_sum = []
    total_sum = []

    dataloading_rel = []
    data_to_device_rel = []
    forward_rel = []
    backward_rel = []
    total_rels = []

    num_gpus = []

    for folder in scaling_list:

        epoch_time_total.append(data[folder]["timings"][1]["epoch"]["epoch_time_total"])

        # Absolute values
        dataloading_abs.append(data[folder]["timings"][1]["batch"]["batch_time_dataloading"])
        data_to_device_abs.append(data[folder]["timings"][1]["batch"]["batch_time_data_to_device"])
        forward_abs.append(data[folder]["timings"][1]["batch"]["batch_time_forward"])
        backward_abs.append(data[folder]["timings"][1]["batch"]["batch_time_backward"])

        # Absolute values summed up
        batch_iterations = data[folder]["batch_iterations"]
        dataloading_sum.append(data[folder]["timings"][1]["batch"]["batch_time_dataloading"]*batch_iterations)
        data_to_device_sum.append(data[folder]["timings"][1]["batch"]["batch_time_data_to_device"]*batch_iterations)
        forward_sum.append(data[folder]["timings"][1]["batch"]["batch_time_forward"]*batch_iterations)
        backward_sum.append(data[folder]["timings"][1]["batch"]["batch_time_backward"]*batch_iterations)
        total_sum.append(data[folder]["timings"][1]["batch"]["batch_time_total"] * batch_iterations)

        # Relative contributions
        total_rel = data[folder]["timings"][1]["batch"]["batch_time_total"]
        total_rels.append(total_rel)
        dataloading_rel.append((data[folder]["timings"][1]["batch"]["batch_time_dataloading"] / total_rel)*100)
        data_to_device_rel.append((data[folder]["timings"][1]["batch"]["batch_time_data_to_device"] / total_rel)*100)
        forward_rel.append((data[folder]["timings"][1]["batch"]["batch_time_forward"] / total_rel)*100)
        backward_rel.append((data[folder]["timings"][1]["batch"]["batch_time_backward"] / total_rel)*100)

        # General
        num_gpus.append(data[folder]["gpus"])

    # General plot setup
    fs = 7
    bar_dist = 0.1  # the width of each bar
    target_path_base = Path(result_path, "figs", "timings")
    os.makedirs(target_path_base, exist_ok=True)

    # Plot absolute values
    fig, ax1 = plt.subplots(figsize=(3.5, 2.0))
    target_path = Path(result_path, "figs", "timings", name + "_timings_abs")
    ax1.set_xticks(range(len(num_gpus)))
    tick_position = np.arange(len(num_gpus))
    lines = []
    l = ax1.bar(tick_position - 0.5*bar_dist, dataloading_abs, bar_dist, label='Datalaoding', color="C0", zorder=100)
    lines.append(l)
    l = ax1.bar(tick_position - 1.5*bar_dist, data_to_device_abs, bar_dist, label='Data to GPU', color="C1", zorder=100)
    lines.append(l)
    l = ax1.bar(tick_position + 0.5*bar_dist, forward_abs, bar_dist, label='Forward', color="C2", zorder=100)
    lines.append(l)
    l = ax1.bar(tick_position + 1.5*bar_dist, backward_abs, bar_dist, label='Backward', color="C4", zorder=100)
    lines.append(l)
    ax1.grid(True, axis='y', linestyle='-', alpha=0.5, zorder=-1, lw=0.5)
    ax1.set_xlabel("Number of GPUs", fontsize=fs)
    ax1.set_xticklabels(num_gpus, fontsize=fs)
    ax1.set_ylabel("Times per batch [s]", fontsize=fs)
    ax1.tick_params(axis='y', labelsize=fs, zorder=101)
    ax1.tick_params(axis='x', labelsize=fs, zorder=101)
    ax1.spines['bottom'].set_zorder(101)
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=True, fontsize=fs, ncol=2)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot absolute values summed
    fig, ax1 = plt.subplots(figsize=(3.5, 2.0))
    target_path = Path(result_path, "figs", "timings", name + "_timings_sum")
    ax1.set_xticks(range(len(num_gpus)))
    tick_position = np.arange(len(num_gpus))
    lines = []
    l = ax1.bar(tick_position - 0.5 * bar_dist, dataloading_sum, bar_dist, label='Datalaoding', color="C0", zorder=100)
    lines.append(l)
    l = ax1.bar(tick_position - 1.5 * bar_dist, data_to_device_sum, bar_dist, label='Data to GPU', color="C1", zorder=100)
    lines.append(l)
    l = ax1.bar(tick_position + 0.5 * bar_dist, forward_sum, bar_dist, label='Forward', color="C2", zorder=100)
    lines.append(l)
    l = ax1.bar(tick_position + 1.5 * bar_dist, backward_sum, bar_dist, label='Backward', color="C4", zorder=100)
    lines.append(l)
    ax1.grid(True, axis='y', linestyle='-', alpha=0.5, zorder=-1, lw=0.5)
    ax1.set_xlabel("Number of GPUs", fontsize=fs)
    ax1.set_xticklabels(num_gpus, fontsize=fs)
    ax1.set_ylabel("Time per epoch [s]", fontsize=fs)
    ax1.tick_params(axis='y', labelsize=fs, zorder=101)
    ax1.tick_params(axis='x', labelsize=fs, zorder=101)
    ax1.spines['bottom'].set_zorder(101)
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=True, fontsize=fs, ncol=2)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Plot relative contributions
    fig, ax1 = plt.subplots(figsize=(3.5, 2.0))
    target_path = Path(result_path, "figs", "timings", name + "_timings_rel")
    ax1.set_xticks(range(len(num_gpus)))
    tick_position = np.arange(len(num_gpus))
    lines = []
    l = ax1.bar(tick_position - 0.5 * bar_dist, dataloading_rel, bar_dist, label='Datalaoding', color="C0", zorder=100)
    lines.append(l)
    l = ax1.bar(tick_position - 1.5 * bar_dist, data_to_device_rel, bar_dist, label='Data to GPU', color="C1", zorder=100)
    lines.append(l)
    l = ax1.bar(tick_position + 0.5 * bar_dist, forward_rel, bar_dist, label='Forward', color="C2", zorder=100)
    lines.append(l)
    l = ax1.bar(tick_position + 1.5 * bar_dist, backward_rel, bar_dist, label='Backward', color="C4", zorder=100)
    lines.append(l)
    ax1.grid(True, axis='y', linestyle='-', alpha=0.5, zorder=-1, lw=0.5)
    ax1.set_xlabel("Number of GPUs", fontsize=fs)
    ax1.set_xticklabels(num_gpus, fontsize=fs)
    ax1.set_ylabel("Share of batch Time [%]", fontsize=fs)
    ax1.tick_params(axis='y', labelsize=fs, zorder=101)
    ax1.tick_params(axis='x', labelsize=fs, zorder=101)
    ax1.spines['bottom'].set_zorder(101)
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=True, fontsize=fs, ncol=2)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gpu_mem(result_path, data, scaling_list, name):
    """
    Plots gpu memory and power.
    Parameters
    __________
    result_path : Path
        Path to results.
    data : dict
        Results saved in nested dictionary.
    scaling_list : dict
        Labels saved in dict.
    """
    fs = 7
    lw = 0.5

    for folder in scaling_list:
        gpus = data[folder]["gpus"]
        lbs = data[folder]["lbs"]
        gbs = data[folder]["gbs"]
        nsamples = data[folder]["nsamples"]
        epochs = data[folder]["epochs"]
        nsamples_per_gpu = int(nsamples / gpus)
        nbatch_iter = nsamples_per_gpu / lbs

        for num, slurm_id in enumerate(scaling_list[folder]):
            if num > 0: break
            print(name, folder, slurm_id, data[folder][slurm_id]["gpu_power_mean"], data[folder]["rmse"]["gpu_power_mean"])
            for num_device, device in enumerate(data[folder][slurm_id]["gpu_power"]):
                fig, ax1 = plt.subplots(figsize=(3.5, 2.0))
                title = device.replace(".", "_")
                target_path = Path(result_path, "figs", "memory", name)
                os.makedirs(target_path, exist_ok=True)
                target_path = Path(result_path, "figs", "memory", name, folder)
                os.makedirs(target_path, exist_ok=True)
                target_path = Path(result_path, "figs", "memory", name, folder, title)
                ax1.set_xlabel("Time [s]", fontsize=fs)
                ax1.set_ylabel("Power [Ws]", fontsize=fs, color="C0")
                ax1.tick_params(axis='y', labelsize=fs)
                ax1.tick_params(axis='x', labelsize=fs)
                ax2 = ax1.twinx()
                ax2.set_ylabel("Utilization [GB]", fontsize=fs, color="C1")
                ax2.tick_params(axis='y', labelsize=fs)

                power_vals = data[folder][slurm_id]["gpu_power"][device]
                mem_vals = data[folder][slurm_id]["gpu_mem"][device]
                time_vals = data[folder][slurm_id]["gpu_timesteps"][device]
                num_total_time_steps = power_vals.shape[0]
                last_time = time_vals[-1]
                time_per_epoch = last_time / epochs
                time_per_batch = time_per_epoch / nbatch_iter
                eval_time = time_per_batch * 25 + 35
                #eval_time = time_per_epoch * 2  + 35
                size = time_vals[time_vals <= eval_time].shape[0]

                ax1.plot(time_vals[:size], power_vals[:size], linestyle='-', color="C0", label="Power", lw=lw)
                ax2.plot(time_vals[:size], mem_vals[:size], linestyle='-', color="C1", label="Util", lw=lw)

                plt.savefig(target_path, dpi=300, bbox_inches='tight')
                plt.close(fig)