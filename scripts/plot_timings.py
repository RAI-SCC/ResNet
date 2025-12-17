from pathlib import Path

import numpy as np
import h5py
from matplotlib import pyplot as plt
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


def get_paths(basepath: str = ""):
    paths = {}
    #paths["4g256b1e"] = Path(f"{basepath}/4g256b4w1e/3661140/times.h5")
    #paths["8g256b1e"] = Path(f"{basepath}/8g256b4w1e/3661139/times.h5")
    #paths["16g256b1e"] = Path(f"{basepath}/16g256b4w1e/3661138/times.h5")
    #paths["32g256b1e"] = Path(f"{basepath}/32g256b4w1e/3661137/times.h5")
    #paths["64g256b1e"] = Path(f"{basepath}/64g256b4w1e/3661136/times.h5")

    #paths["4g64b1e"] = Path(f"{basepath}/4g64b4w1e/3661158/times.h5")
    #paths["8g32b1e"] = Path(f"{basepath}/8g32b4w1e/3661152/times.h5")
    #paths["16g16b1e"] = Path(f"{basepath}/16g16b4w1e/3661151/times.h5")

    paths["4g256b1e"] = Path(f"{basepath}/4g256b4w1e/3661140/times.h5")
    paths["8g128b1e"] = Path(f"{basepath}/8g128b4w1e/3663065/times.h5")
    paths["16g64b1e"] = Path(f"{basepath}/16g64b4w1e/3663064/times.h5")
    paths["32g32b1e"] = Path(f"{basepath}/32g32b4w1e/3663063/times.h5")
    paths["64g16b1e"] = Path(f"{basepath}/64g16b4w1e/3663062/times.h5")
    return paths


def read_paths(paths):
    data = {}
    for exp in paths:
        data[exp] = {}
        h5val = h5py.File(paths[exp], 'r')
        for rank_id, rank_obj in h5val.items():
            data[exp][rank_id] = {}
            for epoch_id, epoch_obj in h5val[rank_id].items():
                epoch_int = int(epoch_id.split("e")[-1])
                if int(epoch_int) > 1:
                    break
                data[exp][rank_id]["batch_time_backward"] = np.array(h5val[rank_id][epoch_id]["batch_time_backward"][3:78])
                data[exp][rank_id]["batch_time_data_to_device"] = np.array(h5val[rank_id][epoch_id]["batch_time_data_to_device"][3:78])
                data[exp][rank_id]["batch_time_dataloading"] = np.array(h5val[rank_id][epoch_id]["batch_time_dataloading"][3:78])
                data[exp][rank_id]["batch_time_forward"] = np.array(h5val[rank_id][epoch_id]["batch_time_forward"][3:78])
                data[exp][rank_id]["batch_time_total"] = np.array(h5val[rank_id][epoch_id]["batch_time_total"][3:78])
    return data


def get_statistics(data):
    stats = {}
    phases = ["batch_time_backward", "batch_time_data_to_device", "batch_time_dataloading", "batch_time_forward", "batch_time_total"]
    full_lists = {}
    for exp in data:
        full_lists[exp] = {}
        stats[exp] = {}
        for phase in phases:
            full_lists[exp][phase] = []
            for rank in data[exp]:
                full_lists[exp][phase] += list(data[exp][rank][phase])[3:]
            mean = np.mean(np.array(full_lists[exp][phase]))
            rmse = np.sqrt(np.mean((np.array(full_lists[exp][phase]) - mean) ** 2))
            stats[exp][phase] = {}
            stats[exp][phase]["mean"] = mean
            stats[exp][phase]["rmse"] = rmse
    return stats


def plot_stats(stats, basepath):

    target_path = basepath + "/resnet_timings"
    fig, ax1 = plt.subplots(figsize=(3.5, 2.0))

    fs = 6
    ms = 2
    lw = 0.5
    alpha = 0.9
    lp = 2
    elw = 1
    cs = 2
    pad = 1.5
    rotation = 0
    bar_dist = 0.1

    gpu_list = []
    lbs_list = []
    gbs_list = []
    exp_list = []

    for exp in stats:
        n_gpus = int(exp.split("g")[0])
        lbs = int((exp.split("g")[1]).split("b")[0])
        gbs = lbs * n_gpus
        gpu_list.append(n_gpus)
        lbs_list.append(lbs)
        gbs_list.append(gbs)
        exp_list.append(exp)

    x_tick_pos = range(len(gpu_list))
    ax1.set_xlim(min(x_tick_pos)-0.5, max(x_tick_pos)+0.5)
    ax1.set_ylim(0, 0.25)
    #ax1.set_ylim(0, 0.07)
    ax1.set_xlabel("# GPUs", fontsize=fs)
    ax1.set_xticks(x_tick_pos)
    ax1.set_xticklabels(gpu_list, fontsize=fs, rotation=rotation)
    ax1.set_ylabel("Time (s)", fontsize=fs, labelpad=lp)
    ax1.tick_params(axis='y', labelsize=fs)
    ax1.tick_params(axis='x', labelsize=fs)
    ax1.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)

    ax2 = ax1.twiny()
    ax2.set_xlim(min(x_tick_pos) - 0.5, max(x_tick_pos) + 0.5)
    ax2.set_xlabel("LBS", fontsize=fs)
    ax2.set_xticks(x_tick_pos)
    ax2.set_xticklabels(lbs_list, fontsize=fs, rotation=rotation)

    for i, exp in enumerate(exp_list):
        dataloading_time_mean = stats[exp]["batch_time_dataloading"]["mean"]
        dataloading_time_mean += stats[exp]["batch_time_data_to_device"]["mean"]
        forward_time_mean = stats[exp]["batch_time_forward"]["mean"]
        backward_time_mean = stats[exp]["batch_time_backward"]["mean"]
        total_time_mean = stats[exp]["batch_time_total"]["mean"]

        dataloading_time_rmse = stats[exp]["batch_time_dataloading"]["rmse"]
        dataloading_time_rmse += stats[exp]["batch_time_data_to_device"]["rmse"]
        forward_time_rmse = stats[exp]["batch_time_forward"]["rmse"]
        backward_time_rmse = stats[exp]["batch_time_backward"]["rmse"]
        print(exp, backward_time_mean, dataloading_time_mean, total_time_mean)

        l1 = ax1.bar(x_tick_pos[i] - 1.0 * bar_dist, dataloading_time_mean, bar_dist, yerr=dataloading_time_rmse,
                     linewidth=lw, capsize=cs, label='Dataloading', color="C0", zorder=2)
        l3 = ax1.bar(x_tick_pos[i] + 0.0 * bar_dist, forward_time_mean, bar_dist, yerr=forward_time_rmse,
                     linewidth=lw, capsize=cs, label='Forward', color="C2", zorder=2)
        l4 = ax1.bar(x_tick_pos[i] + 1.0 * bar_dist, backward_time_mean, bar_dist, yerr=backward_time_rmse,
                     linewidth=lw, capsize=cs, label='Backward', color="C3", zorder=2)
        lines = [l1, l3, l4]

    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=True, fontsize=fs, ncol=1)
    #ax1.legend(lines, labels, frameon=True, fontsize=fs, ncol=3)

    plt.savefig(target_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    basepath = "/Users/philipphuber/Documents/Projects/ResNet/rebuttal/experiments/scaling"
    paths = get_paths(basepath)
    data = read_paths(paths)
    stats = get_statistics(data)
    plot_stats(stats, basepath)
