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

fs = 6
ms = 0.5
lw = 0.1
alpha = 0.9
lp = 2
elw = 1
cs = 2
pad = 1.5
rotation = 0
bar_dist = 0.1

base_path = "/Users/philipphuber/Documents/Projects/ResNet/rebuttal/A100_higher_resolution_for_profiles/"
#add_path = "16g4b1e/afno_backbone/3664004/perun_results/perun.hdf5"
add_path = "16g256b4w1e/3665939/perun/perun.hdf5"
name = "single_power_profile"

exp = add_path.split("/")[0]
gpu = int(exp.split("g")[0])
lbs = int((exp.split("b")[0]).split("g")[-1])
gbs = gpu * lbs
slurm_id = add_path.split("/")[2]

# Get perun h5 file
perun_h5_file = Path(base_path, add_path)
h5val = h5py.File(perun_h5_file, 'r')

# Get perun data
data = {}
h5id, _ = next(iter(h5val["perun/nodes"].items()))
h5_base_path = "perun/nodes/" + h5id + "/nodes/0/nodes"
for node_id, node_obj in h5val[h5_base_path].items():
    h5_node_path = "perun/nodes/" + h5id + "/nodes/0/nodes/" + node_id + "/nodes/gpu/nodes"
    data[node_id] = {}
    for num in [0, 1, 2, 3]:
        data[node_id][num] = {}
        value_path = h5_node_path + f"/CUDA:{num}_POWER/raw_data/values"
        timestep_path = h5_node_path + f"/CUDA:{num}_POWER/raw_data/timesteps"
        print(perun_h5_file)
        print(value_path)
        power = np.array(h5val[value_path])
        mag = float(h5val[value_path].attrs["mag"])
        power = power * mag
        timesteps = np.array(h5val[timestep_path])
        data[node_id][num][("power")] = power
        data[node_id][num][("timesteps")] = timesteps

        target_path = f"{base_path}/profile_{gpu}_{lbs}_{slurm_id}_{node_id.split('.')[0]}_{num}"

        v_list = []
        val_list = []
        bottom_count = 0
        start = False
        bottom_bool = False
        for v, val in enumerate(power):
            if val > 150:
                start = True
            if start:
                if val < 150:
                    if not bottom_bool:
                        bottom_bool = True
                        bottom_count += 1
                        v_list.append(v)
                        val_list.append(val)
                if bottom_bool and val > 150:
                    bottom_bool = False
        time_list = [timesteps[v] for v in v_list]
        print(target_path, bottom_count)\

        fig, ax1 = plt.subplots(figsize=(3.5, 2.0))

        ax1.plot(timesteps, power, linestyle='-', lw=lw, color="C8")
        #ax1.plot(time_list, val_list, linestyle="none", marker="o", ms=ms, color="C5")

        ax1.set_xlabel("Time (s)", fontsize=fs)
        ax1.set_ylabel("Power (W)", fontsize=fs, labelpad=lp)
        ax1.tick_params(axis='y', labelsize=fs)
        ax1.tick_params(axis='x', labelsize=fs)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)

        plt.savefig(target_path, dpi=300, bbox_inches='tight')
        plt.close()