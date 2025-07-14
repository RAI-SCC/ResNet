from pathlib import Path
import h5py
import numpy as np
import torch

from resnet.eval_utils.read_utils import get_perun_data, get_timings
from resnet.eval_utils.plot_utils import plot_scaling, plot_scaling_per_workload, plot_efficiency
from resnet.eval_utils.plot_utils import plot_timings, plot_gpu_mem, plot_power
from resnet.eval_utils.table_utils import make_table, make_small_table, make_table_with_eff


def make_statistics(data, scaling_list, eval_list):
    """
    Makes statistics.

    Parameters
    ----------
    data : dict
        Dictionary with data.
    scaling_list : dict
        Names of specific folders and slurm IDs within result path. IDs saved as integers in lists within dictionary.
    eval_list : list
        List with keys to be treated here.

    """
    for folder in scaling_list:
        dum = {}
        for key in eval_list:
            dum[key] = []
        data[folder]["mean"] = {}
        data[folder]["rmse"] = {}
        for slurm_id in scaling_list[folder]:
            for key in eval_list:
                dum[key].append(data[folder][slurm_id][key])
        for key in eval_list:
            mean = np.mean(np.array(dum[key]))
            rmse = np.sqrt(np.mean((np.array(dum[key]) - mean) ** 2))
            data[folder]["mean"][key] = mean
            data[folder]["rmse"][key] = rmse
    return data


def calc_efficiency(data, scaling_list, eval_list):
    """
    Evaluates efficiency referenced two lowest GPU configuration.

    Parameters
    ----------
    data : dict
        Dictionary with data.
    scaling_list : dict
        Names of specific folders and slurm IDs within result path. IDs saved as integers in lists within dictionary.
    eval_list : list
        List with keys to be treated here.

    """
    lowest_gpus = 256
    for folder in scaling_list:
        gpus = data[folder]["gpus"]
        data[folder]["efficiency"] = {}
        if gpus <= lowest_gpus:
            lowest_gpus = gpus
            ref_folder = folder

    for folder in scaling_list:
        gpus = data[folder]["gpus"]
        for key in eval_list:
            val = data[folder]["mean"][key] / (data[ref_folder]["mean"][key])
            data[folder]["efficiency"][key] = val
    return data


def eval_scaling(result_path, scaling_list, name):
    """
    Plots strong and weak scaling.
    Parameters
    __________
    result_path : Path
        Path to results
    scaling_list : dict
        Names of specific folders and slurm IDs within result path. IDs saved as integers in lists within dictionary.
    name : str
        Name for the plot to be saved.
    """

    eval_timings = True

    data = {}
    for folder in scaling_list:
        n_images_train = 1281167
        n_images_valid = 50000
        data[folder] = {}
        gpus = int(folder.split("g")[0])
        lbs = int((folder.split("b")[0]).split("g")[-1])
        epochs = int((folder.split("e")[0]).split("w")[-1])
        if 'sf' in folder:
            factor = int((folder.split("sf")[0]).split("e")[-1])
            nsamples_train = int(n_images_train/factor)
            nsamples_valid = int(n_images_valid / factor)
        elif 's' in folder:
            nsamples_train = int((folder.split("s")[0]).split("e")[-1])
            nsamples_valid = n_images_valid
        else:
            nsamples_train = n_images_train
            nsamples_valid = n_images_valid

        gbs = lbs * gpus
        nodes = gpus/4
        if nodes < 1:
            nodes = 1
        data[folder]["gpus"] = gpus
        data[folder]["lbs"] = lbs
        data[folder]["gbs"] = gbs
        data[folder]["nsamples"] = nsamples_train
        data[folder]["nodes"] = nodes
        data[folder]["epochs"] = epochs

        for slurm_id in scaling_list[folder]:
            data[folder][slurm_id] = {}
            print(result_path, folder, slurm_id)

            # Get perun data
            perun_h5_file = Path(result_path, folder, slurm_id, "perun", "perun.hdf5")
            h5val = h5py.File(perun_h5_file, 'r')
            perun_data = get_perun_data(h5val)
            # Get perun total energies
            gpu = []
            cpu = []
            cpu_power_single_device_mean_list = []
            cpu_power = {}
            ram = []
            gpu_power_single_device_mean_list = []
            gpu_mem = {}
            gpu_power = {}
            gpu_timesteps = {}
            shapes = {}
            for key, _ in perun_data.items():
                for num, _ in perun_data[key]["gpu"].items():
                    # energy
                    gpu.append(perun_data[key]["gpu"][num]["energy"][-1] / 10 ** 6)
                    # memory and power
                    gpu_timesteps[f"{key}_{num}"] = perun_data[key]["gpu"][num]["timesteps"]
                    shapes[f"{key}_{num}"] = perun_data[key]["gpu"][num]["timesteps"].shape[0]
                    gpu_mem[f"{key}_{num}"] = perun_data[key]["gpu"][num]["memory"]
                    power = perun_data[key]["gpu"][num]["power"]
                    gpu_power[f"{key}_{num}"] = power
                    # power mean
                    timesteps = perun_data[key]["gpu"][num]["timesteps"]
                    gpu_power_single_device_mean = np.trapz(power, timesteps) / (timesteps[-1] - timesteps[0])
                    gpu_power_single_device_mean_list.append(gpu_power_single_device_mean)

                for num, _ in perun_data[key]["cpu"].items():
                    cpu.append(perun_data[key]["cpu"][num]["energy"][-1] / 10 ** 6)
                    # power mean
                    timesteps = perun_data[key]["cpu"][num]["timesteps"]
                    power = perun_data[key]["cpu"][num]["power"]
                    cpu_power_single_device_mean = np.trapz(power, timesteps)
                    cpu_power_single_device_mean_list.append(cpu_power_single_device_mean / (timesteps[-1] - timesteps[0]))
                    cpu_power[f"{key}_{num}"] = power
                    #freqs = np.array([perun_data[key]["cpu"][num]["cpu_freqs"][cpu_num] for cpu_num in sorted(perun_data[key]["cpu"][num]["cpu_freqs"])])
                    #print(np.mean(freqs), min(freqs), max(freqs), freqs.shape)
                for num, _ in perun_data[key]["ram"].items():
                    ram.append(perun_data[key]["ram"][num]["energy"][-1] / 10 ** 6)
            gpu_total = np.array(gpu).sum()
            cpu_total = np.array(cpu).sum()
            ram_total = np.array(ram).sum()
            perun_energy = gpu_total + cpu_total + ram_total
            gpu_power_mean = np.array(gpu_power_single_device_mean_list).sum() / gpus
            cpu_socket_power_mean = np.array(cpu_power_single_device_mean_list).sum() / (2*nodes)
            # Get perun last time:
            time_list = []
            for key, _ in perun_data.items():
                for num, _ in perun_data[key]["gpu"].items():
                    time_list.append(perun_data[key]["gpu"][num]["timesteps"][-1])
                for num, _ in perun_data[key]["cpu"].items():
                    time_list.append(perun_data[key]["cpu"][num]["timesteps"][-1])
                for num, _ in perun_data[key]["ram"].items():
                    time_list.append(perun_data[key]["ram"][num]["timesteps"][-1])
            perun_time = (max(time_list)) / 60
            # Get Top1
            path = Path(result_path, folder, slurm_id, "valid_top1.pt")
            top1_valid_acc = torch.load(path)
            path = Path(result_path, folder, slurm_id, "valid_top5.pt")
            top5_valid_acc = torch.load(path)
            path = Path(result_path, folder, slurm_id, "train_top1.pt")
            top1_train_acc = torch.load(path)
            path = Path(result_path, folder, slurm_id, "train_top5.pt")
            top5_train_acc = torch.load(path)

            # Get slurm data
            path = Path(result_path, folder, slurm_id, f"slurm_{slurm_id}")
            with open(path, "r") as sfile:
                for line in sfile:
                    if "Energy Consumed:" in line:
                        slurm_energy = float(line.split()[-2])
                    if "Job Wall-clock time:" in line:
                        slurm_time_string = line.split()[-1]
                        slurm_time_d = 0
                        if "-" in slurm_time_string:
                            slurm_time_d = slurm_time_string.split("-")[0]
                            slurm_time_string = slurm_time_string.split("-")[1]
                        slurm_time_h = slurm_time_string.split(":")[0]
                        slurm_time_min = slurm_time_string.split(":")[1]
                        slurm_time_s = slurm_time_string.split(":")[2]
                        slurm_time = float(slurm_time_d)*60*24 + float(slurm_time_h)*60 + float(slurm_time_min) + float(slurm_time_s)/60

            data[folder][slurm_id]["perun_time"] = perun_time
            data[folder][slurm_id]["perun_time_per_node"] = data[folder][slurm_id]["perun_time"] / nodes
            data[folder][slurm_id]["perun_time_per_workload"] = data[folder][slurm_id]["perun_time"] * gpus / n_images_train
            data[folder][slurm_id]["perun_gpu_h"] = (data[folder][slurm_id]["perun_time"] * gpus)/60
            data[folder][slurm_id]["perun_energy"] = perun_energy/3.6  # MJ in kWh
            data[folder][slurm_id]["perun_gpu_energy"] = gpu_total / 3.6  # MJ in kWh
            data[folder][slurm_id]["perun_cpu_energy"] = cpu_total / 3.6  # MJ in kWh
            data[folder][slurm_id]["perun_ram_energy"] = ram_total / 3.6  # MJ in kWh
            data[folder][slurm_id]["perun_energy_per_node"] = data[folder][slurm_id]["perun_energy"] / nodes
            data[folder][slurm_id]["perun_energy_per_workload"] = data[folder][slurm_id]["perun_energy"] * gpus / n_images_train
            data[folder][slurm_id]["top1_error_valid"] = 100 - top1_valid_acc[-1]
            data[folder][slurm_id]["top5_error_valid"] = 100 - top5_valid_acc[-1]
            data[folder][slurm_id]["top1_error_train"] = 100 - top1_train_acc[-1]
            data[folder][slurm_id]["top5_error_train"] = 100 - top5_train_acc[-1]
            data[folder][slurm_id]["slurm_energy"] = slurm_energy/1000  # Wh in KWh
            data[folder][slurm_id]["slurm_time"] = slurm_time
            data[folder][slurm_id]["gpu_power"] = gpu_power
            data[folder][slurm_id]["gpu_mem"] = gpu_mem
            data[folder][slurm_id]["gpu_timesteps"] = gpu_timesteps
            data[folder][slurm_id]["gpu_power_mean"] = gpu_power_mean
            data[folder][slurm_id]["cpu_socket_power_mean"] = cpu_socket_power_mean
            #data[folder][slurm_id]["cpu_power_single_device_mean"] = np.array(cpu_power_single_device_mean_list)

        # Get timing data
        if "sf" in folder:
            timing_folder = Path(str(folder).split("4w")[0] + "4w2e" + str(factor) + "sf")
        else:
            timing_folder = Path(str(folder).split("4w")[0] + "4w2e")
        timing_folder = Path(result_path, timing_folder)
        timing_paths = [item for item in timing_folder.iterdir() if item.is_dir()]
        timing_id = 0
        for timing_path in timing_paths:
            timing_id = int(str(timing_path).split("/")[-1])
            int_id = int(timing_id)
            if int_id > timing_id:
                timing_id = int_id
        timing_file = Path(timing_folder, str(timing_id), "times.h5")
        h5val = h5py.File(timing_file, 'r')
        data = get_timings(h5val, data, folder)


    eval_list = []
    exclude_list = ["rmses", "gpu_power", "gpu_mem", "gpu_timesteps"]
    for key in data[folder][slurm_id]:
        if key in exclude_list: continue
        eval_list.append(key)

    # Statistics
    data = make_statistics(data, scaling_list, eval_list)

    # efficiency
    data = calc_efficiency(data, scaling_list, eval_list)

    # Plot it
    plot_scaling(result_path, data, scaling_list, name)
    #plot_scaling_per_workload(result_path, data, scaling_list, name)
    #plot_efficiency(result_path, data, scaling_list, name)
    plot_timings(result_path, data, scaling_list, name)
    plot_gpu_mem(result_path, data, scaling_list, name)
    plot_power(result_path, data, scaling_list, name)

    # make table
    #make_table(result_path, data, scaling_list, name)
    make_table_with_eff(result_path, data, name)
