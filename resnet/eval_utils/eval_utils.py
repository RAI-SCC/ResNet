from pathlib import Path
import h5py
import numpy as np
import torch

from resnet.eval_utils.read_utils import get_perun_data
from resnet.eval_utils.plot_utils import plot_scaling, plot_scaling_per_gpu
from resnet.eval_utils.table_utils import make_table


def get_total_energy(h5val: h5py = None):
    """
    Calculates the total energy consumed.

    Parameters
    ----------
    h5val : h5py
        Value vor hdf5 file with results
    """
    perun_data = get_perun_data(h5val)
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

    print(f" Energies in MJ:\n",
          f"Total GPU energy: {gpu_total:.4f}\n",
          f"Total CPU energy: {cpu_total:.4f}\n",
          f"Total RAM energy: {ram_total:.4f}\n",
          f"Total energy:     {total_energy:.4f}"
          )


def make_statistics(data, scaling_list):

    for folder in scaling_list:
        perun_time_list = []
        perun_time_per_gpu_list = []
        perun_energy_list = []
        perun_energy_per_gpu_list = []
        top1_error_valid_list = []
        top5_error_valid_list = []
        top1_error_train_list = []
        top5_error_train_list = []
        slurm_energy_list = []
        slurm_time_list = []
        data[folder]["mean"] = {}
        data[folder]["rmse"] = {}
        for slurm_id in scaling_list[folder]:
            perun_time_list.append(data[folder][slurm_id]["perun_time"])
            perun_time_per_gpu_list.append(data[folder][slurm_id]["perun_time_per_gpu"])
            perun_energy_list.append(data[folder][slurm_id]["perun_energy"])
            perun_energy_per_gpu_list.append(data[folder][slurm_id]["perun_energy_per_gpu"])
            top1_error_valid_list.append(data[folder][slurm_id]["top1_error_valid"])
            top5_error_valid_list.append(data[folder][slurm_id]["top5_error_valid"])
            top1_error_train_list.append(data[folder][slurm_id]["top1_error_train"])
            top5_error_train_list.append(data[folder][slurm_id]["top5_error_train"])
            slurm_energy_list.append(data[folder][slurm_id]["slurm_energy"])
            slurm_time_list.append(data[folder][slurm_id]["slurm_time"])
        mean = np.mean(np.array(perun_time_list))
        rmse = np.sqrt(np.mean((np.array(perun_time_list) - mean) ** 2))
        data[folder]["mean"]["perun_time"] = mean
        data[folder]["rmse"]["perun_time"] = rmse
        mean = np.mean(np.array(perun_time_per_gpu_list))
        rmse = np.sqrt(np.mean((np.array(perun_time_per_gpu_list) - mean) ** 2))
        data[folder]["mean"]["perun_time_per_gpu"] = mean
        data[folder]["rmse"]["perun_time_per_gpu"] = rmse
        mean = np.mean(np.array(perun_energy_list))
        rmse = np.sqrt(np.mean((np.array(perun_energy_list) - mean) ** 2))
        data[folder]["mean"]["perun_energy"] = mean
        data[folder]["rmse"]["perun_energy"] = rmse
        mean = np.mean(np.array(perun_energy_per_gpu_list))
        rmse = np.sqrt(np.mean((np.array(perun_energy_per_gpu_list) - mean) ** 2))
        data[folder]["mean"]["perun_energy_per_gpu"] = mean
        data[folder]["rmse"]["perun_energy_per_gpu"] = rmse
        mean = np.mean(np.array(slurm_time_list))
        rmse = np.sqrt(np.mean((np.array(slurm_time_list) - mean) ** 2))
        data[folder]["mean"]["slurm_time"] = mean
        data[folder]["rmse"]["slurm_time"] = rmse
        mean = np.mean(np.array(slurm_energy_list))
        rmse = np.sqrt(np.mean((np.array(slurm_energy_list) - mean) ** 2))
        data[folder]["mean"]["slurm_energy"] = mean
        data[folder]["rmse"]["slurm_energy"] = rmse
        mean = np.mean(np.array(top1_error_valid_list))
        rmse = np.sqrt(np.mean((np.array(top1_error_valid_list) - mean) ** 2))
        data[folder]["mean"]["top1_error_valid"] = mean
        data[folder]["rmse"]["top1_error_valid"] = rmse
        mean = np.mean(np.array(top5_error_valid_list))
        rmse = np.sqrt(np.mean((np.array(top5_error_valid_list) - mean) ** 2))
        data[folder]["mean"]["top5_error_valid"] = mean
        data[folder]["rmse"]["top5_error_valid"] = rmse
        mean = np.mean(np.array(top1_error_train_list))
        rmse = np.sqrt(np.mean((np.array(top1_error_train_list) - mean) ** 2))
        data[folder]["mean"]["top1_error_train"] = mean
        data[folder]["rmse"]["top1_error_train"] = rmse
        mean = np.mean(np.array(top5_error_train_list))
        rmse = np.sqrt(np.mean((np.array(top5_error_train_list) - mean) ** 2))
        data[folder]["mean"]["top5_error_train"] = mean
        data[folder]["rmse"]["top5_error_train"] = rmse
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

    data = {}
    for folder in scaling_list:
        data[folder] = {}
        gpus = int(folder.split("g")[0])
        lbs = int((folder.split("b")[0]).split("g")[-1])
        gbs = lbs * gpus
        data[folder]["gpus"] = gpus
        data[folder]["lbs"] = lbs
        data[folder]["gbs"] = gbs

        for slurm_id in scaling_list[folder]:
            data[folder][slurm_id] = {}

            # Get perun data
            perun_h5_file = Path(result_path, folder, slurm_id, "perun", "perun.hdf5")
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
            perun_energy = gpu_total + cpu_total + ram_total
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
                        slurm_time_h = slurm_time_string.split(":")[0]
                        slurm_time_min = slurm_time_string.split(":")[1]
                        slurm_time_s = slurm_time_string.split(":")[2]
                        slurm_time = float(slurm_time_h)*60 + float(slurm_time_min) + float(slurm_time_s)/60

            data[folder][slurm_id]["perun_time"] = perun_time
            data[folder][slurm_id]["perun_time_per_gpu"] = data[folder][slurm_id]["perun_time"] / gpus
            data[folder][slurm_id]["perun_energy"] = perun_energy/3.6  # MJ in kWh
            data[folder][slurm_id]["perun_energy_per_gpu"] = data[folder][slurm_id]["perun_energy"] / gpus
            data[folder][slurm_id]["top1_error_valid"] = 100 - top1_valid_acc[-1]
            data[folder][slurm_id]["top5_error_valid"] = 100 - top5_valid_acc[-1]
            data[folder][slurm_id]["top1_error_train"] = 100 - top1_train_acc[-1]
            data[folder][slurm_id]["top5_error_train"] = 100 - top5_train_acc[-1]
            data[folder][slurm_id]["slurm_energy"] = slurm_energy/1000  # Wh in KWh
            data[folder][slurm_id]["slurm_time"] = slurm_time

    # Statistics
    data = make_statistics(data, scaling_list)
    # Plot it
    plot_scaling(result_path, data, scaling_list, name)
    plot_scaling_per_gpu(result_path, data, scaling_list, name)
    # make table
    make_table(result_path, data, scaling_list, name)
