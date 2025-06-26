from pathlib import Path
import h5py
import numpy as np
import torch

from resnet.eval_utils.read_utils import get_perun_data
from resnet.eval_utils.plot_utils import plot_scaling, plot_scaling_per_gpu, plot_scaling_per_workload, plot_efficiency
from resnet.eval_utils.table_utils import make_table, make_small_table, make_table_with_eff


def get_timings(h5val: h5py = None):
    """
    Evaluates the compute times of the training.

    Parameters
    ----------
    h5val : h5py
        Value vor hdf5 file with results
    """
    epoch_times = {"epoch_time_batches": 0, "epoch_time_allreduce": 0, "epoch_time_total": 0, "epoch_time_validation": 0,
                   "epoch_time_evaluation": 0, "epoch_time_validate_valid": 0, "epoch_time_validate_train": 0, "epoch_time_save_data": 0}

    batch_times = {"e1" : {}, "e2" : {}}
    batch_times["e1"] =  {"batch_time_dataloading": 0, "batch_time_forward": 0,  "batch_time_backward": 0, "batch_time_total": 0}
    batch_times["e2"] = {"batch_time_dataloading": 0, "batch_time_forward": 0,  "batch_time_backward": 0, "batch_time_total": 0}
    validation_valid_times = {"e1": {}, "e2": {}}
    validation_valid_times["e1"] =  {"val_time_dataloading": 0, "val_time_forward": 0, "val_time_eval": 0}
    validation_valid_times["e2"] = {"val_time_dataloading": 0, "val_time_forward": 0, "val_time_eval": 0}
    validation_train_times = {"e1": {}, "e2": {}}
    validation_train_times["e1"] =  {"val_time_dataloading": 0, "val_time_forward": 0, "val_time_eval": 0}
    validation_train_times["e2"] = {"val_time_dataloading": 0, "val_time_forward": 0, "val_time_eval": 0}

    num_gpus = 0
    for rank_id, rank_obj in h5val.items():
        num_gpus = max(int(rank_id)+1, num_gpus)
        for key in epoch_times:
            epoch_times[key] = np.add(epoch_times[key], np.array(h5val[rank_id][key]))

        for key in validation_valid_times["e1"]:
            validation_valid_times["e1"][key] = np.add(validation_valid_times["e1"][key], np.array(h5val[rank_id]["val_times_valid_e1"][key]))

        for key in validation_valid_times["e2"]:
            validation_valid_times["e2"][key] = np.add(validation_valid_times["e2"][key], np.array(h5val[rank_id]["val_times_valid_e2"][key]))

        for key in validation_train_times["e1"]:
            validation_train_times["e1"][key] = np.add(validation_train_times["e1"][key], np.array(h5val[rank_id]["val_times_train_e1"][key]))

        for key in validation_train_times["e2"]:
            validation_train_times["e2"][key] = np.add(validation_train_times["e2"][key], np.array(h5val[rank_id]["val_times_train_e2"][key]))

        for key in batch_times["e1"]:
            batch_times["e1"][key] = np.add(batch_times["e1"][key], np.array(h5val[rank_id]["batch_times_e1"][key]))

        for key in batch_times["e2"]:
            batch_times["e2"][key] = np.add(batch_times["e2"][key], np.array(h5val[rank_id]["batch_times_e2"][key]))

    for key in epoch_times:
        epoch_times[key] /= num_gpus
        print(key, sum(epoch_times[key]) / len(epoch_times[key]))
    for key in batch_times["e1"]:
        batch_times["e1"][key] /= num_gpus
        print("e1", key, sum(batch_times["e1"][key]))
    for key in batch_times["e2"]:
        batch_times["e2"][key] /= num_gpus
        print("e2", key, sum(batch_times["e2"][key]))
    for key in validation_train_times["e1"]:
        validation_train_times["e1"][key] /= num_gpus
        print("e1", key, sum(validation_train_times["e1"][key]))
    for key in validation_train_times["e2"]:
        validation_train_times["e2"][key] /= num_gpus
        print("e2", key, sum(validation_train_times["e2"][key]))
    for key in validation_valid_times["e1"]:
        validation_valid_times["e1"][key] /= num_gpus
        print("e1", key, sum(validation_valid_times["e1"][key]))
    for key in validation_valid_times["e2"]:
        validation_valid_times["e2"][key] /= num_gpus
        print("e2", key, sum(validation_valid_times["e2"][key]))


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

        for slurm_id in scaling_list[folder]:
            data[folder][slurm_id] = {}
            print(result_path, folder, slurm_id)
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
                        slurm_time_d = 0
                        if "-" in slurm_time_string:
                            slurm_time_d = slurm_time_string.split("-")[0]
                            slurm_time_string = slurm_time_string.split("-")[1]
                        slurm_time_h = slurm_time_string.split(":")[0]
                        slurm_time_min = slurm_time_string.split(":")[1]
                        slurm_time_s = slurm_time_string.split(":")[2]
                        slurm_time = float(slurm_time_d)*60*24 + float(slurm_time_h)*60 + float(slurm_time_min) + float(slurm_time_s)/60

            # Get slurm data
            if "s" in folder:
                timing_folder = str(folder).split("4w")[0] + "4w2e"
            else:
                timing_folder = str(folder).split("4w")[0] + "4w2e"
            timing_path = Path(result_path, timing_folder, "times.h5")
            #print(timing_path)

            #if not timing_path.exists:
            #    eval_timings = False
            #    print(f"Timings for {folder} are missing - skip evaluation.")
            #else:
            #    h5val = h5py.File(timing_path, 'r')
            #    get_timings(h5val)

            data[folder][slurm_id]["perun_time"] = perun_time
            data[folder][slurm_id]["perun_time_per_node"] = data[folder][slurm_id]["perun_time"] / nodes
            data[folder][slurm_id]["perun_time_per_workload"] = data[folder][slurm_id]["perun_time"] * gpus / n_images_train
            data[folder][slurm_id]["perun_gpu_h"] = (data[folder][slurm_id]["perun_time"] * gpus)/60
            data[folder][slurm_id]["perun_energy"] = perun_energy/3.6  # MJ in kWh
            data[folder][slurm_id]["perun_energy_per_node"] = data[folder][slurm_id]["perun_energy"] / nodes
            data[folder][slurm_id]["perun_energy_per_workload"] = data[folder][slurm_id]["perun_energy"] * gpus / n_images_train
            data[folder][slurm_id]["top1_error_valid"] = 100 - top1_valid_acc[-1]
            data[folder][slurm_id]["top5_error_valid"] = 100 - top5_valid_acc[-1]
            data[folder][slurm_id]["top1_error_train"] = 100 - top1_train_acc[-1]
            data[folder][slurm_id]["top5_error_train"] = 100 - top5_train_acc[-1]
            data[folder][slurm_id]["slurm_energy"] = slurm_energy/1000  # Wh in KWh
            data[folder][slurm_id]["slurm_time"] = slurm_time

    eval_list = []
    for key in data[folder][slurm_id]:
        eval_list.append(key)

    # Statistics
    data = make_statistics(data, scaling_list, eval_list)

    # efficiency
    data = calc_efficiency(data, scaling_list, eval_list)

    # Plot it
    #plot_scaling(result_path, data, scaling_list, name)
    #plot_scaling_per_gpu(result_path, data, scaling_list, name)
    #plot_scaling_per_workload(result_path, data, scaling_list, name)
    plot_efficiency(result_path, data, scaling_list, name)

    # make table
    #make_table(result_path, data, scaling_list, name)
    make_table_with_eff(result_path, data, name)
