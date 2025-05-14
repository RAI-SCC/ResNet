from pathlib import Path
import glob
import h5py
import numpy as np
import torch
import os
import datetime

from resnet.eval_utils.plot_utils import plot_scaling, plot_top1, plot_scaling_time, plot_scaling_top1
from resnet.eval_utils.eval_utils import get_total_energy
from resnet.eval_utils.read_utils import get_perun_data
# from resnet.eval_utils.read_utils import print_attrs


# result_path = Path("/home/scc/xy6660/ResNet/ResNet/experiments/")
result_path = Path("/hkfs/work/workspace/scratch/vm6493-resnet_imagenet/ResNet/experiments")

def hms_to_seconds(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s

def extract_time_from_slurm(filepath):
    """
    Extract the time from a given slurm file.
    Args:
        filepath (str): Path to the slurm file.
    Returns:
        float: Time in hours if found, otherwise None.
    """
    with open(filepath, 'r') as file:
        for line in file:
            if "Job Wall-clock time" in line:
                try:
                    time_str = line.strip().split(": ", 1)[-1].strip()
                    wall_time_seconds = hms_to_seconds(time_str)
                    job_id = os.path.basename(os.path.dirname(filepath))
                    return wall_time_seconds, job_id
                except (IndexError, ValueError):
                    continue  # Skip if the format is unexpected
    return None  # If not found

def get_time(perun_h5_path):
    """
    Get the total run time from the perun hdf5 file.

    Args:
        perun_h5_path (_type_): Path to perun hdf5 file.

    Returns:
        float: time in hours
    """
    h5val = h5py.File(perun_h5_path, 'r')
    perun_data = get_perun_data(h5val)
    time_list = []
    
    for key, _ in perun_data.items():
        for num, _ in perun_data[key]["gpu"].items():
            time_list.append(perun_data[key]["gpu"][num]["timesteps"][-1])
        for num, _ in perun_data[key]["cpu"].items():
            time_list.append(perun_data[key]["cpu"][num]["timesteps"][-1])
        for num, _ in perun_data[key]["ram"].items():
            time_list.append(perun_data[key]["ram"][num]["timesteps"][-1])
    total_time = (max(time_list)) / 3600
    return total_time

def extract_watthours_from_file(filepath):
    """
    Extract the Watthours value from a given file.
    Args:
        filepath (str): Path to the file.
    Returns:
        float: Watthours value if found, otherwise None.
    """
    with open(filepath, 'r') as file:
        for line in file:
            if "Watthours" in line:
                try:
                    watthours = float(line.split("/")[-1].split()[0])
                    return watthours
                except (IndexError, ValueError):
                    continue  # Skip if the format is unexpected
    return None  # If not found

def find_files_with_date(path, threshold= datetime.datetime(2025, 4, 1)):
    """
    Walk through all subdirectories and find 'slurm' files with a start time after the threshold.
    Ignores directories containing 'deprecated' in their path.
    """
    files_with_date = []
    for root, dirs, files in os.walk(path):
        # Filter out deprecated directories in-place
        # dirs[:] = [d for d in dirs if 'depricated' not in d.lower()]

        for filename in files:
            if not filename.startswith("slurm"):
                continue

            filepath = os.path.join(root, filename)

            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        if line.startswith("Starttime:"):
                            time_str = line[len("Starttime:"):].strip()
                            try:
                                file_time = datetime.datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
                                if file_time > threshold:
                                    files_with_date.append(filepath)
                                break
                            except ValueError:
                                print(f"Could not parse date in {filepath}: {time_str}")
                                break
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    return files_with_date


def get_fastest_experiment_runs(gpu_nums, batch_sizes):
    """
    Get the latest run for each hyperparameter combination.
    Args:
        gpu_nums (list): List of GPU numbers.
        batch_sizes (list): List of batch sizes.  
    """
    full_run_list = []
    run_list = []
    energies = []

    for i in range(len(gpu_nums)):
        hyperparams = f"{gpu_nums[i]}g{batch_sizes[i]}b4w100e"
        path = f"{result_path}/{hyperparams}"
        perun_h5_path = f"{path}/perun/perun.hdf5"
        matched = glob.glob(f"{path}/[0-9]*")
        # list all slurm ids with the same hyperparams
        latest_list = []  
        for m in matched:
            # [-1] for the slurm id
            latest_list.append(int(m.split("/")[-1]))
        # getting the latest run with the highest slurm id
        # latest_list.sort()
        # print(f"gpu:{gpu_nums[i]} bs:{batch_sizes[i]} slurms:{latest_list}")
        # latest = latest_list[0]
        # fastest_slurm = float('inf') # use this to get the fastest run
        fastest_time = float('inf')
        for job_id in latest_list:
            slurm = f"{path}/{job_id}/slurm_{job_id}"
            wall_time, job_id = extract_time_from_slurm(slurm)
            if wall_time < fastest_time:
                fastest_slurm = job_id
                fastest_time = wall_time
                
        # fastest_slurm = latest_list[-1] # use this to get the latest run

        # print(f"gpu:{gpu_nums[i]} lbs:{batch_sizes[i]} kWh:{kWh}")
        scaling_name = f"{hyperparams}/{fastest_slurm}"
        wh = extract_watthours_from_file(f"{path}/{fastest_slurm}/slurm_{fastest_slurm}")
        kWh = float(wh) / 1000 
        energies.append(kWh)
        run_list.append(scaling_name)
        full_run_list.append(f"{path}/{fastest_slurm}")
        # print(f"gpu:{gpu_nums[i]} bs:{batch_sizes[i]} slurm:{fastest_slurm} wall_time:{fastest_time/3600:.2f} h")

    
    return run_list, energies, full_run_list

def get_fastest_by_date(gpu_nums, batch_sizes,date_threshold=datetime.datetime(2025, 5, 1)):
    """
    Get the latest file in a directory.
    Args:
        path (str): Path to the directory.
    Returns:
        str: Path to the latest file.
    """
    full_run_list = []
    run_list = []
    energies = []

    for i in range(len(gpu_nums)):
        hyperparams = f"{gpu_nums[i]}g{batch_sizes[i]}b4w100e"
        path = f"{result_path}/{hyperparams}"
        files = find_files_with_date(path, date_threshold)
        print(files)
        fastest_time = float('inf')
        for slurm in files:
            wall_time, job_id = extract_time_from_slurm(slurm)
            if wall_time < fastest_time:
                fastest_slurm = job_id
                fastest_time = wall_time
                
        # fastest_slurm = latest_list[-1] # use this to get the latest run

        # print(f"gpu:{gpu_nums[i]} lbs:{batch_sizes[i]} kWh:{kWh}")
        scaling_name = f"{hyperparams}/{fastest_slurm}"
        wh = extract_watthours_from_file(f"{path}/{fastest_slurm}/slurm_{fastest_slurm}")
        kWh = float(wh) / 1000 
        energies.append(kWh)
        run_list.append(scaling_name)
        full_run_list.append(f"{path}/{fastest_slurm}")
    
    return run_list, energies, full_run_list

def get_times(hyperparams_slurm):
    """
    Get the total run time from the perun hdf5 file.
    Args:
        hyperparams (str): Hyperparameters for the experiment.
    Returns:
        float: time in hours
    """
    times = []
    for i in range(len(hyperparams_slurm)):
        perun_h5_path = f"{result_path}/{hyperparams_slurm[i]}/perun/perun.hdf5"
        time = get_time(perun_h5_path)
        times.append(time)
    return times

def get_average_power(perun_h5_path):
    """
    Calculate the average power consumption.
    Args:
        component_energy (float): Energy consumed by the component in kWh.
        time (float): Time taken in hours.
    Returns:
        float: Average power consumption in watts.
    """    
    h5val = h5py.File(perun_h5_path, 'r')
    perun_data = get_perun_data(h5val)
    gpu = []
    cpu = []
    ram = []
    for key, _ in perun_data.items():
        for num, _ in perun_data[key]["gpu"].items():
            gpu_energy = (perun_data[key]["gpu"][num]["energy"][-1])
            gpu_time = perun_data[key]["gpu"][num]["timesteps"][-1]
            gpu.append(gpu_energy / gpu_time)
        for num, _ in perun_data[key]["cpu"].items():
            cpu_energy = (perun_data[key]["cpu"][num]["energy"][-1])
            cpu_time = perun_data[key]["cpu"][num]["timesteps"][-1]
            cpu.append(cpu_energy / cpu_time)
        for num, _ in perun_data[key]["ram"].items():
            ram_energy = (perun_data[key]["ram"][num]["energy"][-1])
            ram_time = perun_data[key]["ram"][num]["timesteps"][-1]
            ram.append(ram_energy / ram_time)
    return np.mean(gpu), np.mean(cpu), np.mean(ram)

########################################################################################################################
# Constant gbs
########################################################################################################################
print("constant gbs 8k")
constant_gbs_gpus = [32, 64, 128]
constant_gbs = [256, 128, 64]
constant_gbs_list, constant_gbs_energies, constant_gbs_full = get_fastest_experiment_runs(constant_gbs_gpus, constant_gbs)
constant_gbs_times = get_times(constant_gbs_list)
constant_gbs_error = []

for i in range(len(constant_gbs_list)):
    gpu, cpu, ram = get_average_power(f"{constant_gbs_full[i]}/perun/perun.hdf5")
    path = Path(constant_gbs_full[i], "valid_top1.pt")
    top1_valid_acc = torch.load(path)
    error = 100 - top1_valid_acc[-1]
    constant_gbs_error.append(error)
    if i == 0:
        # print(f"gpu:{constant_gbs_gpus[i]:.2f} lbs:{constant_gbs[i]:.2f} time:{constant_gbs_times[i]:.2f} up:{1:.2f} gpuh:{constant_gbs_times[i]*constant_gbs_gpus[i]:.2f} kWh:{constant_gbs_energies[i]:.2f} gpu_pow:{gpu:.2f} cpu_pow:{cpu:.2f} ram_pow:{error:.2f}")
        print(f"{constant_gbs_gpus[i]} & {constant_gbs_times[i]:.2f} & {1} & {constant_gbs_times[i]*constant_gbs_gpus[i]:.2f} & {constant_gbs_energies[i]:.2f} & {constant_gbs_energies[i]/constant_gbs_gpus[i]*4:.2f} & {gpu:.2f} & {cpu:.2f} & {error:.2f} \\\\")
    else:
        # print(f"gpu:{constant_gbs_gpus[i]:.2f} lbs:{constant_gbs[i]:.2f} time:{constant_gbs_times[i]:.2f} up:{constant_gbs_times[0]/constant_gbs_times[i]:.2f} gpuh:{constant_gbs_times[i]*constant_gbs_gpus[i]:.2f} kWh:{constant_gbs_energies[i]:.2f} gpu_pow:{gpu:.2f} cpu_pow:{cpu:.2f} ram_pow:{error:.2f}")
        print(f"{constant_gbs_gpus[i]} & {constant_gbs_times[i]:.2f} & {constant_gbs_times[0]/constant_gbs_times[i]:.2f} & {constant_gbs_times[i]*constant_gbs_gpus[i]:.2f} & {constant_gbs_energies[i]:.2f} & {constant_gbs_energies[i]/constant_gbs_gpus[i]*4:.2f} & {gpu:.2f} & {cpu:.2f} & {error:.2f} \\\\")

name = "resnet_gbs"
# plot_scaling(result_path=result_path, scaling_list=constant_gbs_list, name="resnet8192gbs")
plot_scaling_time(result_path=result_path, name=name, gpus=constant_gbs_gpus, times=constant_gbs_times, total_energies=constant_gbs_energies)
plot_scaling_top1(result_path=result_path, name=name, gpus=constant_gbs_gpus, total_energies=constant_gbs_energies, top1=constant_gbs_error)
plot_top1(result_path, constant_gbs_list, name="resnet_gbs", key="gbs")

########################################################################################################################
# Constant lbs
########################################################################################################################
print("constant lbs 256")
name = "resnet_lbs"
constant_lbs_gpus = [16, 32, 64, 128]
constant_lbs = [256, 256, 256, 256]
constant_lbs_list, constant_lbs_energies, constant_lbs_full = get_fastest_experiment_runs(constant_lbs_gpus, constant_lbs)
constant_lbs_times = get_times(constant_lbs_list)
constant_lbs_error = []

for i in range(len(constant_lbs_list)):
    gpu, cpu, ram = get_average_power(f"{constant_lbs_full[i]}/perun/perun.hdf5")
    path = Path(constant_lbs_full[i], "valid_top1.pt")
    top1_valid_acc = torch.load(path)
    error = 100 - top1_valid_acc[-1]
    constant_lbs_error.append(error)
    if i == 0:
        # print(f"gpu:{constant_lbs_gpus[i]:.2f} lbs:{constant_lbs[i]:.2f} time:{constant_lbs_times[i]:.2f} up:{1:.2f} gpuh:{constant_lbs_times[i]*constant_lbs_gpus[i]:.2f} kWh:{constant_lbs_energies[i]:.2f} gpu_pow:{gpu:.2f} cpu_pow:{cpu:.2f} ram_pow:{error:.2f}")
        print(f"{constant_lbs_gpus[i]} & {constant_lbs_times[i]:.2f} & {1} & {constant_lbs_times[i]*constant_lbs_gpus[i]:.2f} & {constant_lbs_energies[i]:.2f} & {constant_lbs_energies[i]/constant_lbs_gpus[i]*4:.2f} & {gpu:.2f} & {cpu:.2f} & {error:.2f} \\\\")
    else:
        # print(f"gpu:{constant_lbs_gpus[i]:.2f} lbs:{constant_lbs[i]:.2f} time:{constant_lbs_times[i]:.2f} up:{constant_lbs_times[0]/constant_lbs_times[i]:.2f} gpuh:{constant_lbs_times[i]*constant_lbs_gpus[i]:.2f} kWh:{constant_lbs_energies[i]:.2f} gpu_pow:{gpu:.2f} cpu_pow:{cpu:.2f} ram_pow:{error:.2f}")
        print(f"{constant_lbs_gpus[i]} & {constant_lbs_times[i]:.2f} & {constant_lbs_times[0]/constant_lbs_times[i]:.2f} & {constant_lbs_times[i]*constant_lbs_gpus[i]:.2f} & {constant_lbs_energies[i]:.2f} & {constant_lbs_energies[i]/constant_lbs_gpus[i]*4:.2f} & {gpu:.2f} & {cpu:.2f} & {error:.2f} \\\\")

plot_scaling_time(result_path=result_path, name="resnet_lbs", gpus=constant_lbs_gpus, times=constant_lbs_times, total_energies=constant_lbs_energies)
plot_scaling_top1(result_path=result_path, name="resnet_lbs", gpus=constant_lbs_gpus, total_energies=constant_lbs_energies, top1=constant_lbs_error)
# plot_scaling(result_path=result_path, scaling_list=constant_lbs_list, name="resnet256lbs")
plot_top1(result_path, constant_lbs_list, name="resnet_lbs", key="lbs")


########################################################################################################################
# Constant gpu
########################################################################################################################
# print("constant gpus 64")
# constant_gpu_list = [64, 64, 64]
# constant_gpu_batches = [32, 64, 128]
# constant_gpus_path, constant_gpu_energies, constant_gpu_full = get_fastest_experiment_runs(constant_gpu_list, constant_gpu_batches)
# constant_gpus_times = get_times(constant_gpus_path)
# constant_gpus_error = []

# for i in range(len(constant_gpu_list)):
#     gpu, cpu, ram = get_average_power(f"{constant_gpu_full[i]}/perun/perun.hdf5")
#     path = Path(constant_gpu_full[i], "valid_top1.pt")
#     top1_valid_acc = torch.load(path)
#     error = 100 - top1_valid_acc[-1]
#     constant_gpus_error.append(error)
#     if i == 0:
#         # print(f"gpu:{constant_gpu_list[i]} lbs:{constant_gpu_batches[i]} time:{constant_gpus_times[i]} up:{1} gpuh:{constant_gpus_times[i]*constant_gpu_list[i]} kWh:{constant_gpu_energies[i]} gpu_pow:{gpu} cpu_pow:{cpu} ram_pow:{ram}")
#         print(f"{constant_gpu_list[i]} & {constant_gpus_times[i]:.2f} & {1} & {constant_gpus_times[i]*constant_gpu_list[i]:.2f} & {constant_gpu_energies[i]:.2f} & {constant_gpu_energies[i]/constant_gpu_list[i]*4:.2f} & {gpu:.2f} & {cpu:.2f} & {error:.2f} \\\\")
#     else:
#         # print(f"gpu:{constant_gpu_list[i]:.2f} lbs:{constant_gpu_batches[i]:.2f} time:{constant_gpus_times[i]:.2f} up:{constant_gpus_times[0]/constant_gpus_times[i]:.2f} gpuh:{constant_gpus_times[i]*constant_gpu_list[i]:.2f} kWh:{constant_gpu_energies[i]:.2f} gpu_pow:{gpu:.2f} cpu_pow:{cpu:.2f} ram_pow:{error:.2f}")
#         print(f"{constant_gpu_list[i]} & {constant_gpus_times[i]:.2f} & {constant_gpus_times[0]/constant_gpus_times[i]:.2f} & {constant_gpus_times[i]*constant_gpu_list[i]:.2f} & {constant_gpu_energies[i]:.2f} & {constant_gpu_energies[i]/constant_gpu_list[i]*4:.2f} & {gpu:.2f} & {cpu:.2f} & {error:.2f} \\\\")

# name = "resnet_gpus"
# plot_scaling_time(result_path=result_path, name=name, gpus=constant_gpu_list, times=constant_gpus_times, total_energies=constant_gpu_energies)
# plot_scaling_top1(result_path=result_path, name=name, gpus=constant_gpu_list, total_energies=constant_gpu_energies, top1=constant_gpus_error)
# # plot_scaling(result_path=result_path, scaling_list=constant_gpus_path, name="resnet64gpus")
# plot_top1(result_path, constant_gpus_path, name="resnet_gpus", key="gpu")