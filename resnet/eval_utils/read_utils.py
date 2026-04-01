import ast

import numpy as np
import scipy as sp
import h5py


def print_attrs(name, obj):
    """
    Shows the hdf5 file structure
    """
    print(name)
    for key, val in obj.attrs.items():
        print(f"  - Attribute: {key}: {val}")


def get_h5_paths(h5val) -> [list, dict]:
    """
    Builds hdf5 paths for each node.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.

    Returns
    _______
    h5_paths : list
        List with Paths.
    Nodes : Dict
        Dictionary with nodes and global rank values.
    """
    h5id, _ = next(iter(h5val["perun/nodes"].items()))
    h5_base_path = "perun/nodes/" + h5id + "/nodes/0/nodes"
    h5_paths = []
    nodes = {}
    # get internal hdf5 paths to data for each node
    for node_id, node_obj in h5val[h5_base_path].items():
        h5_paths.append("perun/nodes/" + h5id + "/nodes/0/nodes/" + node_id + "/nodes")
        ranks = h5val["perun/nodes/" + h5id + "/nodes/0/nodes/" + node_id].attrs.get("mpi_ranks")
        ranks = ast.literal_eval(ranks)
        nodes[node_id] = ranks  # not necessarily true - be careful and doublecheck
    return h5_paths, nodes


def get_cores(h5val, h5_path: str = None, key: str = None) -> list:
    """
    Get core numbers.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
    key : str
        gpu, cpu, or ram

    Returns
    _______
    cores : list
        Integer list corresponding to node numbers.
    """
    cores = []
    for name, _ in h5val[h5_path].items():
        num = 0
        if key == "gpu":
            num = name.split(":")[1]
            num = int(num.split("_")[0])
        elif key == "cpu" and "package" in name:
            num = int(name.split("_")[1])
        elif key == "ram" and "dram" in name:
            num = int(name.split("_")[1])
        if num not in cores:
            cores.append(num)
    cores.sort()
    return cores


def get_keys(h5val, h5_paths) -> [list, list]:
    """
    Builds hdf5 paths for each node.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
        _______
    h5_paths : list
        List with Paths.

    Returns
    _______
    keys : list
        Devises monitored by perun.
    """
    keys = ["gpu", "cpu", "ram"]
    for h5_path in h5_paths:
        node_keys = [name for name, obj in h5val[h5_path].items() if isinstance(obj, h5py.Group)]
        if "gpu" not in node_keys:
            keys.remove("gpu")
        if "gpu" not in node_keys:
            keys.remove("cpu")
        if "gpu" not in node_keys:
            keys.remove("ram")
    return keys


def check_regions(h5val) -> [list]:
    """
    Get regions if present.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.

    Returns
    _______
    region_paths : list
        List with Paths.
    """
    h5id, _ = next(iter(h5val["perun/nodes"].items()))
    region_base_path = "perun/nodes/" + h5id + "/nodes/0/regions"
    region_paths = None
    if region_base_path in h5val:
        regions = [name for name, obj in h5val[region_base_path].items() if isinstance(obj, h5py.Group)]
        if regions:
            region_paths = []
            for region in regions:
                region_paths.append(region_base_path + "/" + region)
    return region_paths


def get_region_data(h5val, regions_paths: list = None, nodes: dict = None):
    """
    Get region data.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
    regions_paths : list
        Paths to regions
    Nodes : Dict
        Dictionary with nodes and global rank values.

    Returns
    _______
    region_data : Dict
        Raw data (timestamps) for regions
    """

    perun_region_data = {}

    for node in nodes:
        perun_region_data[node] = {}
        shift = min(nodes[node])
        for global_rank in nodes[node]:
            local_rank = global_rank - shift
            perun_region_data[node][local_rank] = {}
            for region_path in regions_paths:
                region = region_path.split("/")[-1]
                perun_region_data[node][local_rank][region] = {}
                timestamps = np.array(h5val[region_path + "/raw_data/" + str(global_rank)])
                perun_region_data[node][local_rank][region]["timestamps"] = timestamps
    return perun_region_data


def map_regions_to_power(perun_region_data: dict = None, perun_sensor_data: dict = None, keys: list = None):
    """
    Map region data to power profile and calcuclate region energy.

    Parameters
    __________
    perun_region_data : Dict
        Raw data for regions
    perun_sensor_data : Dict
        Perun data
    keys : str
        gpu, cpu, or ram

    Returns
    _______
    perun_region_data : Dict
        Raw data for regions
    """
    for node in perun_region_data:
        for rank in perun_region_data[node]:
            for key in keys:
                if key != "gpu":
                    continue
                power = perun_sensor_data[node][key][rank]["power"]
                timesteps = perun_sensor_data[node][key][rank]["timesteps"]
                for region in perun_region_data[node][rank]:
                    timestamps = perun_region_data[node][rank][region]["timestamps"]
                    perun_region_data[node][rank][region]["avg_power"] = []
                    perun_region_data[node][rank][region]["duration"] = []
                    perun_region_data[node][rank][region]["energy"] = []
                    for i in range(timestamps.shape[0] // 2):
                        if len(power) > 2:
                            start = timestamps[i * 2]
                            end = timestamps[i * 2 + 1]
                            t_inter = np.concatenate([[start], timesteps[np.all([timesteps >= start, timesteps <= end], axis=0)], [end]])
                            avg_p = np.mean(np.interp(t_inter, timesteps, power))
                            duration = end - start
                            perun_region_data[node][rank][region]["avg_power"] = avg_p
                            perun_region_data[node][rank][region]["duration"] = duration
                            perun_region_data[node][rank][region]["energy"] = avg_p * duration
    return perun_region_data


def get_utilization(
        h5val=None, h5_base_path: str = None, num: int = None, key: str = None
) -> [np.array, np.array]:
    """
    Get utilization data from corresponding hdf5 file provided by perun.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
    h5_base_path: str
        Internal path within hdf5 file.
    num : int
        Index of corresponding core
    key : str
        gou, cpu, or ram

    Returns
    _______
    data : dict
        Contains utilization data saved as np.arrays.
    """
    if key == "gpu":
        h5_val_path = f"{h5_base_path}CUDA:{num}_CLOCK_MEM/raw_data/values"
        h5_time_path = f"{h5_base_path}CUDA:{num}_CLOCK_MEM/raw_data/timesteps"
    if key == "cpu":
        h5_val_path = f"{h5_base_path}CPU_USAGE/raw_data/values"
        h5_time_path = f"{h5_base_path}CPU_USAGE/raw_data//timesteps"
    if key == "ram":
        h5_val_path = f"{h5_base_path}RAM_USAGE/raw_data/values"
        h5_time_path = f"{h5_base_path}RAM_USAGE/raw_data/timesteps"
    vals = np.array(h5val[h5_val_path])
    mag = float(h5val[h5_val_path].attrs["mag"])
    util = vals * mag
    timesteps = np.array(h5val[h5_time_path])
    return util, timesteps


def get_cpu_util(
        h5val=None, h5_base_path: str = None, num: int = None, key: str = None
) -> [np.array, np.array]:
    """
    Get utilization data from corresponding hdf5 file provided by perun.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
    h5_base_path: str
        Internal path within hdf5 file.
    num : int
        Index of corresponding core
    key : str
        gou, cpu, or ram

    Returns
    _______
    data : dict
        Contains utilization data saved as np.arrays.
    """
    h5_val_path = f"{h5_base_path}/CPU_UTIL"
    vals = float(h5val[h5_val_path].attrs["value"])
    mag = float(h5val[h5_val_path].attrs["mag"])
    cpu_util = vals * mag
    return cpu_util


def get_gpu_freq(
        h5val=None, h5_base_path: str = None, num: int = None, key: str = None
) -> [np.array, np.array]:
    """
    Get gpu frequencies from corresponding hdf5 file provided by perun.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
    h5_base_path: str
        Internal path within hdf5 file.
    num : int
        Index of corresponding core
    key : str
        gou, cpu, or ram

    Returns
    _______
    data : dict
        Contains utilization data saved as np.arrays.
    """
    if key == "gpu":
        # h5_val_path = f"{h5_base_path}CUDA:{num}_CLOCK_SM/raw_data/values"
        # h5_time_path = f"{h5_base_path}CUDA:{num}_CLOCK_SM/raw_data/timesteps"
        h5_val_path = f"{h5_base_path}CUDA:{num}_CLOCK_GRAPHICS/raw_data/values"
        h5_time_path = f"{h5_base_path}CUDA:{num}_CLOCK_GRAPHICS/raw_data/timesteps"
    vals = np.array(h5val[h5_val_path])
    mag = float(h5val[h5_val_path].attrs["mag"])
    util = vals * mag
    timesteps = np.array(h5val[h5_time_path])
    return util, timesteps


def get_gpu_sm(
        h5val=None, h5_base_path: str = None, num: int = None, key: str = None
) -> [np.array, np.array]:
    """
    Get gpu sm from corresponding hdf5 file provided by perun.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
    h5_base_path: str
        Internal path within hdf5 file.
    num : int
        Index of corresponding core
    key : str
        gou, cpu, or ram

    Returns
    _______
    data : dict
        Contains utilization data saved as np.arrays.
    """
    if key == "gpu":
        h5_val_path = f"{h5_base_path}CUDA:{num}_CLOCK_SM/raw_data/values"
        h5_time_path = f"{h5_base_path}CUDA:{num}_CLOCK_SM/raw_data/timesteps"
    vals = np.array(h5val[h5_val_path])
    mag = float(h5val[h5_val_path].attrs["mag"])
    sm = vals * mag
    timesteps = np.array(h5val[h5_time_path])
    return sm, timesteps


def get_power(
        h5val=None, h5_base_path: str = None, num: int = None, key: str = None
) -> [np.array, np.array]:
    """
    Get gpu power data from corresponding hdf5 file provided by perun.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
    h5_base_path: str
        Internal path within hdf5 file.
    num : int
        Index of corresponding core
    key : str
        gou, cpu, or ram

    Returns
    _______
    data : dict
        Contains the gpu power data saved as np.arrays.
    """
    power = np.array([])
    timesteps = np.array([])
    if key == "gpu":
        h5_power_path = f"{h5_base_path}CUDA:{num}_POWER/raw_data/values"
        h5_time_path = f"{h5_base_path}CUDA:{num}_POWER/raw_data/timesteps"
    if key == "ram":
        h5_power_path = f"{h5_base_path}ram_{num}_dram/raw_data/values"
        h5_time_path = f"{h5_base_path}ram_{num}_dram/raw_data/timesteps"
    if key == "cpu":
        h5_power_path = f"{h5_base_path}cpu_{num}_package-{num}/raw_data/values"
        h5_time_path = f"{h5_base_path}cpu_{num}_package-{num}/raw_data/timesteps"
    if h5_power_path in h5val:
        power = np.array(h5val[h5_power_path])
        mag = float(h5val[h5_power_path].attrs["mag"])
        power = power * mag
        timesteps = np.array(h5val[h5_time_path])
    return power, timesteps


def get_gpu_mem(
        h5val=None, h5_gpu_base_path: str = None, num: int = None
) -> [np.array, np.array]:
    """
    Get gpu memory data from corresponding hdf5 file provided by perun.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
    h5_base_path: str
        Internal path within hdf5 file.
    num : int
        Index of corresponding core

    Returns
    _______
    mem : dict
        Contains the gpu memory data saved as np.arrays.
    timesteps : dict
        Contains timesteps of the gpu memory data saved as np.arrays.
    """
    h5_gpu_mem_path = f"{h5_gpu_base_path}CUDA:{num}_MEM/raw_data/values"
    mem = np.array(h5val[h5_gpu_mem_path])
    mag = h5val[h5_gpu_mem_path].attrs["mag"]
    mem = mem * mag
    h5_gpu_time_path = f"{h5_gpu_base_path}CUDA:{num}_MEM/raw_data/timesteps"
    timesteps = np.array(h5val[h5_gpu_time_path])
    return mem, timesteps


def get_specific_data(h5val=None, h5_base_path: str = None, key: str = None) -> dict:
    """
    Get ram, cpu, or gpu energy/power data from corresponding hdf5 file provided by perun.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
    h5_base_path: str
        Internal path within hdf5 file.
    key : str
        gpu, ram, or cpu

    Returns
    _______
    data : dict
        Contains the cpu, ram, or gpu data saved as np.arrays.
    """
    data = {}
    h5_path = f"{h5_base_path}/{key}/nodes/"
    h5_metric_path = "/".join(h5_base_path.split("/")[0:-1] + ["metrics"])
    cores = get_cores(h5val, h5_path, key=key)
    for num in cores:
        data[num] = {}  # Collects data for each core.
        power, timesteps = get_power(h5val, h5_path, num, key)
        if key == "gpu":
            mem, _ = get_gpu_mem(h5val, h5_path, num)
            data[num]['memory'] = mem / 1024 ** 3  # B to GB
            freq, _ = get_gpu_freq(h5val, h5_path, num, key)
            data[num]["freq"] = freq * 10 ** 6  # Hz to MHz
            sm, _ = get_gpu_sm(h5val, h5_path, num, key)
            data[num]["sm"] = sm / 10 ** 6
        if key == "cpu":
            data[num]["cpu_util"] = get_cpu_util(h5val, h5_metric_path, num, key)
        data[num]["util"], _ = get_utilization(h5val, h5_path, num, key)
        data[num]["power"] = power
        if power.shape[0] > 2:
            data[num]["energy"] = sp.integrate.cumulative_trapezoid(power, x=timesteps)
            data[num]["timesteps"] = timesteps
        else:
            data[num]["energy"] = [0]
            data[num]["timesteps"] = [0]
    return data


def get_perun_data(h5val: h5py = None, name: str = None) -> dict:
    """
    Get all energy and power data from corresponding hdf5 file provided by perun.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.

    Returns
    _______
    perun_data : dict
        Contains the perun data saved as np.arrays.
    """

    perun_sensor_data = {}
    perun_region_data = {}
    h5_base_paths, nodes = get_h5_paths(h5val)
    keys = get_keys(h5val, h5_base_paths)
    regions_paths = check_regions(h5val)
    perun_region_data = {}
    for i, h5_base_path in enumerate(h5_base_paths):
        node = h5_base_path.split("/")[-2]
        perun_sensor_data[node] = {}
        for key in keys:
            perun_sensor_data[node][key] = get_specific_data(h5val, h5_base_path, key)
    if regions_paths is not None:
        perun_region_data = get_region_data(h5val, regions_paths, nodes)
        map_regions_to_power(perun_region_data, perun_sensor_data, keys)
    return perun_sensor_data, perun_region_data


def get_fcn_inference_data(h5val: h5py = None) -> dict:
    """
    Get inference data from fcn.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.

    Returns
    _______
    inference_data : dict
        Contains the inference_data saved as np.arrays.
    """
    inference_data = {}
    vals = np.array(h5val["rmse"])
    inference_data["z500"] = vals[:, :, 14]
    return inference_data


def get_timings(h5val: h5py = None, data: dict = None, folder: str = None):
    """
    Evaluates the compute times of the training.

    Parameters
    ----------
    h5val : h5py
        Value vor hdf5 file with results
    data : dict
        Data.
    folder : str
        Experiment label.
    """
    gpus = data[folder]["gpus"]

    batch_keys = ["batch_time_data_to_device", "batch_time_forward", "batch_time_forward_twostep",
                  "batch_time_loss", "batch_time_loss_twostep", "batch_time_backward",
                  "batch_time_logging", "batch_time_dataloading", "batch_time_single_batch_total",
                  "batch_time_allreduce", "batch_time_total", "batch_time_init", "batch_time_grad_zero"]

    data[folder]["timings"] = {}
    idx = 0
    for n in range(gpus):
        idx = idx + len(np.array(h5val[f"{n}/batch_times_e{1}/batch_time_single_batch_total"]))
    data[folder]["batch_iterations"] = int(idx / gpus)

    for epoch in range(2):
        data[folder]["timings"][epoch] = {}
        data[folder]["timings"][epoch]["batch"] = {}
        for key in batch_keys:
            timings = 0
            for n in range(gpus):
                timings = timings + np.mean(np.array(h5val[f"{n}/batch_times_e{epoch + 1}/{key}"]))
            data[folder]["timings"][epoch]["batch"][key] = timings / gpus
    return data
