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


def get_h5_paths(h5val) -> [list, list]:
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
    nodes : list
        List with node names.
    """
    h5id, _ = next(iter(h5val["perun/nodes"].items()))
    h5_base_path = "perun/nodes/" + h5id + "/nodes/0/nodes"
    h5_paths = []
    nodes = []
    # get internal hdf5 paths to data for each node
    for node_id, node_obj in h5val[h5_base_path].items():
        h5_paths.append("perun/nodes/" + h5id + "/nodes/0/nodes/" + node_id + "/nodes")
        nodes.append(node_id)
    return h5_paths, nodes


def adjust_energy(energy: np.array = None, max_val: float = None) -> np.array:
    """
    Manipulates energy values from perun hdf5 file to get correct values.

    Parameters
    __________
    energy : np.array
        Energy values to b e adjusted.
    key : str
        cpu or ram
    max_val : float
        Device overflow limit.

    Returns
    _______
    energy_adjusted : np.array
        Adjusted energy values.
    """
    e_start = energy[0]
    energy_adjusted = []
    e_prev = 0
    shift_num = 0
    for val in energy:
        e = val - e_start
        if val + shift_num * max_val < e_prev:
            shift_num = shift_num + 1
        e = e + shift_num * max_val
        e_prev = e
        energy_adjusted.append(e)
    energy_adjusted = np.array(energy_adjusted)
    return np.array(energy_adjusted)


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
    if key == "gpu":
        h5_power_path = f"{h5_base_path}CUDA:{num}_POWER/raw_data/values"
        h5_time_path = f"{h5_base_path}CUDA:{num}_POWER/raw_data/timesteps"
    if key == "ram":
        h5_power_path = f"{h5_base_path}ram_{num}_dram/raw_data/values"
        h5_time_path = f"{h5_base_path}ram_{num}_dram/raw_data/timesteps"
    if key == "cpu":
        h5_power_path = f"{h5_base_path}cpu_{num}_package-{num}/raw_data/values"
        h5_time_path = f"{h5_base_path}cpu_{num}_package-{num}/raw_data/timesteps"
    power = np.array(h5val[h5_power_path])
    mag = float(h5val[h5_power_path].attrs["mag"])
    power = power* mag
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
    mem = mem*mag
    h5_gpu_time_path = f"{h5_gpu_base_path}CUDA:{num}_MEM/raw_data/timesteps"
    timesteps = np.array(h5val[h5_gpu_time_path])
    return mem, timesteps


def get_energy(
    h5val=None, h5_path: str = None, num: int = None, key: str = None
) -> [np.array, np.array]:
    """
    Get ram or cpu energy data from corresponding hdf5 file provided by perun.

    Parameters
    __________
    h5val : HDF5
        Key value to hdf5 file.
    h5_base_path: str
        Internal path within hdf5 file.
    num : int
        Index of corresponding core
    key : str
        gpu or ram

    Returns
    _______
    data : dict
        Contains the cpu or ram data saved as np.arrays.
    """
    energy = np.array([])
    timesteps = np.array([])
    if key == "ram":
        h5_energy_path = f"{h5_path}ram_{num}_dram/raw_data/alt_values"
        h5_time_path = f"{h5_path}ram_{num}_dram/raw_data/timesteps"
    if key == "cpu":
        h5_energy_path = f"{h5_path}cpu_{num}_package-{num}/raw_data/alt_values"
        h5_time_path = f"{h5_path}cpu_{num}_package-{num}/raw_data/timesteps"
    energy = np.array(h5val[h5_energy_path])
    mag = h5val[h5_energy_path].attrs["mag"]
    max_val = h5val[h5_energy_path].attrs["max_val"] * mag
    energy = energy * mag
    energy = adjust_energy(energy, max_val)
    timesteps = np.array(h5val[h5_time_path])
    return energy, timesteps


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
    cores = get_cores(h5val, h5_path, key=key)
    for num in cores:
        data[num] = {}  # Collects data for each core.
        power, timesteps = get_power(h5val, h5_path, num, key)
        if key == "gpu":
            mem, _ = get_gpu_mem(h5val, h5_path, num)
            data[num]['memory'] = mem * 1024 ** 3  # B to GB
        data[num]["power"] = power
        data[num]["energy"] = sp.integrate.cumulative_trapezoid(power, x=timesteps)
        data[num]["timesteps"] = timesteps

    return data


def get_perun_data(h5val: h5py = None) -> dict:
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
    keys = ["gpu", "cpu", "ram"]
    perun_data = {}
    h5_base_paths, nodes = get_h5_paths(h5val)
    for i, h5_base_path in enumerate(h5_base_paths):
        node = nodes[i]  # Collects data for each node.
        perun_data[node] = {}
        for key in keys:
            perun_data[node][key] = get_specific_data(h5val, h5_base_path, key)
    return perun_data
