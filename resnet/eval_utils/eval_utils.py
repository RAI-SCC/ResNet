from pathlib import Path
import h5py
import numpy as np

from resnet.eval_utils.read_utils import get_perun_data


def get_total_energy(h5val: h5py = None):
    """
    Calculates the total energy consumed.

    Parameters
    __________
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
