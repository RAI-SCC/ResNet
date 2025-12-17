from pathlib import Path

import h5py
import numpy as np

def print_attrs(name, obj):
    """
    Shows the hdf5 file structure
    """
    print(name)
    for key, val in obj.attrs.items():
        print(f"  - Attribute: {key}: {val}")


result_path = Path("/Users/philipphuber/Documents/Projects/ResNet/experiments")
perun_h5_file = Path("/Users/philipphuber/Documents/Projects/ResNet/experiments/times.h5")
h5val = h5py.File(perun_h5_file, 'r')
h5val.visititems(print_attrs)

gpus = len(h5val)

batch_keys = ["batch_time_dataloading", "batch_time_data_to_device",
              "batch_time_forward", "batch_time_backward", "batch_time_total"]

valid_keys = ["val_time_dataloading", "val_time_data_to_device", "val_time_forward", "val_time_in-batch_eval",
              "val_time_out-batch_eval", "val_time_total"]

epoch_keys = ["epoch_time_init", "epoch_time_batches", "epoch_time_validate_train", "epoch_time_validate_valid",
              "epoch_time_allreduce", "epoch_time_evaluation", "epoch_time_prints",
              "epoch_time_step", "epoch_time_total"]

timings = {}
total = 0
for key in batch_keys:
    timings[key] = 0
    for n in range(gpus):
        timings[key] = timings[key] + np.mean(np.array(h5val[f"{n}/batch_times_e1/{key}"]))
    timings[key] = timings[key]/(gpus)
    print(key, timings[key]*313)
print(np.array(h5val[f"0/batch_times_e1/batch_time_backward"]).shape)

timings = {}
total = 0
for key in valid_keys:
    timings[key] = 0
    for n in range(gpus):
        timings[key] = timings[key] + sum(np.array(h5val[f"{n}/val_times_valid_e1/{key}"]))
    timings[key] = timings[key]/(gpus)
    print("valid", key, timings[key])

timings = {}
total = 0
for key in valid_keys:
    timings[key] = 0
    for n in range(gpus):
        timings[key] = timings[key] + sum(np.array(h5val[f"{n}/val_times_train_e1/{key}"]))
    timings[key] = timings[key]/(gpus)
    print("train", key, timings[key])

timings = {}
total = 0
for key in epoch_keys:
    timings[key] = 0
    for n in range(gpus):
        timings[key] = timings[key] + sum(np.array(h5val[f"{n}/{key}"]))
    timings[key] = timings[key]/(gpus)
    print(key, timings[key])
