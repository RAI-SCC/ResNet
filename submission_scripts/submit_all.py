import os
from pathlib import Path
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, help='Path to folder with submission scripts')
args = parser.parse_args()

if args.path is None:
    raise ValueError("Please provide a path to submission scripts.")

path_to_script = Path(args.path)

if not path_to_script.exists:
    raise ValueError(f"Path {path_to_script} does not exist.")

# large scaling
experiments = {"256g256l": dict([("RUNTIME", "02:00:00"), ("NNODES", "64"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4")]),  # 0.5 h
               "256g32l": dict([("RUNTIME", "02:00:00"), ("NNODES", "64"), ("LBS", "32"), ("TASKS", "4"), ("GPUs", "4")]),  # 0.5 h
               "128g256l": dict([("RUNTIME", "04:00:00"), ("NNODES", "32"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4")]),  # 2 h
               "128g64l": dict([("RUNTIME", "04:00:00"), ("NNODES", "32"), ("LBS", "64"), ("TASKS", "4"), ("GPUs", "4")]),  # 2 h
               "64g256l": dict([("RUNTIME", "04:00:00"), ("NNODES", "16"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4")]),  # 2 h
               "64g128l": dict([("RUNTIME", "04:00:00"), ("NNODES", "16"), ("LBS", "128"), ("TASKS", "4"), ("GPUs", "4")]),  # 2 h
               "32g256l": dict([("RUNTIME", "05:00:00"), ("NNODES", "8"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4")]),  # 3 h
               "16g256l": dict([("RUNTIME", "08:00:00"), ("NNODES", "4"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4")])}  # 6 h
# small scaling const LBS
experiments = {"1g256l": dict([("RUNTIME", "90:00:00"), ("NNODES", "1"), ("LBS", "256"), ("TASKS", "1"), ("GPUs", "1")]),  # 982 h
               "2g256l": dict([("RUNTIME", "48:00:00"), ("NNODES", "1"), ("LBS", "256"), ("TASKS", "2"), ("GPUs", "2")]),  # 48 h
               "4g256l": dict([("RUNTIME", "24:00:00"), ("NNODES", "1"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4")]),  # 24 h
               "8g256l": dict([("RUNTIME", "12:00:00"), ("NNODES", "2"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4")])}  # 12 h
# small scaling const GBS
experiments = {"1g256l": dict([("RUNTIME", "90:00:00"), ("NNODES", "1"), ("LBS", "256"), ("TASKS", "1"), ("GPUs", "1")]),  # 82 h
               "2g128l": dict([("RUNTIME", "48:00:00"), ("NNODES", "1"), ("LBS", "128"), ("TASKS", "2"), ("GPUs", "2")]),  # 40 h
               "4g64l": dict([("RUNTIME", "26:00:00"), ("NNODES", "1"), ("LBS", "64"), ("TASKS", "4"), ("GPUs", "4")]),  # 22 h
               "8g32l": dict([("RUNTIME", "14:00:00"), ("NNODES", "2"), ("LBS", "32"), ("TASKS", "4"), ("GPUs", "4")]),  # 12 h
               "16g16l": dict([("RUNTIME", "8:00:00"), ("NNODES", "4"), ("LBS", "16"), ("TASKS", "4"), ("GPUs", "4")])}  # 7 h

# timings
experiments = {"256g256l": dict([("RUNTIME", "00:10:00"), ("NNODES", "64"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "256g32l": dict([("RUNTIME", "00:10:00"), ("NNODES", "64"), ("LBS", "32"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "128g256l": dict([("RUNTIME", "00:15:00"), ("NNODES", "32"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "128g64l": dict([("RUNTIME", "00:15:00"), ("NNODES", "32"), ("LBS", "64"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "64g256l": dict([("RUNTIME", "00:15:00"), ("NNODES", "16"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "64g128l": dict([("RUNTIME", "00:15:00"), ("NNODES", "16"), ("LBS", "128"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "32g256l": dict([("RUNTIME", "00:15:00"), ("NNODES", "8"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "16g256l": dict([("RUNTIME", "00:20:00"), ("NNODES", "4"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "1g256l": dict([("RUNTIME", "2:00:00"), ("NNODES", "1"), ("LBS", "256"), ("TASKS", "1"), ("GPUs", "1"), ("EPOCHS", "2")]),
               "2g256l": dict([("RUNTIME", "1:10:00"), ("NNODES", "1"), ("LBS", "256"), ("TASKS", "2"), ("GPUs", "2"), ("EPOCHS", "2")]),
               "4g256l": dict([("RUNTIME", "00:50:00"), ("NNODES", "1"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "8g256l": dict([("RUNTIME", "00:30:00"), ("NNODES", "2"), ("LBS", "256"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "2g128l": dict([("RUNTIME", "1:20:00"), ("NNODES", "1"), ("LBS", "128"), ("TASKS", "2"), ("GPUs", "2"), ("EPOCHS", "2")]),
               "4g64l": dict([("RUNTIME", "00:25:00"), ("NNODES", "1"), ("LBS", "64"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "8g32l": dict([("RUNTIME", "00:20:00"), ("NNODES", "2"), ("LBS", "32"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")]),
               "16g16l": dict([("RUNTIME", "00:15:00"), ("NNODES", "4"), ("LBS", "16"), ("TASKS", "4"), ("GPUs", "4"), ("EPOCHS", "2")])}

epochs = 100
for key in experiments:
    runtime = experiments[key]["RUNTIME"]
    nnodes = experiments[key]["NNODES"]
    lbs = experiments[key]["LBS"]
    tasks_per_node = experiments[key]["TASKS"]
    gpus_per_node = experiments[key]["GPUs"]
    if "EPOCHS" in experiments[key]:
        epochs = experiments[key]["EPOCHS"]

    env = os.environ.copy()
    env["LBS"] = lbs
    env["EPOCHS"] = epochs
    subprocess.run(f"sbatch -N {nnodes} -t {runtime} --ntasks-per-node {tasks_per_node} --gpus-per-node {gpus_per_node} {path_to_script}", shell=True, env=env)
