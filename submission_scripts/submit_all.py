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

experiments = {"256g256l": dict([("RUNTIME", "02:00:00"), ("NNODES", "64"), ("LBS", "256")]),  # 0.5 h
               "256g32l": dict([("RUNTIME", "02:00:00"), ("NNODES", "64"), ("LBS", "32")]),  # 0.5 h
               "128g256l": dict([("RUNTIME", "04:00:00"), ("NNODES", "32"), ("LBS", "256")]),  # 2 h
               "128g64l": dict([("RUNTIME", "04:00:00"), ("NNODES", "32"), ("LBS", "64")]),  # 2 h
               "64g256l": dict([("RUNTIME", "04:00:00"), ("NNODES", "16"), ("LBS", "256")]),  # 2 h
               "64g128l": dict([("RUNTIME", "04:00:00"), ("NNODES", "16"), ("LBS", "128")]),  # 2 h
               "32g256l": dict([("RUNTIME", "05:00:00"), ("NNODES", "8"), ("LBS", "256")]),  # 3 h
               "16g256l": dict([("RUNTIME", "08:00:00"), ("NNODES", "4"), ("LBS", "256")])}  # 6 h

for key in experiments:
    runtime = experiments[key]["RUNTIME"]
    nnodes = experiments[key]["NNODES"]
    lbs = experiments[key]["LBS"]

    env = os.environ.copy()
    env["LBS"] = lbs
    subprocess.run(f"sbatch -N {nnodes} -t {runtime} {path_to_script}", shell=True, env=env)
