from pathlib import Path
import h5py
import torch

from resnet.eval_utils.eval_utils import get_total_energy
from resnet.eval_utils.plot_utils import plot_loss
from resnet.eval_utils.read_utils import print_attrs


result_path = Path("/home/scc/xy6660/ResNet/ResNet/experiments/")
result_path = Path("/Users/philipphuber/Documents/Projects/ResNet/ResNet/experiments/")
job = Path("job_3025698")
perun_h5_file = Path(result_path, job, "perun", "perun.hdf5")

h5val = h5py.File(perun_h5_file, 'r')
#h5val.visititems(print_attrs)
get_total_energy(h5val)

path = Path(result_path, job, "valid_loss.pt")
loss = torch.load(path)
path = Path(result_path, job, "train_acc.pt")
train_acc = torch.load(path)
path = Path(result_path, job, "valid_acc.pt")
valid_acc = torch.load(path)
path = Path(result_path, job, "time.pt")
time = torch.load(path)

plot_loss(loss, time)