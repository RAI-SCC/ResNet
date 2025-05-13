from pathlib import Path
import h5py
import numpy as np
import scipy as sp

from resnet.eval_utils.plot_utils import plot_single_device
from resnet.eval_utils.read_utils import get_perun_data
from resnet.eval_utils.read_utils import print_attrs

result_path = Path("/Users/philipphuber/Documents/Projects/ResNet/ResNet/experiments/")
exp_list = ["128g256b4w100e"]

for exp in exp_list:
    perun_h5_file = Path(result_path, exp, "perun", "perun.hdf5")
    h5val = h5py.File(perun_h5_file, 'r')
    #h5val.visititems(print_attrs)
    perun_data = get_perun_data(h5val)
    # Get Total energies
    lim = 5
    for key, _ in perun_data.items():
        for num, _ in perun_data[key]["gpu"].items():
            lim = 1.0e+09
            lim = 210
            power = perun_data[key]["gpu"][num]["power"]
            time = perun_data[key]["gpu"][num]["timesteps"]
            freq = perun_data[key]["gpu"][num]["util"]

            high = [x for x in freq if x > lim]
            low = [x for x in freq if x <= lim]
            print(key, num, len(low))

            #integrated = sp.integrate.trapezoid(freq, x=time)
            #mean = (integrated / time[-1]) * 10**(-6)
            #integrated = sp.integrate.trapezoid(power, x=time)
            #mean = (integrated / time[-1])
            #print(key, num, mean)

            #plot_single_device(power=power, time=time, label=f"{key}_{num}_gpu")


        #for num, _ in perun_data[key]["cpu"].items():
        #    time = perun_data[key]["gpu"][num]["timesteps"]
        #    cpu_util = perun_data[key]["cpu"][num]["util"]
        #    high = [x for x in cpu_util if x > lim]
        #    low = [x for x in cpu_util if x < lim]
        #    print(key, num, (len(high) / len(low)))
