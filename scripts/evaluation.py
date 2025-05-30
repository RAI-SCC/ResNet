from pathlib import Path

import h5py

from resnet.eval_utils.plot_utils import plot_top1
from resnet.eval_utils.eval_utils import eval_scaling
from resnet.eval_utils.read_utils import print_attrs

# result_path = Path("/home/scc/xy6660/ResNet/ResNet/experiments/")
result_path = Path("/Users/philipphuber/Documents/Projects/ResNet/experiments/")

constant_gbs_list = {"32g256b4w100e": ["3172377", "3172802", "3173813", "3175067", "3179925"],
                     "64g128b4w100e": ["3172777", "3172801", "3173812", "3175066", "3179924"],
                     "128g64b4w100e": ["3172774", "3172799", "3173810", "3175064", "3179922"],
                     "256g32b4w100e": ["3172773", "3172797", "3173808", "3175062", "3179920"]}

constant_lbs_list = {"16g256b4w100e": ["3172378", "3172803", "3173814", "3175068", "3179926"],
                     "32g256b4w100e": ["3172377", "3172802", "3173813", "3175067", "3179925"],
                     "64g256b4w100e": ["3170541", "3172800", "3173811", "3175065", "3179923"],
                     #"128g256b4w100e": ["3171624", "3173809", "3175063", "3179921"],
                     "128g256b4w100e": ["3171624", "3172798", "3173809", "3175063", "3179921"],
                     "256g256b4w100e": ["3167090", "3172796", "3173807", "3175061", "3179919"]}

test_list = {"8g16b4w2e": ["3219827"]}
eval_scaling(result_path=result_path, scaling_list=test_list, name="test")

#eval_scaling(result_path=result_path, scaling_list=constant_gbs_list, name="resnet8192gbs")
#eval_scaling(result_path=result_path, scaling_list=constant_lbs_list, name="resnet256lbs")

#plot_top1(result_path, constant_gbs_list, name="resnet_constant_gbs", key="constant_gbs")
#plot_top1(result_path, constant_lbs_list, name="resnet_constant_lbs", key="constant_lbs")

#perun_h5_file = Path("/Users/philipphuber/Documents/Projects/ResNet/experiments/128g256b4w100e/3171624/perun/perun.hdf5")
#h5val = h5py.File(perun_h5_file, 'r')
#h5val.visititems(print_attrs)
