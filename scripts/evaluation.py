from pathlib import Path

from resnet.eval_utils.plot_utils import plot_scaling, plot_top1
# from resnet.eval_utils.read_utils import print_attrs


# result_path = Path("/home/scc/xy6660/ResNet/ResNet/experiments/")
result_path = Path("/Users/philipphuber/Documents/Projects/ResNet/ResNet/experiments/")

strong_scaling_list = ["16g256b4w100e", "32g128b4w100e", "64g64b4w100e", "128g32b4w100e"]
weak_scaling_list = ["16g256b4w100e", "32g256b4w100e", "64g256b4w100e", "128g256b4w100e"]


plot_scaling(result_path=result_path, scaling_list=strong_scaling_list, name="resnet4096gbs")
plot_scaling(result_path=result_path, scaling_list=weak_scaling_list, name="resnet256lbs")

plot_top1(result_path, strong_scaling_list, name="resnet_sclist")

# h5val.visititems(print_attrs)
