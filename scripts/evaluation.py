from pathlib import Path

from resnet.eval_utils.plot_utils import plot_scaling, plot_top1
# from resnet.eval_utils.read_utils import print_attrs


# result_path = Path("/home/scc/xy6660/ResNet/ResNet/experiments/")
result_path = Path("/hkfs/work/workspace/scratch/vm6493-resnet_imagenet/ResNet/experiments")

constant_gbs_list = ["32g256b4w100e", "64g128b4w100e", "128g64b4w100e", "256g32b4w100e"]
constant_gbs_slurms = [3040217, 3047906, 3047836, 3047855]
constant_gbs_path =[]
for i in range(len(constant_gbs_list)):
    constant_gbs_path.append(
        Path(
            constant_gbs_list[i], str(constant_gbs_slurms[i])
            )
        )


constant_lbs_list = ["32g256b4w100e", "64g256b4w100e", "128g256b4w100e", "256g256b4w100e"]
constant_lbs_slurms = [3040217, 3040210, 3040222, 3062064]
constant_lbs_path = []
for i in range(len(constant_lbs_list)):
    constant_lbs_path.append(
        Path(
            constant_lbs_list[i], str(constant_lbs_slurms[i])
            ))


plot_scaling(result_path=result_path, scaling_list=constant_gbs_path, name="resnet8192gbs")
plot_scaling(result_path=result_path, scaling_list=constant_lbs_path, name="resnet256lbs")

plot_top1(result_path, constant_gbs_path, name="resnet_lbslist", key="ss")
plot_top1(result_path, constant_lbs_path, name="resnet_gbslist", key="ws")

# h5val.visititems(print_attrs)
