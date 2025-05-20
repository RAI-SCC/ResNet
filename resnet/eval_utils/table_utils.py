from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def rounded(val, position=0.01):
    """
    Rounds a number to a specific position and returns ot as str.

    Paremeters
    ----------
    val : float
        Number to be rounded
    position : float
        Position to be rounded at.

    Returns
    -------
    out_val : str
        Rounded number.
    """
    out_val = Decimal(str(val)).quantize(Decimal(str(position)), rounding=ROUND_HALF_UP)
    return str(out_val)


def make_table(result_path, data, scaling_list, name):

    path = Path(result_path, name+"_table_all.txt")

    with open(path, "w") as outf:
        print_list = [" ",
                      "$E(perun)$",
                      "$E(XCC)$",
                      "t(perun)",
                      "t(wct)",
                      "Top1V",
                      "Top5V",
                      "Top1T",
                      "Top5T",
                      ]
        line = " & ".join(print_list)
        line = line + "\\\\ \n"
        outf.write(line)
        print_list = [" ",
                      "(kWh)",
                      "(kWh)",
                      "(min)",
                      "(min)",
                      "(\\%)",
                      "(\\%)",
                      "(\\%)",
                      "(\\%)",
                      ]
        line = " & ".join(print_list)
        line = line + "\\\\ \n"
        outf.write(line)

        for experiment in data:
            gpus = data[experiment]["gpus"]
            lbs = data[experiment]["lbs"]
            gbs = data[experiment]["gbs"]
            outf.write("\\midrule \\midrule \n")
            line = "\multicolumn{9}{c}{" + f"GPUs: {gpus}, LBS: {lbs} GBS: {gbs}" + "} \\\\ \n"
            outf.write(line)
            outf.write("\\midrule \n")

            mean_perun_energy = data[experiment]["mean"]["perun_energy"]
            mean_perun_time = data[experiment]["mean"]["perun_time"]
            mean_slurm_energy = data[experiment]["mean"]["slurm_energy"]
            mean_slurm_time = data[experiment]["mean"]["slurm_time"]
            mean_top1_error_valid = data[experiment]["mean"]["top1_error_valid"]
            mean_top5_error_valid = data[experiment]["mean"]["top5_error_valid"]
            mean_top1_error_train = data[experiment]["mean"]["top1_error_train"]
            mean_top5_error_train = data[experiment]["mean"]["top5_error_train"]
            rmse_perun_energy = data[experiment]["rmse"]["perun_energy"]
            rmse_perun_time = data[experiment]["rmse"]["perun_time"]
            rmse_slurm_energy = data[experiment]["rmse"]["slurm_energy"]
            rmse_slurm_time = data[experiment]["rmse"]["slurm_time"]
            rmse_top1_error_valid = data[experiment]["rmse"]["top1_error_valid"]
            rmse_top5_error_valid = data[experiment]["rmse"]["top5_error_valid"]
            rmse_top1_error_train = data[experiment]["rmse"]["top1_error_train"]
            rmse_top5_error_train = data[experiment]["rmse"]["top5_error_train"]

            for slurm_id in scaling_list[experiment]:
                perun_energy = data[experiment][slurm_id]["perun_energy"]
                perun_time = data[experiment][slurm_id]["perun_time"]
                slurm_energy = data[experiment][slurm_id]["slurm_energy"]
                slurm_time = data[experiment][slurm_id]["slurm_time"]
                top1_error_valid = data[experiment][slurm_id]["top1_error_valid"]
                top5_error_valid = data[experiment][slurm_id]["top5_error_valid"]
                top1_error_train = data[experiment][slurm_id]["top1_error_train"]
                top5_error_train = data[experiment][slurm_id]["top5_error_train"]
                print_list = [" ",
                              rounded(perun_energy),
                              rounded(slurm_energy),
                              rounded(perun_time),
                              rounded(slurm_time),
                              rounded(top1_error_valid),
                              rounded(top5_error_valid),
                              rounded(top1_error_train),
                              rounded(top5_error_train),
                              ]
                line = " & ".join(print_list)
                line = line + "\\\\ \n"
                outf.write(line)

            outf.write("\\midrule \n")
            print_list = ["mean",
                          rounded(mean_perun_energy) + " $\pm$ " + rounded(rmse_perun_energy),
                          rounded(mean_slurm_energy) + " $\pm$ " + rounded(rmse_slurm_energy),
                          rounded(mean_perun_time) + " $\pm$ " + rounded(rmse_perun_time),
                          rounded(mean_slurm_time) + " $\pm$ " + rounded(rmse_slurm_time),
                          rounded(mean_top1_error_valid),  # + " $\pm$ " + rounded(rmse_top1_error_valid),
                          rounded(mean_top5_error_valid),  # + " $\pm$ " + rounded(rmse_top5_error_valid),
                          rounded(mean_top1_error_train),  # + " $\pm$ " + rounded(rmse_top1_error_train),
                          rounded(mean_top5_error_train),  # + " $\pm$ " + rounded(rmse_top5_error_train),
                          ]
            line = " & ".join(print_list)
            line = line + "\\\\ \n"
            outf.write(line)


