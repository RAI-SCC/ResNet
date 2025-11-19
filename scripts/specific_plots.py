from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from cycler import cycler


kitcolors = {
    "brown": (167 / 255, 130 / 255, 46 / 255),
    "purple": (163 / 255, 16 / 255, 124 / 255),
    "cyan": (35 / 255, 161 / 255, 224 / 255),
    "peagreen": (140 / 255, 182 / 255, 60 / 255),
    "yellow": (252 / 255, 229 / 255, 0 / 255),
    "orange": (223 / 255, 155 / 255, 27 / 255),
    "red": (162 / 255, 34 / 255, 35 / 255),
    "green": (0 / 255, 150 / 255, 130 / 255),
    "blue": (70 / 255, 100 / 255, 170 / 255),
    "white": (0 / 255, 150 / 255, 130 / 255),
    "black100": (0 / 255, 0 / 255, 0 / 255),
    "black70": (64 / 255, 64 / 255, 64 / 255),
}
plt.rcParams['axes.prop_cycle'] = cycler(color=list(kitcolors.values()))


def ws_gpuh(result_path):
    target_path = Path(result_path, "figs", "special_plots")
    os.makedirs(target_path, exist_ok=True)
    target_path = Path(result_path, "figs", "special_plots", "ws_gpuh_scaling")
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 1.5), gridspec_kw={'width_ratios': [1.5, 1]})

    fs = 5
    ms = 1
    lw = 1
    elw = 1
    alpha = 0.8

    vals = {}
    # ws gbs 256
    vals["ws_gbs_256"] = {}
    vals["ws_gbs_256"]["label"] = "Const. GBS = 256"
    vals["ws_gbs_256"]["folders"] = ['1g256b4w100e16sf', '2g128b4w100e8sf', '4g64b4w100e4sf', '8g32b4w100e2sf', '16g16b4w100e']
    vals["ws_gbs_256"]["gpus"] = [1, 2, 4, 8, 16]
    vals["ws_gbs_256"]["lbs"] = [256, 128, 64, 32, 16]
    vals["ws_gbs_256"]["gbs"] = [256, 256, 256, 256, 256]
    vals["ws_gbs_256"]["nsamples"] = [80072, 160145, 320291, 640583, 1281167]
    vals["ws_gbs_256"]["mean_energy"] = [2.4801675846352698, 3.8994012149453767, 6.934681734108912, 15.011009015235349, 33.27702351990173]
    vals["ws_gbs_256"]["rmse_energy"] = [0.025093654649451274, 0.03187974058490285, 0.06258504328886773, 0.20412648197251662, 0.23837288374649154]
    vals["ws_gbs_256"]["mean_gpuh"] = [5.1389366319444445, 10.252672309027778, 21.936998697916668, 47.9395078125, 113.66412326388891]
    vals["ws_gbs_256"]["rmse_gpuh"] = [0.014967692621660882, 0.021665674837891457, 0.05364724769744266, 0.09682621791422925, 0.1041821566897172]
    # ws gbs 8k
    vals["ws_gbs_8k"] = {}
    vals["ws_gbs_8k"]["label"] = "Const. GBS = 8192"
    vals["ws_gbs_8k"]["folders"] = ['32g256b4w100e8sf', '64g128b4w100e4sf', '128g64b4w100e2sf', '256g32b4w100e']
    vals["ws_gbs_8k"]["gpus"] = [32, 64, 128, 256]
    vals["ws_gbs_8k"]["lbs"] = [256, 128, 64, 32]
    vals["ws_gbs_8k"]["gbs"] = [8192, 8192, 8192, 8192]
    vals["ws_gbs_8k"]["nsamples"] = [160145, 320291, 640583, 1281167]
    vals["ws_gbs_8k"]["mean_energy"] = [3.8067864522045873, 7.5333364526834545, 15.497942629295963, 33.2573029742361]
    vals["ws_gbs_8k"]["rmse_energy"] =  [0.010050782120950607, 0.07565018662804106, 0.06889202545582743, 0.11167512254851615]
    vals["ws_gbs_8k"]["mean_gpuh"] = [14.287587239583335, 27.19115060763889, 54.27109809027777, 113.97601562500002]
    vals["ws_gbs_8k"]["rmse_gpuh"] = [0.13976259856393078, 0.37659955282182406, 0.6500164783732909, 0.1379945730827119]
    # ws lbs 256
    vals["ws_lbs_256"] = {}
    vals["ws_lbs_256"]["label"] = "Const. LBS = 256"
    vals["ws_lbs_256"]["folders"] = ['1g256b4w100e256sf', '2g256b4w100e128sf', '4g256b4w100e64sf', '8g256b4w100e32sf', '16g256b4w100e16sf',
               '32g256b4w100e8sf', '64g256b4w100e4sf', '128g256b4w100e2sf', '256g256b4w100e']
    vals["ws_lbs_256"]["gpus"] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    vals["ws_lbs_256"]["lbs"] = [256, 256, 256, 256, 256, 256, 256, 256, 256],
    vals["ws_lbs_256"]["gbs"] = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    vals["ws_lbs_256"]["nsamples"] = [5004, 10009, 20018, 40036, 80072, 160145, 320291, 640583, 1281167]
    vals["ws_lbs_256"]["mean_energy"] = [0.18188253932638007, 0.2669868710051297, 0.4606460902761998, 0.9388569572898305, 1.8773880737510615,
                   3.8067864522045873, 7.5210982715111925, 15.259855984611093, 30.2074785081961]
    vals["ws_lbs_256"]["rmse_energy"] = [0.0006584721675040511, 0.004253398359366104, 0.004508553688952146, 0.0068663146139803675,
                   0.013660071597147534, 0.010050782120950607, 0.036434596951436964, 0.09094601621826873, 0.09984616597351016]
    vals["ws_lbs_256"]["mean_gpuh"] = [0.4108456963433159, 0.8108969862196181, 1.717406711154514, 3.4819896918402784, 7.0243912760416665,
                  14.287587239583335, 28.237729600694443, 58.232215277777776, 113.48748437500001]
    vals["ws_lbs_256"]["rmse_gpuh"] = [0.002447083882372398, 0.005561857090943572, 0.023016967917907367, 0.03256237243775963,
                 0.04082715472001487, 0.13976259856393078, 0.22305416388884503, 0.10207500462794077, 0.09380972480288226]

    x_labels = vals["ws_lbs_256"]["nsamples"]
    x_tick_pos = range(len(x_labels))

    ax1.set_xlabel("# Samples", fontsize=fs)
    ax1.set_xticks(x_tick_pos)
    ax1.set_xticklabels(x_labels, fontsize=fs)
    ax1.set_ylabel("GPU h [h]", fontsize=fs)
    ax1.tick_params(axis='y', labelsize=fs)

    ax2.set_xlabel("GPU h [h]", fontsize=fs)
    ax2.set_ylabel("Energy [kWh]", fontsize=fs)
    ax2.tick_params(axis='y', labelsize=fs)
    ax2.tick_params(axis='x', labelsize=fs)

    marker = ["o", "o", "o"]
    colors = ["C0", "C1", "C2"]

    for num, key in enumerate(vals):
        x1 = vals[key]["nsamples"]
        y1 = vals[key]["mean_gpuh"]
        y1_rmse = vals[key]["rmse_gpuh"]
        y2 = vals[key]["mean_energy"]
        y2_rmse = vals[key]["rmse_energy"]
        label = vals[key]["label"]

        x1_idx = []
        for x1_val in x1:
            if x1_val in x_labels:
                x1_idx.append(x_labels.index(x1_val))

        ax1.errorbar(x1_idx, y1, yerr=y1_rmse, marker=marker[num], ms=ms, linestyle=':', color=colors[num], label=label, lw=lw, capsize=2, elinewidth=elw, alpha=alpha)
        ax2.errorbar(y1, y2, xerr=y1_rmse, yerr=y2_rmse, marker=marker[num], ms=ms, linestyle=':', color=colors[num], label=label, lw=lw, capsize=2, elinewidth=elw, alpha=alpha)
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax2.set_xscale('log')
    ax1.legend(fontsize=fs)
    ax2.legend(fontsize=fs)
    plt.savefig(target_path, dpi=300, bbox_inches='tight')

# do plots
result_path = Path("/Users/philipphuber/Documents/Projects/ResNet/experiments/")
ws_gpuh(result_path)