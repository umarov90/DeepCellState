import os
import pandas as pd
import numpy as np
import random
from CellData import CellData
from scipy import stats

# For reproducibility
random.seed(0)
np.random.seed(0)

os.chdir(open("../data_dir").read().strip())
# Load fixed order of perts and cell types
perts = pd.read_csv("Hodos/their_data/hodos_data_large_tensor/perts.csv",
                    sep="\t", header=None, names=["Values"])['Values'].values.tolist()
cell_types = pd.read_csv("Hodos/their_data/hodos_data_large_tensor/cell_types.csv",
                         sep="\t", header=None, names=["Values"])['Values'].values.tolist()

# Load all the data
cell_data = CellData("Hodos/their_data/hodos_data_large_tensor", None, None, "trt_cp", revision=True)

# Load the previously constructed folds
folds = []
for i in range(10):
    folds.append(np.loadtxt("Hodos/their_data/hodos_folds_their_data/"+str(i+1), dtype='str'))

# Iterate through each fold
perf = []
for i, fold in enumerate(folds):
    print(f"Fold {i+1}")
    for cell in cell_types:
        res_hodos_dnpp = pd.read_csv("Hodos/their_data/output/fold_"+str(i+1) + "/" + cell + "_DNPP.csv", delimiter=",",
                                     header=None).values
        res_hodos_falrtc = pd.read_csv("Hodos/their_data/output/fold_" + str(i + 1) + "/" + cell + "_FaLRTC.csv", delimiter=",",
                                       header=None).values
        for pert_index, pert in enumerate(perts):
            if cell + "," + pert in fold:
                ground_truth = cell_data.get_profile_cell_pert(cell_data.train_data, cell_data.meta_dictionary_pert[pert], cell, pert)
                ground_truth = np.squeeze(ground_truth)
                prediction_dnpp = res_hodos_dnpp[pert_index]
                prediction_falrtc = res_hodos_falrtc[pert_index]
                # Compute pearson correlation between ground truth and prediction
                pcc_dnpp = stats.pearsonr(ground_truth.flatten(), prediction_dnpp.flatten())[0]
                pcc_falrtc = stats.pearsonr(ground_truth.flatten(), prediction_falrtc.flatten())[0]
                perf.append([pcc_dnpp, pcc_falrtc, cell, pert])

# output the results
with open("Hodos/their_data/perf_hodos.csv", 'w+') as f:
    f.write(f"DNPP,FaLRTC,Cell,Pert\n")
    for p in perf:
        f.write(f"{p[0]},{p[1]},{p[2]},{p[3]}\n")

