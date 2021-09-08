import os
import pandas as pd
import numpy as np
import random
from pathlib import Path
from CellData import CellData
from random import shuffle

# For reproducibility
random.seed(0)
np.random.seed(0)

os.chdir(open("../data_dir").read().strip())
# Load fixed order of perts and cell types
perts = pd.read_csv("Hodos/our_data/perts.csv", sep="\t", header=None, names=["Values"])['Values'].values.tolist()
cell_types = pd.read_csv("Hodos/our_data/cell_types.csv", sep="\t", header=None, names=["Values"])['Values'].values.tolist()

# Get all the available profiles
all_data = []
cell_data = CellData("data/lincs_phase_1_2.tsv", None, None, "trt_cp")
for i in range(len(cell_data.train_data)):
    meta_object = cell_data.train_meta[i]
    all_data.append(meta_object[0] + "," + meta_object[1])

# shuffle the profiles for randomness
shuffle(all_data)

# split the profiles into ten folds
folds = np.array_split(all_data, 10)

# write the folds to the disk
for i, fold in enumerate(folds):
    with open("Hodos/our_data/hodos_folds_our_data/"+str(i+1), 'w+') as f:
        f.write('\n'.join(list(fold.flatten())))

# Construct input for methods from Hodos et al where test set is replaced by nan values
for i, fold in enumerate(folds):
    for cell in cell_types:
        profiles = []
        for pert in perts:
            p = None
            # This is the test profile
            if cell + "," + pert in fold:
                profiles.append(','.join(['nan' for _ in range(978)]))
                continue
            if pert in cell_data.meta_dictionary_pert.keys():
                p = cell_data.get_profile_cell_pert(cell_data.train_data, cell_data.meta_dictionary_pert[pert], cell, pert)
            # the profile is not test but it does not exist in the data
            if p is None:
                profiles.append(','.join(['nan' for _ in range(978)]))
                continue
            p = np.squeeze(p)
            profiles.append(','.join([str(num) for num in p]))
        print(f"Cell {cell} number of perts {len(profiles)}")

        # write the olds on the disk
        cell_path = "Hodos/our_data/input/fold_" + str(i + 1) + "/"
        Path(cell_path).mkdir(parents=True, exist_ok=True)
        with open(cell_path + cell + ".csv", 'w+') as f:
             f.write('\n'.join(profiles))

