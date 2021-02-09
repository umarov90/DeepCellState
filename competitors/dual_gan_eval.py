import argparse
import os
import pickle

from scipy import stats
from competitors import VAE_anchor, dual_gan
from figures import profiles_viz
from CellData import CellData
import numpy as np
import pandas as pd
import random
from shutil import copyfile


random.seed(0)
np.random.seed(0)
folds_folder = "../data/folds/"
test_folds = ["30percent"]
input_size = 978
latent_dim = 128

wdir = open("data_dir").read().strip() + "dual_gan"
if not os.path.exists(wdir):
    os.makedirs(wdir)
os.chdir(wdir)
# copyfile("/home/user/PycharmProjects/DeepFake/deepfake.py", "deepfake.py")

for r, test_fold in enumerate(test_folds):
    test_fold = str(test_fold)
    # cell_data = CellData("../data/lincs_phase_1_2.tsv", folds_folder + test_fold, "MCF7,PC3,", "trt_cp")
    # pickle.dump(cell_data, open("cell_data30.p", "wb"))
    cell_data = pickle.load(open("../cell_data30.p", "rb"))
    generators = dual_gan.get_generators(input_size, cell_data)
    results = {}
    seen_perts = []
    print("Total test objects: " + str(len(cell_data.test_data)))
    all_results = []
    test_trt = "trt_cp"
    for i in range(len(cell_data.test_data)):
        if i % 100 == 0:
            print(str(i) + " - ", end="", flush=True)
        test_meta_object = cell_data.test_meta[i]
        if test_meta_object[2] != test_trt:
            continue
        if test_meta_object[0] not in ["MCF7"]:
            continue
        closest, closest_profile, mean_profile, all_profiles = cell_data.get_profile(cell_data.test_data,
                                                                                     cell_data.meta_dictionary_pert_test[
                                                                                         test_meta_object[1]],
                                                                                     test_meta_object)
        if closest_profile is None:
            continue
        seen_perts.append(test_meta_object[1])
        test_profile = np.asarray([cell_data.test_data[i]])

        decoded1 = generators[test_meta_object[0]].predict(closest_profile)

        all_results.append(str(stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]))


with open("dual_gan_results.csv", 'w+') as f:
    f.write("\n".join(all_results))
    f.write("\n")

