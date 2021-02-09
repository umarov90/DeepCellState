import argparse
import os
from scipy import stats
import deepfake
from figures import profiles_viz
from CellData import CellData
import numpy as np
import pandas as pd
import random
from shutil import copyfile

def get_options():
    parser = argparse.ArgumentParser(description='Version: 1.0')
    parser.add_argument('-O', metavar='output', default="DeepCellState_output",
                        help='Output directory')
    parser.add_argument('-CT', metavar='cell types', default="",
                        type=str, help='Comma separated list of cell types to use in addition to MCF7 and PC3')
    parser.add_argument('-PT', metavar='pert type', default="trt_cp",
                        type=str, help='Perturbation type to be used, defaults to trt_cp')
    parser.add_argument('-N', metavar='number of runs', default=2,
                        type=int, help='Number of models trained for each fold.'
                                       ' The model with best validation performance is picked.')
    parser.add_argument('-SM', metavar='special models', default=0,
                        type=int, help='Set to 1 to train drug MoA family models or'
                                       ' set to 2 to train external validation model.'
                                       ' Defaults to 0, i.e. 10-fold cross-validation.')

    args = parser.parse_args()

    return args


def test_loss(prediction, ground_truth):
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))


def main():
    random.seed(0)
    np.random.seed(0)
    args = get_options()
    regul_stren = 2
    if args.CT is not None and len(args.CT)>0:
        regul_stren = 1
    folds_folder = "../data/folds/"
    if args.PT == "trt_sh":
        folds_folder = "../data/folds_sh+cp/"
    if args.SM == 0:
        test_folds = range(1, 11)
    elif args.SM == 1:
        test_folds = ["antibiotics_ids", "adrenergic_ids", "cholinergic_ids",
                      "5-HT modulator_ids", "TKI_ids", "COX inh._ids",
                      "histaminergic_ids", "antipsychotic_ids", "GABAergic_ids", "dopaminergic_ids"]
    else:
        test_folds = ["ext_val"]
        regul_stren = 3
    input_size = 978
    latent_dim = 128

    wdir = open("data_dir").read().strip() + args.O
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    os.chdir(wdir)
    # copyfile("/home/user/PycharmProjects/DeepFake/deepfake.py", "deepfake.py")
    df = pd.read_csv("../data/GSE70138_Broad_LINCS_pert_info.txt", sep="\t")
    good = []
    tsne_perts = []
    tsne_input = []
    tsne_latent = []
    tr1 = []
    tr2 = []
    ts1 = []
    ts2 = []
    cell_data = CellData("../data/lincs_phase_1_2.tsv", folds_folder + "30percent", "MCF7,PC3," + args.CT, args.PT)
    for i in range(len(cell_data.test_data)):
        test_meta_object = cell_data.test_meta[i]
        if test_meta_object[0] != "PC3":
            continue
        p = np.squeeze(cell_data.test_data[i])
        ts1.append(','.join([str(num) for num in p]))
        _, closest_profile, _, _ = cell_data.get_profile(cell_data.test_data,
                                                                            cell_data.meta_dictionary_pert_test[
                                                                            test_meta_object[1]],
                                                                            test_meta_object)
        p2 = np.squeeze(closest_profile)
        ts2.append(','.join([str(num) for num in p2]))

    with open("PC3_ts.csv", 'w+') as f:
        f.write('\n'.join(ts1))

    with open("MCF7_ts.csv", 'w+') as f:
        f.write('\n'.join(ts2))

    for i in range(len(cell_data.train_data)):
        train_meta_object = cell_data.train_meta[i]
        p = np.squeeze(cell_data.train_data[i])
        if train_meta_object[0] == "PC3":
            tr1.append(','.join([str(num) for num in p]))
        else:
            tr2.append(','.join([str(num) for num in p]))

    with open("PC3_tr.csv", 'w+') as f:
        f.write('\n'.join(tr1))

    with open("MCF7_tr.csv", 'w+') as f:
        f.write('\n'.join(tr2))
    exit()


if __name__ == '__main__':
    main()
