import cmapPy.pandasGEXpress.parse_gctx as pg
import cmapPy.pandasGEXpress.subset_gctoo as sg
import pandas as pd
import os
import numpy as np
from scipy import stats


def find_closest_corr(train_data, meta, input_profile, cell):
    best_corr = -1
    best_ind = -1
    for i, p in enumerate(train_data):
        if meta[i][0] != cell:
            continue
        p_corr = stats.pearsonr(p.flatten(), input_profile.flatten())[0]
        if p_corr > best_corr:
            best_corr = p_corr
            best_ind = i
    # best_corr = stats.pearsonr(train_data[best_ind].flatten(), test_profile.flatten())[0]
    return best_corr, meta[best_ind]


def parse_data(file):
    print("Parsing data at " + file)
    df = pd.read_csv(file, sep="\t")
    df.reset_index(drop=True, inplace=True)
    try:
        df = df.drop('Unnamed: 0', 1)
    except Exception as e:
        pass
    try:
        df = df.drop('distil_id', 1)
    except Exception as e:
        pass
    print("Total: " + str(df.shape))
    #df = df[(df['cell_id'] == "MCF7") | (df['cell_id'] == "PC3")]
    df = df[(df['pert_type'] == "trt_cp") | (df['pert_type'] == "trt_sh") |
            (df['pert_type'] == "trt_sh.cgs") | (df['pert_type'] == "trt_sh.css") |
            (df['pert_type'] == "trt_oe") | (df['pert_type'] == "trt_lig")]
    # df = df[(df['pert_type'] == "trt_cp")]
    print("Cell filtering: " + str(df.shape))
    df = df.groupby(['cell_id', 'pert_id']).filter(lambda x: len(x) >= 2)
    print("Pert filtering: " + str(df.shape))
    # df = df.drop_duplicates(['cell_id', 'pert_id', 'pert_idose', 'pert_itime', 'pert_type'])
    # df = df.groupby(['cell_id', 'pert_id'], as_index=False).mean()
    # df = df.groupby(['cell_id', 'pert_id', 'pert_type'], as_index=False).mean()  # , 'pert_type'
    # print("Merging: " + str(df.shape))
    #df.pert_type.value_counts().to_csv("trt_count_final.tsv", sep='\t')
    cell_ids = df["cell_id"].values
    pert_ids = df["pert_id"].values
    all_pert_ids = set(pert_ids)
    pert_idose = df["pert_idose"].values
    pert_itime = df["pert_itime"].values
    pert_type = df["pert_type"].values
    perts = np.stack([cell_ids, pert_ids, pert_idose, pert_itime, pert_type]).transpose()  # , pert_idose, pert_itime, pert_type
    df = df.drop(['cell_id', 'pert_id', 'pert_idose', 'pert_itime', 'pert_type'],
                 1)  # , 'pert_idose', 'pert_itime', 'pert_type'
    try:
        df = df.drop('cid', 1)
    except Exception as e:
        pass
    data = df.values
    #data = data / max(np.max(data), abs(np.min(data)))
    # for i in range(len(data)):
    #    data[i] = data[i] / max(np.max(data[i]), abs(np.min(data[i])))
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = np.expand_dims(data, axis=-1)
    return data, perts, all_pert_ids


def read_profile(file, genes):
    df = pd.read_csv(file, sep="\t")
    df["mean"] = df.mean(numeric_only=True, axis=1)
    profile = []
    for g in genes:
        if len(df[(df['Gene_Symbol'] == g)]) != 0:
            profile.append(df[(df['Gene_Symbol'] == g)][df.columns[3]].tolist()[0])
        else:
            profile.append(0)
    profile = np.asarray(profile)
    profile = (profile - np.min(profile)) / (np.max(profile) - np.min(profile))
    return profile


data_folder = "/home/user/data/DeepFake/"
os.chdir(data_folder)

gctx_file = "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx"

sig_info7 = pd.read_csv("GSE70138_Broad_LINCS_sig_info_2017-03-06.txt", sep="\t")
gene_info = pd.read_csv("GSE92742_Broad_LINCS_gene_info.txt", sep="\t", dtype=str)
genes = gene_info["pr_gene_symbol"][gene_info["pr_is_lm"] == "1"].tolist()

df_a375 = read_profile("validation_data/a375_foxm1.txt", genes)
df_mcf7 = read_profile("validation_data/mcf7_foxm1.txt", genes)

data, meta, all_pert_ids = parse_data("lincs_phase_1_2.tsv")

closest_cor, info = find_closest_corr(data, meta, df_a375, "A375")
print(closest_cor)
print(info)
closest_cor, info = find_closest_corr(data, meta, df_mcf7, "MCF7")
print(closest_cor)
print(info)