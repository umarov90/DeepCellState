import cmapPy.pandasGEXpress.parse_gctx as pg
import cmapPy.pandasGEXpress.subset_gctoo as sg
import pandas as pd
import os
import numpy as np

data_folder = "/home/user/data/DeepFake/"
os.chdir(data_folder)

# phase 2
sig_info = pd.read_csv("GSE92742_Broad_LINCS_sig_info.txt", sep="\t")
sig_info7 = pd.read_csv("GSE70138_Broad_LINCS_sig_info_2017-03-06.txt", sep="\t")
sig_info7 = pd.concat([sig_info, sig_info7])
gene_info = pd.read_csv("GSE92742_Broad_LINCS_gene_info.txt", sep="\t", dtype=str)
landmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]

sig_info7.set_index("sig_id", inplace=True)
family = "glutamatergic"
test_perts = np.loadtxt(family, dtype='str', delimiter="; ").tolist()
sub_sig_info = sig_info7[sig_info7['pert_iname'].isin(test_perts)]

col_one_arr = list(set(sub_sig_info['pert_id'].to_list()))

np.savetxt(family + '_ids', np.asarray(col_one_arr), delimiter=',', fmt='%s')
