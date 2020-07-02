# Broad_LINCS_Level2_GEX_n113012x978_2015-12-31.gct
# 978 genes
# 112634 experiments
# 33 cells [('A549', 9012), ('MCF7', 8332)]

# GSE70138_Broad_LINCS_Level2_GEX_n78980x978_2015-06-30.gct
# 978 genes
# 78980 experiments
# 16 cells [('A549', 8920), ('MCF7', 8147)]

# from collections import Counter
#
# counter = 0
# with open("Broad_LINCS_Level2_GEX_n113012x978_2015-12-31.gct") as f:
#     for line in f:
#         print(line[:140])
#         vals = line.split("\t")
#         size = len(vals)
#         val_set = set(vals)
#         c = Counter(vals)
#         print(c.most_common(4))
#         counter = counter + 1
#
# print(counter)

import cmapPy.pandasGEXpress.parse_gctx as pg
import cmapPy.pandasGEXpress.subset_gctoo as sg
import pandas as pd
import os
import numpy as np

data_folder = "/home/user/data/DeepFake/"
os.chdir(data_folder)

# phase 2
sig_info7 = pd.read_csv("GSE70138_Broad_LINCS_sig_info_2017-03-06.txt", sep="\t")
gene_info = pd.read_csv("GSE92742_Broad_LINCS_gene_info.txt", sep="\t", dtype=str)
landmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]

sig_info7.set_index("sig_id", inplace=True)
family = "antipsychotic"
test_perts = np.loadtxt(family, dtype='str', delimiter="; ").tolist()
sub_sig_info = sig_info7[sig_info7['pert_iname'].isin(test_perts)]

col_one_arr = list(set(sub_sig_info['pert_id'].to_list()))

np.savetxt(family + '_ids', np.asarray(col_one_arr), delimiter=',', fmt='%s')
