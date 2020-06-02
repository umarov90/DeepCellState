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

gctx_file = "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"

sig_info = pd.read_csv("GSE92742_Broad_LINCS_sig_info.txt", sep="\t")
sig_info7 = pd.read_csv("GSE70138_Broad_LINCS_sig_info_2017-03-06.txt", sep="\t")
gene_info = pd.read_csv("GSE92742_Broad_LINCS_gene_info.txt", sep="\t", dtype=str)
landmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]


overlap = np.intersect1d(sig_info.pert_id.unique(), sig_info7.pert_id.unique())
pert = "KRAS"
dose = "2 µL"
time = "96 h"
sub_sig_info = sig_info[(sig_info["pert_iname"] == pert) & (sig_info["pert_idose"] == dose) & (sig_info["pert_itime"] == time)]
# set sig_id as index
sub_sig_info.set_index("sig_id", inplace=True)
sub_gctoo = pg.parse(gctx_file, cid=sub_sig_info.index.tolist(), rid=landmark_gene_row_ids)
# annotate with its corresponding sample annotations
sub_gctoo.col_metadata_df = sub_sig_info.copy()

df_data = sub_gctoo.data_df
av = df_data.mean(numeric_only=True, axis=1)
md = df_data.median(numeric_only=True, axis=1)
df_data['average'] = av
df_data['median'] = md
df_data = df_data.transpose()
df_data['corr_median'] = df_data.corrwith(md, axis=1)
df_data['corr_avg'] = df_data.corrwith(av, axis=1)
print(dose + " / " + time)
print("Count:\t" + str(len(sub_sig_info)))
print("AVG profile:\t" + str(df_data["corr_avg"].mean()))
print("Median profile:\t" +str(df_data["corr_median"].mean()))


df_data["cell_id"] = sub_gctoo.col_metadata_df["cell_id"]
df_data["pert_id"] = sub_gctoo.col_metadata_df["pert_id"]
df_data["pert_idose"] = sub_gctoo.col_metadata_df["pert_idose"]
df_data["pert_itime"] = sub_gctoo.col_metadata_df["pert_itime"]

#df_data["distil_id"] = vorinostat_gctoo.col_metadata_df["distil_id"]


df_data.to_csv("sub_data.tsv", sep='\t')
