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

sig_info = pd.read_csv("GSE92742_Broad_LINCS_sig_info.txt", sep="\t")
gene_info = pd.read_csv("GSE92742_Broad_LINCS_gene_info.txt", sep="\t", dtype=str)
landmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]
#sig_info.set_index("sig_id", inplace=True)
#sub_sig_info = sig_info[(sig_info["pert_type"] == "trt_oe") | (sig_info["pert_type"] == "trt_oe.mut") |
#                        (sig_info["pert_type"] == "trt_sh") | (sig_info["pert_type"] == "trt_sh.cgs") |
#                        (sig_info["pert_type"] == "trt_cp") | (sig_info["pert_type"] == "trt_lig")]
#sub_sig_info = sig_info.copy()
sig_info.set_index("sig_id", inplace=True)


gctoo = pg.parse("GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", cid=sig_info.index.tolist(), rid=landmark_gene_row_ids)
gctoo.col_metadata_df = sig_info.copy()

df_data_1 = gctoo.data_df
df_data_1 = df_data_1.transpose()

df_data_1["cell_id"] = gctoo.col_metadata_df["cell_id"]
df_data_1["pert_id"] = gctoo.col_metadata_df["pert_id"]
df_data_1["pert_idose"] = gctoo.col_metadata_df["pert_idose"]
df_data_1["pert_itime"] = gctoo.col_metadata_df["pert_itime"]
df_data_1["pert_type"] = gctoo.col_metadata_df["pert_type"]

#df_data_1.to_csv("cell_data_trt_sh.tsv", sep='\t')


# phase 2
sig_info7 = pd.read_csv("GSE70138_Broad_LINCS_sig_info_2017-03-06.txt", sep="\t")
gene_info = pd.read_csv("GSE92742_Broad_LINCS_gene_info.txt", sep="\t", dtype=str)
landmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]
#sig_info.set_index("sig_id", inplace=True)
#sub_sig_info = sig_info[(sig_info["pert_type"] == "trt_oe") | (sig_info["pert_type"] == "trt_oe.mut") |
#                        (sig_info["pert_type"] == "trt_sh") | (sig_info["pert_type"] == "trt_sh.cgs") |
#                        (sig_info["pert_type"] == "trt_cp") | (sig_info["pert_type"] == "trt_lig")]

#df_data_1 = df_data_1.drop(df_data_1[~df_data_1.pert_id.isin(overlap)].index.tolist())

#sub_sig_info = sig_info.copy()
sig_info7.set_index("sig_id", inplace=True)


gctoo = pg.parse("GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx", cid=sig_info7.index.tolist(), rid=landmark_gene_row_ids)
gctoo.col_metadata_df = sig_info7.copy()

df_data_2 = gctoo.data_df
df_data_2 = df_data_2.transpose()

df_data_2["cell_id"] = gctoo.col_metadata_df["cell_id"]
df_data_2["pert_id"] = gctoo.col_metadata_df["pert_id"]
df_data_2["pert_idose"] = gctoo.col_metadata_df["pert_idose"]
df_data_2["pert_itime"] = gctoo.col_metadata_df["pert_itime"]
df_data_2["pert_type"] = gctoo.col_metadata_df["pert_type"]

df_data_3 = pd.concat([df_data_1, df_data_2]).drop_duplicates().reset_index(drop=True)
print(df_data_1.shape)
print(df_data_2.shape)
print(df_data_3.shape)

df_data_3.to_csv("lincs_phase_1_2.tsv", sep='\t')
