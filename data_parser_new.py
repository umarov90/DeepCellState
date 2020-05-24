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

data_folder = "/home/user/data/DeepFake/"
os.chdir(data_folder)

gctx_file = "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"

sig_info = pd.read_csv("GSE92742_Broad_LINCS_sig_info.txt", sep="\t")
gene_info = pd.read_csv("GSE92742_Broad_LINCS_gene_info.txt", sep="\t", dtype=str)
landmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]
sig_info.set_index("sig_id", inplace=True)

gctoo = pg.parse(gctx_file, cid=sig_info.index.tolist(), rid=landmark_gene_row_ids)
gctoo.col_metadata_df = sig_info.copy()

# vorinostat_sig_id_info =  sig_info[sig_info["pert_iname"] == "vorinostat"]
# vorinostat_sig_id_info[0:5]
# vorinostat_sig_id_info.set_index("sig_id", inplace=True)
# vorinostat_sig_id_info.index

#.set_index("sig_id", inplace=True)
#df.col_metadata_df = sig_info

# .value_counts()[:10].index.tolist()

#a375_ids = df.col_metadata_df.index[df.col_metadata_df["cell_id"] == "A375"].tolist()
#mcf7_ids = df.col_metadata_df.index[df.col_metadata_df["cell_id"] == "MCF7"].tolist()

#a375_gctoo = sg.subset_gctoo(df, cid=a375_ids)
#mcf7_gctoo = sg.subset_gctoo(df, cid=mcf7_ids)

#a375_pert_ids = df.col_metadata_df.index[df.col_metadata_df["cell_id"] == "A375"].tolist()
#meta_data = df.col_metadata_df.tolist()

#vorinostat_sig_info = sig_info[(sig_info["pert_iname"] == "vorinostat") & (sig_info["pert_idose"] == "10 µM") & (sig_info["pert_itime"] == "24 h")]
# set sig_id as index
#vorinostat_sig_info.set_index("sig_id", inplace=True)
#vorinostat_gctoo = sg.subset_gctoo(df, cid=vorinostat_sig_info.index.tolist())
# annotate with its corresponding sample annotations
#vorinostat_gctoo.col_metadata_df = vorinostat_sig_info.copy()

df_data = gctoo.data_df
#df2 = mcf7_gctoo.data_df

#df1 = df1.drop('rid', 1)
#df2 = df2.drop('rid', 1)

df_data = df_data.transpose()
#df2 = df2.transpose()

df_data["cell_id"] = gctoo.col_metadata_df["cell_id"]
df_data["pert_id"] = gctoo.col_metadata_df["pert_id"]
df_data["pert_idose"] = gctoo.col_metadata_df["pert_idose"]
df_data["pert_itime"] = gctoo.col_metadata_df["pert_itime"]
df_data["distil_id"] = gctoo.col_metadata_df["distil_id"]


df_data.to_csv("cell_data.tsv", sep='\t')

# print("A375 " + str(a375_gctoo.data_df.shape))
# print("MCF7 " + str(mcf7_gctoo.data_df.shape))
