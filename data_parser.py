import cmapPy.pandasGEXpress.parse_gctx as pg
import pandas as pd
import os

data_folder = "/home/user/data/DeepFake/LINCS/"
os.chdir(data_folder)

# Phase 1
sig_info = pd.read_csv("GSE92742_Broad_LINCS_sig_info.txt", sep="\t")
gene_info = pd.read_csv("GSE92742_Broad_LINCS_gene_info.txt", sep="\t", dtype=str)
landmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]
sub_sig_info = sig_info[(sig_info["pert_type"] == "trt_cp")
                        (sig_info["pert_type"] == "trt_sh") |
                        (sig_info["pert_type"] == "trt_sh.cgs")]
sub_sig_info.set_index("sig_id", inplace=True)

gctoo = pg.parse("GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx", cid=sub_sig_info.index.tolist(), rid=landmark_gene_row_ids)
gctoo.col_metadata_df = sub_sig_info.copy()

df_data_1 = gctoo.data_df
rids = df_data_1.index.tolist()
symbols = []
for i in range(len(rids)):
    gene_symbol = gene_info["pr_gene_symbol"][gene_info["pr_gene_id"] == rids[i]].values[0]
    symbols.append(gene_symbol)

with open("gene_symbols.csv", 'w+') as f:
    f.write('\n'.join(symbols))

df_data_1 = df_data_1.transpose()

df_data_1["cell_id"] = gctoo.col_metadata_df["cell_id"]
df_data_1["pert_id"] = gctoo.col_metadata_df["pert_id"]
df_data_1["pert_idose"] = gctoo.col_metadata_df["pert_idose"]
df_data_1["pert_itime"] = gctoo.col_metadata_df["pert_itime"]
df_data_1["pert_type"] = gctoo.col_metadata_df["pert_type"]

# Phase 2
sig_info7 = pd.read_csv("GSE70138_Broad_LINCS_sig_info_2017-03-06.txt", sep="\t")
sub_sig_info7 = sig_info7[(sig_info7["pert_type"] == "trt_cp")]
sub_sig_info7.set_index("sig_id", inplace=True)


gctoo = pg.parse("GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx", cid=sub_sig_info7.index.tolist(), rid=landmark_gene_row_ids)
gctoo.col_metadata_df = sub_sig_info7.copy()

df_data_2 = gctoo.data_df
df_data_2 = df_data_2.transpose()

df_data_2["cell_id"] = gctoo.col_metadata_df["cell_id"]
df_data_2["pert_id"] = gctoo.col_metadata_df["pert_id"]
df_data_2["pert_idose"] = gctoo.col_metadata_df["pert_idose"]
df_data_2["pert_itime"] = gctoo.col_metadata_df["pert_itime"]
df_data_2["pert_type"] = gctoo.col_metadata_df["pert_type"]

df_data_3 = pd.concat([df_data_1, df_data_2]).drop_duplicates().reset_index(drop=True)
df_data_3.to_csv("lincs_phase_1_2.tsv", sep='\t')
