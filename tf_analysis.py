import os
import random
import numpy as np
import pickle
import pandas as pd
import tensorflow
from tensorflow.python.keras import backend as K
from CellData import CellData
from tensorflow import keras
tensorflow.compat.v1.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.chdir(open("data_dir").read().strip())
model = "best_autoencoder_ext_val/"
autoencoder = keras.models.load_model(model + "main_model/")
cell_decoders = {"MCF7": pickle.load(open(model + "MCF7" + "_decoder_weights", "rb")),
                 "PC3": pickle.load(open(model +  "PC3" + "_decoder_weights", "rb"))}
encoder = autoencoder.get_layer("encoder")
decoder = autoencoder.get_layer("decoder")
autoencoder.get_layer("decoder").set_weights(cell_decoders["MCF7"])
cell_data = CellData("data/lincs_phase_1_2.tsv", "data/folds/ext_val")
symbols = np.loadtxt("data/gene_symbols.csv", dtype="str")

top_genes_num = 50
top_targets_num = 50
tf_data = {"MCF7": {}, "Average": {}}
tf_sum = {"MCF7": {}, "Average": {}}
tf_cell_types = {}
directory = "data/TFS"
cell_names = set()
for filename in os.listdir(directory):
    try:
        if filename not in symbols:
            continue
        top_genes = {}
        df = pd.read_csv(os.path.join(directory, filename), sep="\t", index_col="Target_genes")
        avg_columns = []
        mcf7_columns = []
        cell_types = []
        for col in df.columns:
            if "MCF-7" not in col:
                if "Average" not in col:
                    avg_columns.append(col)
            else:
                mcf7_columns.append(col)
            try:
                if "MCF-7" not in col and "Average" not in col:
                    tf_cell_types.setdefault(filename, set([])).add(col.split("|")[1])
            except:
                pass
        if len(mcf7_columns) == 0:
            continue
        df["MCF7"] = df[mcf7_columns].astype(float).mean(axis=1)
        df["Average"] = df[avg_columns].astype(float).mean(axis=1)
        df = df.drop(mcf7_columns, 1)
        df = df.drop(avg_columns, 1)
        df = df[df.index.isin(symbols)]
        top = df.sort_values("MCF7", ascending=False).head(top_targets_num)[df.MCF7 > 100]
        gene_list = top.index.to_list()
        if len(gene_list) < top_genes_num:
            continue
        tf_data["MCF7"][filename] = gene_list
        tf_data["Average"][filename] = df.sort_values("Average", ascending=False).head(top_targets_num).index.to_list()[:len(gene_list)]
    except Exception as e:
        print(e)

tf_vals = {}
for i in range(len(cell_data.train_data)):
    if cell_data.train_meta[i][0] != "MCF7":
        continue
    for tf in tf_data["MCF7"].keys():
        tf_vals.setdefault(tf, []).append(cell_data.train_data[i][list(symbols.flatten()).index(tf), 0])

tf_chosen = []
figure_vals_fold = {"MCF7": [], "Average": []}
figure_vals_num = {"MCF7": [], "Average": []}
sum = 0
n = 0
for key in ["MCF7", "Average"]:
    print(key + "________________________________________________")
    for tf in tf_data["MCF7"].keys():
        tf_info = np.asarray(tf_vals[tf])
        if np.max(tf_info) < 0.5:
            continue
        elif tf not in tf_chosen:
            tf_chosen.append(tf)

        output_layer = autoencoder.output
        input_layer = autoencoder.input
        gene = np.where(symbols==tf)[0][0]
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        tensorflow.random.set_seed(seed)
        loss = output_layer[:, gene, 0] - 5 * K.mean(input_layer)
        grads = K.gradients(loss, autoencoder.input)[0]
        grads = K.l2_normalize(grads)
        func = K.function([autoencoder.input], [loss, grads])
        input_profile = 0.0004 * np.random.random((1, 978, 1)) - 0.0002
        for i in range(10):
            loss_val, grads_val = func([input_profile])
            input_profile += grads_val

        a = autoencoder.predict(input_profile)
        pb = np.squeeze(a)
        idx = (-1 * pb).argsort()
        top_genes = symbols[idx[:top_genes_num]].tolist() # + symbols[idx[-10:]].tolist()
        common_genes = list(set(tf_data[key][tf]).intersection(top_genes))
        top_targets_num_final = min(top_targets_num, len(tf_data[key][tf]))
        # top_targets_num_final = top_targets_num
        res = len(common_genes) / ((top_targets_num_final / 978) * top_genes_num)
        print(tf + " " + str(res) + " " + str(np.max(tf_info)) + " " + str(np.min(tf_info)) + " " +
              str(np.mean(tf_info)) + " " + str(np.std(tf_info)) + " " + str(np.median(tf_info)))
        figure_vals_fold[key].append(res)
        figure_vals_num[key].append(len(common_genes))
        sum = sum + res
        n = n + 1
    print(key)
    print(sum/n)
    print("Done")

figure_data = np.stack([figure_vals_num["Average"], figure_vals_fold["Average"],
                        figure_vals_num["MCF7"], figure_vals_fold["MCF7"]], axis=0).transpose()
df = pd.DataFrame(data=figure_data, index=tf_chosen, columns=["Average_num", "Average_fold", "MCF7_num", "MCF7_fold"])
df.index.name = "TF"
df.to_csv("figures_data/tf_analysis_data.csv")
