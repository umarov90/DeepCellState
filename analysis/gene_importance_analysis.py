import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import numpy as np
from numpy import zeros
from copy import deepcopy
from scipy import stats
import pickle
import pandas as pd
from collections import Counter
from CellData import CellData

os.chdir(open("../data_dir").read())
# cell_data = CellData("LINCS/lincs_phase_1_2.tsv", "LINCS/folds/ext_val")
# pickle.dump(cell_data, open("cell_data.p", "wb"))
cell_data = pickle.load(open("cell_data.p", "rb"))
input_size = 978
latent_dim = 128
model = "sub2/best_autoencoder_ext_val/"
autoencoder = keras.models.load_model(model + "main_model/")
cell_decoders = {}
for cell in cell_data.cell_types:
    cell_decoders[cell] = pickle.load(open(model + cell + "_decoder_weights", "rb"))
encoder = autoencoder.get_layer("encoder")
decoder = autoencoder.get_layer("decoder")



symbols = np.loadtxt("gene_symbols.csv", dtype="str")

final_sets = {}
importance_scores = zeros((len(cell_data.cell_types), input_size))
for cn, key in enumerate(cell_data.cell_types):
    print(key + "________________________________________________")
    autoencoder.get_layer("decoder").set_weights(cell_decoders[key])
    total_results = []
    seen_perts = []
    num = 0
    for i in range(len(cell_data.train_data)):
        if i % 100 == 0:
            print(str(i) + " - ", end="", flush=True)
        train_meta_object = cell_data.train_meta[i]
        if train_meta_object[0] != key:
            continue
        closest, closest_profile, mean_profile, all_profiles = cell_data.get_profile(cell_data.train_data,
                                                                                     cell_data.meta_dictionary_pert[
                                                                                         train_meta_object[1]],
                                                                                     train_meta_object)
        if closest_profile is None:
            continue
        if train_meta_object[1] in seen_perts:
            continue
        seen_perts.append(train_meta_object[1])
        num = num + 1
        test_profile = np.asarray([cell_data.train_data[i]]).flatten()
        # results = []
        # for k in range(1000):
        #     damaged_profile = np.zeros(closest_profile.shape)
        #     inds = random.sample(range(0, 978), 100)
        #     damaged_profile[0, inds] = closest_profile[0, inds]
        #
        #     decoded1 = autoencoder.predict(damaged_profile)
        #     pcc = stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]
        #     results.append([pcc, inds])
        # results.sort(key=lambda x: x[0], reverse=True)
        # results = results[:10]
        # total_results.extend(results)
        results = []
        for k in range(20):
            damaged_profile = closest_profile.copy()
            inds = random.sample(range(0, 978), 100)
            damaged_profile[0, inds] = 0
            decoded1 = autoencoder.predict(damaged_profile).flatten()
            decoded1 = np.delete(decoded1, inds)
            test_profile_copy = np.delete(test_profile, inds)
            pcc = stats.pearsonr(decoded1, test_profile_copy)[0]
            results.append([pcc, inds])
        results.sort(key=lambda x: x[0], reverse=False)
        results = results[:2]
        total_results.extend(results)
    total_results = np.asarray([r[1] for r in total_results]).flatten()
    pickle.dump(total_results, open("total_results_" + key + ".p", "wb"))
    # total_results = pickle.load(open("total_results_" + key + ".p", "rb"))
    c = Counter(total_results)
    for i in range(978):
        importance_scores[cn][i] = c[i] / num
    top_genes_tuples = c.most_common(100)
    top_genes = []
    for x, y in top_genes_tuples:
        top_genes.append(x)
    top_genes = symbols[top_genes]
    final_sets[key] = top_genes
    np.savetxt("figures_data/top_genes_" + key + ".tsv", top_genes, delimiter="\t", fmt="%s")

importance_scores = (importance_scores - np.min(importance_scores)) / (np.max(importance_scores) - np.min(importance_scores))
df = pd.DataFrame.from_records(importance_scores)
rows = []
for cn, cell in enumerate(cell_data.cell_types):
    rows.append(cell)

genes = []
for i in range(input_size):
    genes.append(symbols[i])

df.columns = genes
df.index = rows
df = df.reindex(df.sum().sort_values(ascending=True).index, axis=1)
# df = df.iloc[:, : 50]
# df.drop(df.columns[df.apply(lambda col: col.max() < 1.1)], axis=1, inplace=True)

df.to_csv("figures_data/clustermap.csv")

# common = set(final_sets["MCF7"]) & set(final_sets["PC3"])
# np.savetxt("figures_data/top_genes_MCF7.tsv", list(set(final_sets["MCF7"]) - set(final_sets["PC3"])), delimiter="\t", fmt="%s")
# np.savetxt("figures_data/top_genes_PC3.tsv", list(set(final_sets["PC3"]) - set(final_sets["MCF7"])), delimiter="\t", fmt="%s")