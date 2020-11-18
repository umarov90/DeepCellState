import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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


def test_loss(prediction, ground_truth):
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))


input_size = 978
latent_dim = 128
data_folder = "/home/user/data/DeepFake/sub2/"
os.chdir(data_folder)

# cell_data = CellData("../LINCS/lincs_phase_1_2.tsv", "1", 10)
# pickle.dump(cell_data, open("cell_data.p", "wb"))
cell_data = pickle.load(open("cell_data.p", "rb"))

autoencoder = keras.models.load_model("best_autoencoder_1/main_model/")
cell_decoders = {}
for cell in cell_data.cell_types:
    cell_decoders[cell] = pickle.load(open("best_autoencoder_1/" + cell + "_decoder_weights", "rb"))
encoder = autoencoder.get_layer("encoder")
decoder = autoencoder.get_layer("decoder")

symbols = np.loadtxt("../gene_symbols.csv", dtype="str")


# mat = []
# for p in cell_data.train_data:
#     v = encoder.predict(np.asarray([p]))
#     mat.append(v)
#
# mat = np.asarray(mat)
# max_vec = np.squeeze(np.max(mat, axis=0))
# min_vec = np.squeeze(np.min(mat, axis=0))
# pickle.dump(max_vec, open("max_vec.p", "wb"))
# pickle.dump(min_vec, open("min_vec.p", "wb"))

# max_vec = pickle.load(open("max_vec.p", "rb"))
# min_vec = pickle.load(open("min_vec.p", "rb"))
#
# tf_data = pickle.load(open("tf_data.p", "rb"))
# Hep_G2
# tf_data = {"MCF7": {}, "PC3": {}}
# directory = "/home/user/data/DeepFake/TFS"
# cell_names = set()
# for filename in os.listdir(directory):
#     try:
#         top_genes = {}
#         for cell_type in ["MCF-7", "PC-3"]:
#             df = pd.read_csv(os.path.join(directory, filename), sep="\t", index_col="Target_genes")
#             to_drop = []
#             to_merge = []
#             for col in df.columns:
#                 name = col.split("|")[-1]
#                 cell_names.add(name)
#             for col in df.columns:
#                 if cell_type not in col:
#                     to_drop.append(col)
#                 else:
#                     to_merge.append(col)
#             if len(to_merge) == 0:
#                 continue
#             df = df.drop(to_drop, 1)
#             df[cell_type] = df[to_merge].astype(float).mean(axis=1)
#             df = df.drop(to_merge, 1)
#             df = df[df.index.isin(symbols)]
#             df = df.sort_values(cell_type, ascending=False).head(50)
#             # df = df[(df.T != 0).any()]
#             top_genes[cell_type] = df.index.to_list()
#             # if len(top_genes) != 10:
#             #     continue
#         if len(top_genes.keys()) > 1:
#             for cell_type in ["MCF-7", "PC-3"]:
#                 tf_data[cell_type.replace("-", "")][filename] = top_genes[cell_type]
#     except Exception as e:
#         print(e)
# pickle.dump(tf_data, open("tf_data.p", "wb"))
final_sets = {}
importance_scores = zeros((len(cell_data.cell_types), input_size))
for cn, key in enumerate(cell_data.cell_types):  # ["PC3", "MCF7"]:
    print(key + "________________________________________________")
    autoencoder.get_layer("decoder").set_weights(cell_decoders[key])
    total_results = []
    seen_perts = []
    num = 0
    for i in range(len(cell_data.test_data)):
        if i % 100 == 0:
            print(str(i) + " - ", end="", flush=True)
        test_meta_object = cell_data.test_meta[i]
        if test_meta_object[0] != key:
            continue
        closest, closest_profile, mean_profile, all_profiles = cell_data.get_profile(cell_data.test_data,
                                                                                     cell_data.meta_dictionary_pert_test[
                                                                                         test_meta_object[1]],
                                                                                     test_meta_object)
        if closest_profile is None:
            continue
        if test_meta_object[1] in seen_perts:
            continue
        seen_perts.append(test_meta_object[1])
        num = num + 1
        test_profile = np.asarray([cell_data.test_data[i]])
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
        # results = []
        # for k in range(100):
        #     damaged_profile = closest_profile.copy()
        #     inds = random.sample(range(0, 978), 100)
        #     damaged_profile[0, inds] = 0
        #     decoded1 = autoencoder.predict(damaged_profile)
        #     pcc = stats.pearsonr(decoded1[0, inds].flatten(), test_profile[0, inds].flatten())[0]
        #     results.append([pcc, inds])
        # results.sort(key=lambda x: x[0], reverse=True)
        # results = results[:10]
        # total_results.extend(results)
    # total_results = np.asarray([r[1] for r in total_results]).flatten()
    # pickle.dump(total_results, open("total_results_" + key + ".p", "wb"))
    total_results = pickle.load(open("total_results_" + key + ".p", "rb"))
    c = Counter(total_results)
    for i in range(978):
        importance_scores[cn][i] = c[i] / num
    top_genes_tuples = c.most_common(100)
    top_genes = []
    for x, y in top_genes_tuples:
        top_genes.append(x)
    top_genes = symbols[top_genes]
    final_sets[key] = top_genes
    np.savetxt("top_genes_" + key + ".tsv", top_genes, delimiter="\t", fmt="%s")

df = pd.DataFrame.from_records(importance_scores)
rows = []
for cn, cell in enumerate(cell_data.cell_types):
    rows.append(cell)

genes = []
for i in range(input_size):
    genes.append(symbols[i])

df.columns = genes
df.index = rows
# df = df.reindex(df.sum().sort_values().index, axis=1)
df.drop(df.columns[df.apply(lambda col: col.max() < 2.45)], axis=1, inplace=True)

df.to_csv("clustermap.csv")

for key, value in final_sets.items():
    a = set([])
    # for key2, value2 in final_sets.items():
    #     if key == key2:
    #         continue
    #     if len(a) == 0:
    #         a = set(value2)
    #     else:
    #         a = a | set(value2)
    b = set(value) - a
    # np.savetxt("top_genes_" + key + ".tsv", np.array(list(b)), delimiter="\t", fmt="%s")