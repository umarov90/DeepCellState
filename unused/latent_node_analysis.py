import os
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

from CellData import CellData


def test_loss(prediction, ground_truth):
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))


input_size = 978
latent_dim = 128
data_folder = "/home/user/data/DeepFake/sub_complete/"
os.chdir(data_folder)
data = CellData("../LINCS/lincs_phase_1_2.tsv", "1")

autoencoder = keras.models.load_model("best_autoencoder_1/main_model/")
cell_decoders = {}
for cell in data.cell_types:
    cell_decoders[cell] = pickle.load(open("best_autoencoder_1/" + cell + "_decoder_weights", "rb"))
encoder = autoencoder.get_layer("encoder")
decoder = autoencoder.get_layer("decoder")

######################################################################################################
# Latent vector output for analysis
######################################################################################################
# inds = [i for i, p in enumerate(train_meta) if p[2] == "trt_sh"]
# latent_vectors = encoder.predict(train_data[inds])
# np.savetxt("sh", latent_vectors, delimiter=',')

symbols = np.loadtxt("../gene_symbols.csv", dtype="str")
# print("ALL cells________________________________________________")
# assoc = {}
# smallest_cor = 1.0
# for i in range(input_size):
#     print(i, end=" - ")
#     data_slice = deepcopy(train_data[:1000])
#     latent_vectors_1 = encoder.predict(data_slice)
#     for j in range(len(data_slice)):
#         data_slice[j][i] = -1 * data_slice[j][i]
#     latent_vectors_2 = encoder.predict(data_slice)
#     for j in range(latent_dim):
#         corr = stats.pearsonr(latent_vectors_1[:, j], latent_vectors_2[:, j])[0]
#         if corr < smallest_cor:
#             smallest_cor = corr
#         if corr < 0.99:
#             assoc.setdefault(j, []).append(symbols[i])
#
# for key in sorted(assoc):
#     print(key, end="\t")
#     for v in assoc[key]:
#         print(v, end="\t")
#     print()
pert_ids = list(data.all_pert_ids)
# importance_scores = pickle.load(open("importance_scores.p", "rb"))
# importance_list = pickle.load(open("importance_list.p", "rb"))
importance_scores = zeros((len(data.cell_types), input_size))
importance_list = {}
for cn, cell in enumerate(data.cell_types):  # ["PC3", "MCF7"]:
    print(cell + "________________________________________________")
    autoencoder.get_layer("decoder").set_weights(cell_decoders[cell])
    best_cor = 1.0
    for k in range(input_size):
        print(k, end=" - ")
        pcc = 0
        count = 0
        for pert in data.test_perts:
            all_profiles_a = data.get_profile_cell(data.test_data,
                                                   data.meta_dictionary_pert_test[pert],
                                                   "A375")
            all_profiles_b = data.get_profile_cell(data.test_data,
                                                   data.meta_dictionary_pert_test[pert],
                                                   cell)

            if all_profiles_a is None or all_profiles_b is None:
                continue

            decoded1 = autoencoder.predict(all_profiles_a)
            all_profiles_a[0][k] = 0
            decoded2 = autoencoder.predict(all_profiles_a)
            pcc = pcc + (stats.pearsonr(decoded1.flatten(), all_profiles_b.flatten())[0]
                         - stats.pearsonr(decoded2.flatten(), all_profiles_b.flatten())[0])
            count = count + 1
            if count > 100:
                break

        pcc = pcc / count
        importance_list.setdefault(cell, []).append([symbols[k], pcc])
        importance_scores[cn][k] = pcc

pickle.dump(importance_list, open("importance_list.p", "wb"))
pickle.dump(importance_scores, open("importance_scores.p", "wb"))

df = pd.DataFrame.from_records(importance_scores)
rows = []
for cn, cell in enumerate(data.cell_types):
    rows.append(cell)

genes = []
for i in range(input_size):
    genes.append(symbols[i])

df.columns = genes
df.index = rows

df.drop(df.columns[df.apply(lambda col: col.max() < 0.001)], axis=1, inplace=True)

df.to_csv("clustermap.csv")


for key, value in importance_list.items():
    value = sorted(value, key=lambda x: x[1], reverse=True)
    positive = [p[0] for p in value if p[1] > 0.0]
    importance_list[key] = positive

for key, value in importance_list.items():
    a = set([])
    # for key2, value2 in importance_list.items():
    #     if key == key2:
    #         continue
    #     if len(a) == 0:
    #         a = set(value2)
    #     else:
    #         a = a | set(value2)
    b = set(value) - a
    with open("top_genes/" + key + ".csv", 'w+') as f:
        for v in b:
            f.write(v)
            f.write("\n")
to_delete = []

for i in range(input_size):
    if np.max(importance_scores[:, i]) < 0.001:
        to_delete.append(i)
importance_scores = np.delete(importance_scores, to_delete, 1)
genes = [i for j, i in enumerate(genes) if j not in set(to_delete)]
np.savetxt('cells_genes_heat.csv', importance_scores, delimiter=',')
with open("cells_genes_meta.csv", 'a+') as f:
    for i in range(len(genes)):
        f.write(genes[i])
        f.write("\n")
    f.write("\n")
    for cn, cell in enumerate(data.cell_types):
        f.write(cell)
        f.write("\n")
# for ln in range(128):
#     print(str(ln) + "________________________________________________")
#     importance_list = zeros(input_size)
#     count = 0
#     for i in range(len(data.test_data)):
#         test_meta_object = data.test_meta[i]
#         autoencoder.get_layer("decoder").set_weights(cell_decoders[test_meta_object[0]])
#         test_profile = np.asarray([data.test_data[i]])
#         closest, closest_profile, mean_profile, all_profiles = data.get_profile(data.test_data,
#                                                                                 data.meta_dictionary_pert_test[
#                                                                                     test_meta_object[1]],
#                                                                                 test_meta_object)
#         if closest_profile is None:
#             continue
#         latent_vector_1 = encoder.predict(test_profile)
#         reconstruction1 = decoder.predict(latent_vector_1).flatten()
#         latent_vector_1[0][ln] = 0
#         reconstruction2 = decoder.predict(latent_vector_1).flatten()
#         test_profile = test_profile.flatten()
#         for k in range(input_size):
#             importance_list[k] = importance_list[k] + \
#                                  abs(test_profile[k] - reconstruction2[k]) - abs(test_profile[k] - reconstruction1[k])
#         count = count + 1
#         if count > 9:
#             break
#     importance_list = importance_list / count
#     # importance_list = sorted(importance_list, key=lambda x: x[1], reverse=True)
#     print()
#
#     with open("latent_genes_heat.tsv", 'a+') as f:
#         for i in range(len(importance_list)):
#             f.write(str(importance_list[i]) + "\t")  # str(importance_list[i][0]) +
#         f.write("\n")


# for cell in data.cell_types:
#     print(cell + "________________________________________________")
#     autoencoder.get_layer("decoder").set_weights(cell_decoders[cell])
#     assoc = {}
#     smallest_cor = 1.0
#     for i in range(latent_dim):
#         print(i, end=" - ")
#         data_slice = np.asarray(deepcopy([data.train_data[i] for i, p in enumerate(data.train_meta) if p[0] == cell][:1000]))
#         latent_vectors_1 = encoder.predict(data_slice)
#         reconstruction1 = decoder.predict(latent_vectors_1)
#         for j in range(len(latent_vectors_1)):
#             latent_vectors_1[j][i] = -1 * latent_vectors_1[j][i]
#         reconstruction2 = decoder.predict(latent_vectors_1)
#         for j in range(input_size):
#             corr = stats.pearsonr(reconstruction1[:, j].flatten(), reconstruction2[:, j].flatten())[0]
#             if corr < smallest_cor:
#                 smallest_cor = corr
#             if corr < 0.99:
#                 assoc.setdefault(i, []).append(symbols[j])
#
#     print()
#     print(smallest_cor)
#
#     for key in sorted(assoc):
#         print(key, end="\t")
#         for v in assoc[key]:
#             print(v, end="\t")
#         print()
