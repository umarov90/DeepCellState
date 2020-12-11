import math
import pandas as pd
import os
import pickle
import numpy as np
from scipy import stats
from tensorflow import keras
from figures import utils1
from scipy.stats import ttest_ind
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def read_profile(file):
    df = pd.read_csv(file, sep=",")
    profiles_trt = []
    profiles_ctrl = []
    trt_names = []
    for i in range(1, len(df.columns)):
        profile = []
        for g in genes:
            if len(df[(df['Gene_Symbol'] == g)]) != 0:
                profile.append(df[(df['Gene_Symbol'] == g)][df.columns[i]].tolist()[0])
            else:
                profile.append(0)
        profile = np.asarray(profile)
        profile = profile + 2
        profile = (1000000 * profile) / np.sum(profile)
        profile_name = df.columns[i]
        if profile_name.startswith(
                "T") and "-6-" not in profile_name:  # in trt df[(df['Gene_Symbol'] == "HMGCR")][df.columns[i]]
            profiles_trt.append(profile)
            trt_names.append(profile_name)
        elif profile_name.startswith("C"):
            profiles_ctrl.append(profile)
    trt_profile = np.mean(np.asarray(profiles_trt), axis=0)
    ctrl_profile = np.mean(np.asarray(profiles_ctrl), axis=0)
    for p in profiles_trt:
        incor = stats.pearsonr(profiles_trt[0].flatten(), p.flatten())[0]
        if incor < 0.5:
            print("Err")
    utils1.draw_one_profiles([trt_profile], len(genes), file + "trt_profiles.png")
    utils1.draw_one_profiles([ctrl_profile], len(genes), file + "ctrl_profiles.png")
    profile = np.zeros(trt_profile.shape)
    for i in range(len(genes)):
        if ctrl_profile[i] != 0 and trt_profile[i] != 0:
            try:
                profile[i] = math.log(trt_profile[i] / ctrl_profile[i])
            except Exception as e:
                print(e)
    profile = profile / np.max(np.abs(profile))
    profile = np.expand_dims(profile, axis=-1)
    return profile


os.chdir(open("../data_dir").read().strip())
genes = np.loadtxt("data/gene_symbols.csv", dtype="str")
input_data = np.asarray([read_profile("../data/statins/H_Flu.csv"), read_profile("../data/statins/H_Ato.csv"),
                         read_profile("../data/statins/H_Ros.csv"), read_profile("../data/statins/H_Sim.csv")])
output_data = np.asarray([read_profile("../data/statins/M_Flu.csv"), read_profile("../data/statins/M_Ato.csv"),
                          read_profile("../data/statins/M_Ros.csv"), read_profile("../data/statins/M_Sim.csv")])

treatments = ["Fluvastatin", "Atrovastatin", "Rosuvastatin", "Simvastatin"]
total_corr_base = 0
total_corr_our = 0

for i in range(len(input_data)):
    output_data[i][np.isnan(output_data[i])] = 0
    input_data[i][np.isnan(input_data[i])] = 0

baseline_corrs = []
our_corrs = []
for i in range(len(input_data)):
    print(str(treatments[i]), end="\t")
    test_index = i
    df_hepg2 = input_data[test_index]
    df_mcf7 = output_data[test_index]
    baseline_corr = stats.pearsonr(df_hepg2.flatten(), df_mcf7.flatten())[0]
    print(str(baseline_corr), end="\t")
    baseline_corrs.append(baseline_corr)
    total_corr_base = total_corr_base + baseline_corr
    autoencoder = keras.models.load_model("best_autoencoder_ext_val/main_model")
    cell_decoders = {"MCF7": pickle.load(open("best_autoencoder_ext_val/" + "MCF7" + "_decoder_weights", "rb")),
                     "PC3": pickle.load(open("best_autoencoder_ext_val/" + "PC3" + "_decoder_weights", "rb"))}
    autoencoder.get_layer("decoder").set_weights(cell_decoders["MCF7"])
    decoded = autoencoder.predict(np.asarray([df_hepg2]))
    decoded = decoded.flatten()
    corr = stats.pearsonr(decoded, df_mcf7.flatten())[0]
    print(corr)
    our_corrs.append(corr)
    total_corr_our = total_corr_our + corr

print(str(total_corr_base / len(treatments)) + "\t" + str(total_corr_our / len(treatments)))
t, p = ttest_ind(baseline_corrs, our_corrs)
print("DeepCellState p: " + str(p))
print("Improvement: " + str(total_corr_our / total_corr_base))
