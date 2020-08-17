import os
from scipy import stats
import deepfake
import utils1
from CellData import CellData
import numpy as np
import random
random.seed(0)
np.random.seed(0)

# parameters
wdir = "sub1/"
test_folds = range(1, 11)
# test_folds = ["antibiotics_ids", "adrenergic_ids", "cholinergic_ids",
#               "5-HT modulator_ids", "TKI_ids", "COX inh._ids",
#               "histaminergic_ids", "antipsychotic_ids", "GABAergic_ids", "dopaminergic_ids"]
# test_folds = ["final_test"]
input_size = 978
latent_dim = 128
data_folder = "/home/user/data/DeepFake/" + wdir


def test_loss(prediction, ground_truth):
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))


os.chdir(data_folder)
print(data_folder)
for test_fold in test_folds:
    test_fold = str(test_fold)
    cell_data = CellData("../LINCS/lincs_phase_1_2.tsv", test_fold)
    autoencoder, cell_decoders = deepfake.get_best_autoencoder(input_size, latent_dim, cell_data, test_fold, 1)
    encoder = autoencoder.get_layer("encoder")
    results = {}
    img_count = 0
    seen_perts = []
    print("Total test objects: " + str(len(cell_data.test_data)))
    all_results = []

    test_trt = "trt_cp"
    for i in range(len(cell_data.test_data)):
        if i % 100 == 0:
            print(str(i) + " - ", end="", flush=True)
        test_meta_object = cell_data.test_meta[i]
        if test_meta_object[2] != test_trt:
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
        test_profile = np.asarray([cell_data.test_data[i]])
        weights = cell_decoders[cell_data.test_meta[i][0]]
        autoencoder.get_layer("decoder").set_weights(weights)
        decoded1 = autoencoder.predict(closest_profile)

        results["count"] = results.get("count", 0) + 1
        results["Our performance is: "] = results.get("Our performance is: ", 0) + test_loss(decoded1, test_profile)

        results["Our correlation is: "] = results.get("Our correlation is: ", 0) + \
                                          stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]

        predictions = []
        for p in all_profiles:
            predictions.append(autoencoder.predict(np.asarray([p])))

        special_decoded = np.mean(np.asarray(predictions), axis=0, keepdims=True)

        results["Our multi-correlation is: "] = results.get("Our multi-correlation is: ", 0) + \
                                                stats.pearsonr(special_decoded.flatten(), test_profile.flatten())[0]

        results["Our multi-performance is: "] = results.get("Our multi-performance is: ", 0) + \
                                                test_loss(special_decoded, test_profile)

        decoded1 = autoencoder.predict(mean_profile)
        results["Our performance is (mean profile): "] = results.get("Our performance is (mean profile): ",
                                                                     0) + test_loss(decoded1, test_profile)

        results["Our correlation (mean profile): "] = results.get("Our correlation (mean profile): ", 0) + \
                                                      stats.pearsonr(decoded1.flatten(), test_profile.flatten())[0]

        results["Baseline correlation (mean profile): "] = results.get("Baseline correlation (mean profile): ", 0) + \
                                                           stats.pearsonr(mean_profile.flatten(),
                                                                          test_profile.flatten())[0]

        results["Baseline performance (mean profile): "] = results.get("Baseline performance (mean profile): ", 0) + \
                                                           test_loss(mean_profile, test_profile)

        all_results.append(str(stats.pearsonr(special_decoded.flatten(), test_profile.flatten())[0]) + ", " +
                           str(stats.pearsonr(mean_profile.flatten(), test_profile.flatten())[0]) + ", "
                           + test_meta_object[0] + ", " + test_meta_object[1] + ", " + str(len(all_profiles)))

        results["closest profile: "] = results.get("closest profile: ", 0) + test_loss(closest_profile, test_profile)
        results["closest profile correlation is: "] = results.get("closest profile correlation is: ", 0) + \
                                                      stats.pearsonr(closest_profile.flatten(), test_profile.flatten())[0]
        bp = stats.pearsonr(mean_profile.flatten(), test_profile.flatten())[0]
        dp = stats.pearsonr(special_decoded.flatten(), test_profile.flatten())[0]
        if bp < 0.1 and dp > 0.6:
            utils1.draw_profiles(test_profile, special_decoded, closest_profile,
                             input_size, "profiles/" + cell_data.test_meta[i][0] + "_" + str(i)
                                 + "_" + str(dp) + "_" + str(bp) + "_" + str(test_meta_object[1]) + ".png")
            latent_vectors_1 = encoder.predict(closest_profile)
            utils1.draw_vectors(latent_vectors_1, "vectors/" + str(i) + ".png")

    print(" Done")
    with open("log.txt", 'a+') as f:
        for key, value in results.items():
            if key == "count":
                continue
            f.write(key + str(value / results["count"]))
            f.write("\n")

    performance = str(results["Our performance is: "] / results["count"]) + "\t" + \
                  str(results["Our correlation is: "] / results["count"]) + "\t" + \
                  str(results["Our multi-performance is: "] / results["count"]) + "\t" + \
                  str(results["Our multi-correlation is: "] / results["count"]) + "\t" + \
                  str(results["closest profile: "] / results["count"]) + "\t" + \
                  str(results["closest profile correlation is: "] / results["count"]) + "\t" + \
                  str(results["Baseline correlation (mean profile): "] / results["count"]) + "\t" + \
                  str(results["Baseline performance (mean profile): "] / results["count"])

    with open("final_result.tsv", 'a+') as f:
        f.write(performance)
        f.write("\n")

    with open("all_results", 'a+') as f:
        f.write("\n".join(all_results))
        f.write("\n")
