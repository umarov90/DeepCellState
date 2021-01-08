import argparse
import os
from scipy import stats
import deepfake
from figures import profiles_viz
from CellData import CellData
import numpy as np
import pandas as pd
import random

def get_options():
    parser = argparse.ArgumentParser(description='Version: 1.0')
    parser.add_argument('-O', metavar='output', default="DeepCellState_output",
                        help='Output directory')
    parser.add_argument('-CT', metavar='cell types', default="",
                        type=str, help='Comma separated list of cell types to use in addition to MCF7 and PC3')
    parser.add_argument('-PT', metavar='pert type', default="trt_cp",
                        type=str, help='Perturbation type to be used, defaults to trt_cp')
    parser.add_argument('-N', metavar='number of runs', default=2,
                        type=int, help='Number of models trained for each fold.'
                                       ' The model with best validation performance is picked.')
    parser.add_argument('-SM', metavar='special models', default=0,
                        type=int, help='Set to 1 to train drug MoA family models or'
                                       ' set to 2 to train external validation model.'
                                       ' Defaults to 0, i.e. 10-fold cross-validation.')

    args = parser.parse_args()

    return args


def test_loss(prediction, ground_truth):
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))


def main():
    random.seed(0)
    np.random.seed(0)
    args = get_options()
    regul_stren = 2
    if args.CT is not None and len(args.CT)>0:
        regul_stren = 1
    folds_folder = "../data/folds/"
    if args.PT == "trt_sh":
        folds_folder = "../data/folds_sh+cp/"
    if args.SM == 0:
        test_folds = range(1, 11)
    elif args.SM == 1:
        test_folds = ["antibiotics_ids", "adrenergic_ids", "cholinergic_ids",
                      "5-HT modulator_ids", "TKI_ids", "COX inh._ids",
                      "histaminergic_ids", "antipsychotic_ids", "GABAergic_ids", "dopaminergic_ids"]
    else:
        test_folds = ["ext_val"]
        regul_stren = 3
    input_size = 978
    latent_dim = 128

    wdir = open("data_dir").read().strip() + args.O
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    os.chdir(wdir)

    df = pd.read_csv("../data/GSE70138_Broad_LINCS_pert_info.txt", sep="\t")
    good = []
    tsne_perts = []
    tsne_input = []
    tsne_latent = []
    for r, test_fold in enumerate(test_folds):
        test_fold = str(test_fold)
        cell_data = CellData("../data/lincs_phase_1_2.tsv", folds_folder + test_fold, "MCF7,PC3," + args.CT, args.PT)
        autoencoder, cell_decoders = deepfake.get_best_autoencoder(input_size, latent_dim,
                                                                   cell_data, test_fold, args.N, regul_stren)
        encoder = autoencoder.get_layer("encoder")
        results = {}
        img_count = 0
        seen_perts = []
        print("Total test objects: " + str(len(cell_data.test_data)))
        all_results = []
        good_perts = []
        test_trt = "trt_cp"
        vectors = []
        input_profiles = []
        perts_order = []
        for i in range(len(cell_data.test_data)):
            if i % 100 == 0:
                print(str(i) + " - ", end="", flush=True)
            test_meta_object = cell_data.test_meta[i]
            if test_meta_object[2] != test_trt:
                continue
            if test_meta_object[0] not in ["MCF7", "PC3"]:
                continue
            closest, closest_profile, mean_profile, all_profiles = cell_data.get_profile(cell_data.test_data,
                                                                                         cell_data.meta_dictionary_pert_test[
                                                                                             test_meta_object[1]],
                                                                                         test_meta_object)
            if closest_profile is None:
                continue
            # if test_meta_object[1] in seen_perts:
            #     continue
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
                                                          stats.pearsonr(closest_profile.flatten(), test_profile.flatten())[
                                                              0]
            bp = stats.pearsonr(mean_profile.flatten(), test_profile.flatten())[0]
            dp = stats.pearsonr(special_decoded.flatten(), test_profile.flatten())[0]
            if dp > 0.4: # and bp < 0.5
                os.makedirs("profiles", exist_ok=True)
                pname = profiles_viz.fix(df.query('pert_id=="' + str(test_meta_object[1]) + '"')["pert_iname"].tolist()[0])
                profiles_viz.draw_profiles(test_profile, special_decoded, closest_profile, pname,
                                     input_size, "profiles/" + cell_data.test_meta[i][0] + "_" + str(i)
                                     + "_" + str(dp) + "_" + str(bp) + "_" + pname + ".svg")
                profiles_viz.draw_scatter_profiles(test_profile, special_decoded, closest_profile, pname,
                                  "profiles/" + cell_data.test_meta[i][0] + "_" + str(i)
                                             + "_" + str(dp) + "_" + str(bp) + "_" +
                                             pname + "_scatter.svg")
            tsne_perts.append(["PC3" if test_meta_object[0] == "MCF7" else "MCF7",
                               df.query('pert_id=="' + str(test_meta_object[1]) + '"')["pert_iname"].tolist()[0]])
            tsne_input.append(closest_profile.flatten())
            tsne_latent.append(encoder.predict(closest_profile).flatten())
            if test_meta_object[0] == "MCF7":
                good_perts.append([test_meta_object[1], bp])
        np.savetxt("../figures_data/tsne_perts.csv", np.array(tsne_perts), delimiter=',', fmt="%s")
        np.savetxt("../figures_data/tsne_input.csv", np.array(tsne_input), delimiter=',')
        np.savetxt("../figures_data/tsne_latent.csv", np.array(tsne_latent), delimiter=',')
        good_perts.sort(key=lambda x: x[1], reverse=True)
        matrix = np.zeros((len(good_perts), len(good_perts)))
        for i in range(len(good_perts)):
            for j in range(len(good_perts)):
                a = cell_data.get_profile_cell_pert(cell_data.test_data, cell_data.test_meta, "MCF7",
                                                    good_perts[i][0])
                b = cell_data.get_profile_cell_pert(cell_data.test_data, cell_data.test_meta, "PC3",
                                                    good_perts[j][0])
                if a is None or b is None:
                    continue
                vector1 = encoder.predict(np.asarray(a))
                vector2 = encoder.predict(np.asarray(b))
                vpcc = stats.pearsonr(vector1.flatten(), vector2.flatten())[0]
                matrix[i][j] = vpcc
        for i in range(len(good_perts)):
            good_perts[i] = df.query('pert_id=="'+str(good_perts[i][0]) + '"')["pert_iname"].tolist()[0]
        df1 = pd.DataFrame(data=matrix, index=good_perts, columns=good_perts)
        df1.to_pickle("../figures_data/latent.p")

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
            f.write(str(latent_dim) + "\t" + performance) # str(tr_size) + "\t" +
            f.write("\n")

        with open("all_results", 'a+') as f:
            f.write("\n".join(all_results))
            f.write("\n")


if __name__ == '__main__':
    main()
