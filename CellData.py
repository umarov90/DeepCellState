import numpy as np
import random
from random import randint
from random import shuffle
import pandas as pd
random.seed(0)
np.random.seed(0)

class CellData:
    def __init__(self, data_file, test_fold):
        data, meta, all_pert_ids = self.parse_data(data_file)
        train_data, train_meta, test_data, test_meta, val_data, \
        val_meta, cell_types, train_perts, val_perts, test_perts = \
            self.split_data(data, meta, all_pert_ids, test_fold)
        meta_dictionary_pert = {}
        for pert_id in train_perts:
            meta_dictionary_pert[pert_id] = [[p, i] for i, p in enumerate(train_meta) if p[1] == pert_id]
        meta_dictionary_pert_test = {}
        for pert_id in test_perts:
            meta_dictionary_pert_test[pert_id] = [[p, i] for i, p in enumerate(test_meta) if p[1] == pert_id]
        meta_dictionary_pert_val = {}
        for pert_id in val_perts:
            meta_dictionary_pert_val[pert_id] = [[p, i] for i, p in enumerate(val_meta) if p[1] == pert_id]

        self.test_perts = test_perts
        self.train_data = train_data
        self.test_data = test_data
        self.train_meta = train_meta
        self.test_meta = test_meta
        self.cell_types = cell_types
        self.all_pert_ids = all_pert_ids
        self.val_data = val_data
        self.val_meta = val_meta
        self.meta_dictionary_pert = meta_dictionary_pert
        self.meta_dictionary_pert_test = meta_dictionary_pert_test
        self.meta_dictionary_pert_val = meta_dictionary_pert_val

        print("----------------------------------------------")
        print(train_data.shape)
        print(test_data.shape)
        print("----------------------------------------------")

    def parse_data(self, file):
        print("Parsing data at " + file)
        df = pd.read_csv(file, sep="\t")
        df.reset_index(drop=True, inplace=True)
        print("Total: " + str(df.shape))
        df = df[(df['cell_id'] == "MCF7") | (df['cell_id'] == "PC3")]
        print(df.groupby(['cell_id']).size())
        # df = df[(df['pert_type'] == "trt_cp") | (df['pert_type'] == "trt_sh") |
        #         (df['pert_type'] == "trt_sh.cgs") |
        #         (df['pert_type'] == "trt_oe") | (df['pert_type'] == "trt_lig")]
        # df = df[(df['cell_id'] == "MCF7") | (df['cell_id'] == "PC3") | (df['cell_id'] == "A375") |
        #         (df['cell_id'] == "HT29") | (df['cell_id'] == "HA1E") | (df['cell_id'] == "YAPC") |
        #         (df['cell_id'] == "HELA")]
        # df = df[(df['pert_type'] == "trt_sh") | (df['pert_type'] == "trt_sh.cgs") | (df['pert_type'] == "trt_cp")]
        df = df[(df['pert_type'] == "trt_cp")]
        print("Cell filtering: " + str(df.shape))
        df = df.groupby(['cell_id', 'pert_id']).filter(lambda x: len(x) > 1)
        print("Pert filtering: " + str(df.shape))
        df = df.groupby(['cell_id', 'pert_id', 'pert_type'], as_index=False).mean()
        print("Merging: " + str(df.shape))
        df = df.groupby(['cell_id']).filter(lambda x: len(x) > 1000)
        print("Count filtering: " + str(df.shape))
        # df = df.drop_duplicates(['cell_id', 'pert_id', 'pert_idose', 'pert_itime', 'pert_type'])
        # df = df.groupby(['cell_id', 'pert_id'], as_index=False).mean()
        # print("Merging: " + str(df.shape))
        # df.pert_type.value_counts().to_csv("trt_count_final.tsv", sep='\t')
        print(df.groupby(['cell_id']).size())
        cell_ids = df["cell_id"].values
        pert_ids = df["pert_id"].values
        all_pert_ids = set(pert_ids)
        # pert_idose = df["pert_idose"].values
        # pert_itime = df["pert_itime"].values
        pert_type = df["pert_type"].values
        perts = np.stack([cell_ids, pert_ids, pert_type]).transpose()
        df = df.drop(['cell_id', 'pert_id', 'pert_type', 'Unnamed: 0'], 1)
        data = df.values
        # data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # for i in range(len(data)):
        data = data / max(np.max(data), abs(np.min(data)))
        # data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data))  - 1
        data = np.expand_dims(data, axis=-1)
        return data, perts, all_pert_ids

    def split_data(self, data, meta, all_pert_ids, test_fold):
        print(test_fold)
        cell_types = set([meta[i][0] for i, x in enumerate(meta)])
        rng_state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(rng_state)
        np.random.shuffle(meta)
        all_pert_ids_list = list(all_pert_ids)
        shuffle(all_pert_ids_list)

        test_perts = np.loadtxt("../LINCS/folds/" + str(test_fold), dtype='str')# _sh+cp
        z = list(all_pert_ids - set(test_perts))
        shuffle(z)
        train_perts = z[:int(0.95 * len(z))]
        val_perts = z[int(0.95 * len(z)):]

        train_data = np.asarray(
            [data[i] for i, m in enumerate(meta) if m[1] in train_perts])  # and m[0] != "A375"
        test_data = np.asarray([data[i] for i, m in enumerate(meta) if m[1] in test_perts])
        val_data = np.asarray([data[i] for i, m in enumerate(meta) if m[1] in val_perts])
        train_meta = np.asarray(
            [m for i, m in enumerate(meta) if m[1] in train_perts])  # and m[0] != "A375"
        test_meta = np.asarray([m for i, m in enumerate(meta) if m[1] in test_perts])
        val_meta = np.asarray([m for i, m in enumerate(meta) if m[1] in val_perts])

        return train_data, train_meta, test_data, test_meta, val_data, val_meta, cell_types, train_perts, val_perts, test_perts

    def get_profile(self, data, meta_data, test_pert, train_data=False):
        # if train_data:
        pert_list = [p[1] for p in meta_data if p[0][0] != test_pert[0]]
        # else:
        #     pert_list = [p[1] for p in meta_data if
        #                  p[0][0] != test_pert[0] and p[0][0] == "A375"]
        if len(pert_list) > 0:
            random_best = randint(0, len(pert_list) - 1)
            mean_profile = np.mean(np.asarray(data[pert_list]), axis=0, keepdims=True)
            return random_best, np.asarray([data[pert_list[random_best]]]), mean_profile, data[pert_list]
        else:
            return -1, None, None, None

    def get_profile_cell(self, data, meta_data, cell):
        pert_list = [p[1] for p in meta_data if p[0][0] == cell]
        if len(pert_list) > 0:
            return data[pert_list]
        else:
            return None