import os
import numpy as np
import math
import pandas as pd


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


def sort_positions(pos_dict):
    for key, value in pos_dict.items():
        value.sort()


def bed_to_gene_expression(filename, gencode, genes_names, genes_names_loc):
    profile = np.zeros(len(genes_names))
    with open(filename) as file:
        for line in file:
            vals = line.split("\t")
            chrn = vals[0]
            pos = int(vals[1]) - 1
            strand = vals[5].strip()
            if chrn + strand not in gencode:
                continue
            arr = gencode[chrn + strand]
            g = find_nearest(arr, pos)
            if abs(g - pos) < 1000:
                gene = genes_names_loc[chrn + "_" + str(g)]
                gene_index = genes_names.index(gene)
                profile[gene_index] = profile[gene_index] + int(vals[4])
    return profile


def check_name(name):
    gene_name_conversion = {'HDGFRP3':'HDGFL3', 'HIST1H2BK':'H2BC12', 'HIST2H2BE':'H2BC21', 'PAPD7':'TENT4A',
                            'NARFL':'CIAO3', 'IKBKAP':'ELP1', 'KIAA0196':'WASHC5', 'TOMM70A':'TOMM70',
                            'FAM63A':'MINDY1', 'HN1L':'JPT2', 'PRUNE':'PRUNE1', 'H2AFV':'H2AZ2', 'TMEM110':'STIMATE',
                            'KIAA1033':'WASHC4', 'ADCK3':'COQ8A', 'LRRC16A':'CARMIL1', 'FAM69A':'DIPK1A', 'WRB':'GET1',
                            'TMEM5':'RXYLT1', 'KIF1BP':'KIFBP', 'TMEM2':'CEMIP2', 'ATP5S':'DMAC2L', 'KIAA0907':'KHDC4',
                            'SQRDL':'SQOR', 'FAM57A':'TLCD3A', 'AARS':'AARS1', 'EPRS':'EPRS1'}
    gene_name_conversion2 = {}
    for key, value in gene_name_conversion.items():
        gene_name_conversion2[value] = key
    if name in gene_name_conversion.keys():
        return gene_name_conversion[name]
    else:
        return None


def load_ctss_folder(ctss_folder, annotation):
    gene_name_conversion = {'HDGFRP3':'HDGFL3', 'HIST1H2BK':'H2BC12', 'HIST2H2BE':'H2BC21', 'PAPD7':'TENT4A',
                            'NARFL':'CIAO3', 'IKBKAP':'ELP1', 'KIAA0196':'WASHC5', 'TOMM70A':'TOMM70',
                            'FAM63A':'MINDY1', 'HN1L':'JPT2', 'PRUNE':'PRUNE1', 'H2AFV':'H2AZ2', 'TMEM110':'STIMATE',
                            'KIAA1033':'WASHC4', 'ADCK3':'COQ8A', 'LRRC16A':'CARMIL1', 'FAM69A':'DIPK1A', 'WRB':'GET1',
                            'TMEM5':'RXYLT1', 'KIF1BP':'KIFBP', 'TMEM2':'CEMIP2', 'ATP5S':'DMAC2L', 'KIAA0907':'KHDC4',
                            'SQRDL':'SQOR', 'FAM57A':'TLCD3A', 'AARS':'AARS1', 'EPRS':'EPRS1'}
    gene_name_conversion2 = {}
    for key, value in gene_name_conversion.items():
        gene_name_conversion2[value] = key
    gencode = {}
    genes_names = []
    genes_names_loc = {}
    treatments = ["Ros", "Ato", "Sim", "Flu"] #, "Ato", "Sim", "Flu"
    l1000 = np.loadtxt("gene_symbols.csv", dtype="str")
    data = {}
    names = {}
    control = {}
    names_control = {}
    with open(annotation) as file:
        for line in file:
            if line.startswith("#"):
                continue
            vals = line.split("\t")
            if not (vals[2] == "gene" or vals[2] == "transcript"):
                continue
            chrn = vals[0]
            start = int(vals[3]) - 1
            strand = vals[6]
            info = dict(item.split("=") for item in vals[8].split(";"))
            # ????????????????????????????????
            gene_name = info["gene_name"].split(".")[0]
            if gene_name in gene_name_conversion2.keys():
                gene_name = gene_name_conversion2[gene_name]
            # ????????????????????????????????
            if gene_name in l1000:
                if gene_name not in genes_names:
                    genes_names.append(gene_name)
                genes_names_loc[chrn + "_" + str(start)] = gene_name
                gencode.setdefault(chrn + strand, []).append(start)
    sort_positions(gencode)
    set(l1000) - set(genes_names)
    for filename in os.listdir(ctss_folder):
        if filename.endswith(".bed"):
            vals = filename.split(".")
            vals2 = vals[1].split("-")
            cell_type = vals2[0].split("_")[-1]
            if cell_type not in data.keys():
                data[cell_type] = {}
                names[cell_type] = {}
                for t in treatments:
                    data[cell_type][t] = []
                    names[cell_type][t] = []
                control[cell_type] = []
                names_control[cell_type] = []
            pert = vals2[1].split("_")[0]
            path = ctss_folder + filename
            if pert in treatments:
                data[cell_type][pert].append(bed_to_gene_expression(path, gencode, genes_names, genes_names_loc))
                names[cell_type][pert].append("T_" + filename)
                print("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t", end="")
            if pert == "Non":
                control[cell_type].append(bed_to_gene_expression(path, gencode, genes_names, genes_names_loc))
                names_control[cell_type].append("C_" + filename)
            print(filename)

    for k in control.keys():
        ndata = np.asarray(control[k]).transpose()
        control[k] = ndata

    for k in data.keys():
        for d in data[k].keys():
            ndata = np.asarray(data[k][d]).transpose()
            header = names[k][d]
            header.extend(names_control[k])
            ndata = np.hstack((ndata, control[k]))
            df = pd.DataFrame(data=ndata, index=genes_names, columns=header)
            df.index.rename("Gene_Symbol", inplace=True)
            df.to_csv("profiles/" + k + "_" + d + ".csv")


# load_ctss_folder("/home/user/Desktop/GSE134817_RAW/", "/home/user/data/to_copy/gencode.v34lift37.annotation.gff3")