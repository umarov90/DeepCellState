import os
from scipy import stats
import numpy as np

os.chdir(open("data_dir").read().strip())
# pc3 = np.genfromtxt('competitor_data/PC3_ts.csv', delimiter=',')
mcf7_pred = np.genfromtxt('competitor_data/MCF7_ts_pred.csv', delimiter=',')
mcf7 = np.genfromtxt('competitor_data/MCF7_ts.csv', delimiter=',')
all_results = []
for i in range(len(mcf7)):
    all_results.append(str(stats.pearsonr(mcf7_pred[i].flatten(), mcf7[i].flatten())[0]))


with open("scgen_results.csv", 'w+') as f:
    f.write("\n".join(all_results))
    f.write("\n")
