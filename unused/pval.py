import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import ttest_ind

os.chdir(open("../data_dir").read().strip())
sns.set(font_scale=1.3, style='ticks')

df = pd.read_csv("pvals/simple.csv", sep="\t")
df2 = pd.read_csv("pvals/orig.csv", sep="\t")
t, p = ttest_ind(df.to_numpy().flatten(), df2.to_numpy().flatten())
print("p value: " + str(p))