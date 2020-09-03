import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
matplotlib.use("Agg")

data = pd.read_csv("tr_sizes.csv") #, dtype={'Training set size': 'str'}
fig, ax = plt.subplots()
sns.barplot(x="Training set size", y="PCC", data=data, color="salmon", saturation=0.5, linewidth=1, edgecolor="0.2", ax=ax)
#sns.lineplot(x="Training set size", y="PCC", data=data, markers=True, dashes=False, color="salmon", ax=ax)
plt.savefig("training_size.png")