import pandas as pd
import urllib.request

df = pd.read_csv("/home/user/data/DeepFake/data/analysisList.tab.txt", sep="\t")
df = df[(df['Genome'] == "hg19")]

tfs = df['TF'].tolist()

for tf in tfs:
    try:
        url = "http://dbarchive.biosciencedbc.jp/kyushu-u/hg19/target/" + tf + ".5.tsv"
        urllib.request.urlretrieve(url, "/home/user/data/DeepFake/TFS/" + tf)
    except Exception as e:
        pass
