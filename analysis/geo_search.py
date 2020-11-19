import pandas as pd

df = pd.read_csv("/home/user/Desktop/single_drug_perturbations-p1.0.csv")
seen_drugs = []
drug_cells = {}
geo_ids = {}
for index, row in df.iterrows():
    if row["organism"] == "human":
        drug_cells.setdefault(str(row["drug_name"]), set()).add(row["cell_type"])
        geo_ids.setdefault(str(row["drug_name"]) + ":" + str(row["cell_type"]), []).append(row["geo_id"])

for key, value in drug_cells.items():
    if len(value) > 1:
        print(key + ": ", end="")
        for v in value:
            geo = ",".join(geo_ids[key + ":" + str(v)])
            print(str(v) + " [" + geo + "]; ", end="")
        print("")