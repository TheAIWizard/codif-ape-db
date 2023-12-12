import fasttext

list_libs = ["prestat servic webmarketing redact web referenc web community manag AUTO_X NAT_SICORE_NaN SURF_NaN EVT_SICORE_01P"]

model_input = {
    "query": list_libs,
    "k": 5
}

model_name = "FastText-APE"
version = "v6"

model = fasttext.load_model(
        f"/home/onyxia/work/codif-ape-db/{model_name}/{version}/default.bin"
    )

predictions = model.predict(model_input["query"], model_input["k"])

print(predictions)

# import pandas as pd
# db = pd.read_parquet("db-sirene-2023/base_sirene_ape_2023.parquet")
# print(db[db["liasse_numero"] == "J00025076621"]["libelle_activite_apet"].iloc[0])