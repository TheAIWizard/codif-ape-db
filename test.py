import pandas as pd
from predict_fasttext import predict

db = pd.read_parquet("db-sirene-2023/base_sirene_ape_2023.parquet")

db_auto=db[db["mode_calcul_ape"]=="AUTO"]

db_final=predict(db_auto.head(50), version="v3")[["liasse_numero","apet_finale","apet_1_nace","mode_calcul_ape"]]

print(db_final)

