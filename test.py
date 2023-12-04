import pandas as pd
from predict_fasttext import predict

db = pd.read_parquet("db-sirene-2023/base_sirene_ape_2023.parquet")

db_auto=db[db["mode_calcul_ape"]=="AUTO"]

# Définir la plage de dates à tester (avril à juin 26)
date_debut = pd.Timestamp('2023-06-27 18:20:43', tz='UTC')
date_fin = pd.Timestamp('2023-03-01', tz='UTC')

# Filtrer les lignes où la date_modification est comprise entre avril et le 26 juin
db_auto_periode = db_auto[(db_auto['date_modification'] >= date_debut)] #& (db_auto['date_modification'] <= date_fin)]

db_final=predict(db_auto_periode.head(5000), version="v6")[["liasse_numero","apet_finale","apet_1_nace","mode_calcul_ape","date_modification"]]

print(db_final)

# Sélectionner les lignes où les valeurs de 'apet_finale' et 'apet_1_nace' diffèrent
result = db_final[db_final['apet_finale'] != db_final['apet_1_nace']]

# Afficher le résultat avec les dates correspondantes
print(result[['liasse_numero', 'apet_finale', 'apet_1_nace', 'mode_calcul_ape', 'date_modification']])

# Assurez-vous que la colonne date_modification est de type datetime
result['date_modification'] = pd.to_datetime(result['date_modification'])

# Extraire les mois de la colonne date_modification
mois_uniques = result['date_modification'].dt.month.unique()

# Afficher l'ensemble des mois uniques
print(mois_uniques)
