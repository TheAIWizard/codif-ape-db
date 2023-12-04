# concatenate categorical variables as : cleaned_lib AUTO_{liasse_type}  NAT_SICORE_{activ_nat_et} SURF_{‍activ_surf_et} EVT_SICORE_{evenement_type}  If empty put "NaN"
import pandas as pd


def concatenate_var_cat(cleaned_lib_df):
    cleaned_lib_df['libelle_nettoye'] = cleaned_lib_df.apply(
        lambda row: f'{row["libelle_activite_apet"]} AUTO_{row["liasse_type"] if pd.notna(row["liasse_type"]) else "NaN"}' +
                    f' NAT_SICORE_{row["activ_surf_et"] if pd.notna(row["activ_surf_et"]) else "NaN"}' +
                    f' SURF_{row["activ_surf_et"] if pd.notna(row["activ_surf_et"]) else "NaN"}' +
                    f' EVT_SICORE_{row["evenement_type"]}' if pd.notna(row["evenement_type"]) else "NaN",
        axis=1
    )
    return cleaned_lib_df
