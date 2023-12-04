import fasttext
from preprocessor import Preprocessor
import pandas as pd

list_libs = ["guerisseur de boulanger", "boulanger"]

model_input = {
    "query": list_libs,
    "k": 5
}


def predict(df, model_input=model_input, model_name="FastText-APE", version="v3"):

    cleaned_df = Preprocessor().clean_text(
            df,
            text_feature="libelle_activite_apet",
        )

    model = fasttext.load_model(
        f"/home/onyxia/work/codif-ape-db/{model_name}/{version}/default.bin"
    )
    texts = cleaned_df.apply(
        lambda row: row["libelle_activite_apet"],
        axis=1).to_list()

    predictions = model.predict(texts, model_input["k"])

    predictions_formatted = {
                cleaned_df['liasse_numero'].iloc[i]: {
                    f"apet_{rank_pred+1}": {
                        "nace": predictions[0][i][rank_pred].replace(
                            "__label__", ""),
                        "probability": float(predictions[1][i][rank_pred]),
                    }
                    for rank_pred in range(model_input["k"])
                }
                for i in range(len(predictions[0]))
            }
    # Use a list comprehension to create a list of DataFrames
    dfs = [pd.json_normalize({**v, 'liasse_numero': k}, sep='_') for k, v in predictions_formatted.items()]

    # Concatenate the list of DataFrames into a single DataFrame
    result_df = pd.concat(dfs, ignore_index=True)

    # Reorder columns to have 'liasse_numero' as the first column
    result_df = result_df[['liasse_numero'] + [col for col in result_df.columns if col != 'liasse_numero']]

    # Merge the initial and new DataFrame on the 'liasse_numero' column
    merged_df = pd.merge(result_df, df, on='liasse_numero', how='inner')

    return merged_df
