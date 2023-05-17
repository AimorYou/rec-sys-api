import pandas as pd
import numpy as np


def prepare_data(df_req, items_df, aim):
    if aim == "health":
        items_df = (items_df[(items_df["body_part"] == "cardio") |
                             (items_df["name"].str.contains("stretch")) |
                             (items_df["body_part"] == "neck")])
    df_req['key'] = 0
    items_df['key'] = 0

    full_df = df_req.merge(items_df, on='key', how='outer')
    final_df = full_df.drop(columns=["key", "equipment", "id_x", "name"])

    return final_df


def predict_muscle_group(df_req, final_df):
    mapping = {
        'trainHands': "arms",
        'trainLegs': "legs",
        'trainBack': "back",
        'trainPress': "abs",
        'trainChest': "chest",
        'trainShoulders': "shoulders"
    }

    ex_per_group = 6 // df_req[mapping.keys()].values.sum()

    rec_ids = []
    for group in mapping:
        if not df_req[group][0]:
            continue

        rec_ids_group = (final_df[final_df["body_part"] == mapping[group]]
                         .sort_values(by="predict", ascending=False)[["id_y"]]
                         .head(ex_per_group).values.reshape(ex_per_group, ).tolist())
        rec_ids.extend(rec_ids_group)

    return rec_ids
