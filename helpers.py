import pandas as pd
import numpy as np


TRAIN_BODY_PART = ['height', 'weight', 'level',
                   'trainHands','trainLegs','trainBack',
                   'trainPress', 'trainChest', 'trainShoulders']
AIM_HEALTH = 'health'
AIM_WEIGHT_LOSS = 'weight loss'
AIM_MUSCLE_GROUP = 'muscle group'

interactions_df = pd.read_csv("interactions.csv")
users_df = pd.read_csv("users.csv")
items_df = pd.read_csv("items.csv")


def prepare_data(df_req, items_df, aim):
    if aim == AIM_HEALTH:
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


def get_user_item_features(users, items):
    sex_features = users['sex'].unique()
    aim_features = users['aim'].unique()

    items_features = np.append(
        items.body_part.unique(),
        ['difficulty']).astype(str)

    users_features = np.hstack([
        sex_features,
        aim_features,
        TRAIN_BODY_PART]).astype(str)

    return users_features, items_features


def make_item_features_dct(row):
    feature_dct = {el: 1 if row['body_part'] == el else 0 for el in items_df['body_part'].unique()}
    feature_dct['difficulty'] = row['difficulty']
    return feature_dct


def make_user_features_dct(row):
    features_dct = {el: 1 if row['sex'] == el else 0 for el in users_df['sex'].unique()}

    if row['aim'] == AIM_WEIGHT_LOSS:
        features_dct[AIM_WEIGHT_LOSS] = 1
        features_dct[AIM_MUSCLE_GROUP] = 0
        features_dct[AIM_HEALTH] = 0
    elif row['aim'] == AIM_HEALTH:
        features_dct[AIM_WEIGHT_LOSS] = 0
        features_dct[AIM_MUSCLE_GROUP] = 0
        features_dct[AIM_HEALTH] = 1
    else:
        features_dct[AIM_WEIGHT_LOSS] = 0
        features_dct[AIM_MUSCLE_GROUP] = 1
        features_dct[AIM_HEALTH] = 0

    for el in TRAIN_BODY_PART:
        features_dct[el] = row[el]

    return features_dct


def df_to_tuple_iterator(df):

    return zip(*df.values.T)


def concat_last_to_list(t):

    return t[0], list(t[1:])[0]


def df_to_tuple_list_iterator(df):

    return map(concat_last_to_list, zip(*df.values.T))


def fill_dataset_create_mapper(dataset, interactions, users, items):
    dataset.fit(interactions['user'].unique(),
                interactions['item'].unique())  # add user, item mappings

    users_features, items_features = get_user_item_features(users, items)

    dataset.fit_partial(user_features=users_features,
                        item_features=items_features)  # add user and item features to dataset

    users['features'] = users.apply(make_user_features_dct, axis=1)
    items['features'] = items.apply(make_item_features_dct, axis=1)

    lightfm_mapping = dataset.mapping()
    lightfm_mapping = {
        'users_mapping': lightfm_mapping[0],
        'user_features_mapping': lightfm_mapping[1],
        'items_mapping': lightfm_mapping[2],
        'item_features_mapping': lightfm_mapping[3],
    }

    lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
    lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}

    return lightfm_mapping


def create_sparse_data(dataset, interactions, users, items):
    interactions_mat, _ = dataset.build_interactions(df_to_tuple_iterator(interactions[['user', 'item', 'rank']]))

    known_users_filter = users['id'].isin(interactions['user'].unique())
    users_features = dataset.build_user_features(
        df_to_tuple_list_iterator(
            users.loc[known_users_filter, ['id', 'features']]
        )
    )

    known_items_filter = items['id'].isin(interactions['item'].unique())
    items_features = dataset.build_item_features(
        df_to_tuple_list_iterator(
            items.loc[known_items_filter, ['id', 'features']]
        )
    )
    return interactions_mat, users_features, items_features


def predict_items_ids(model, users_features, items_features, lightfm_mapping, user_id, top_n=6):
    row_id = lightfm_mapping['users_mapping'][user_id]
    all_cols = list(lightfm_mapping['items_mapping'].values())
    pred = model.predict(row_id,
                         all_cols,
                         user_features=users_features,
                         item_features=items_features)
    top_cols = np.argpartition(pred, -np.arange(50))[-50:][::-1]
    top_cols = np.random.choice(top_cols, top_n, replace=False)
    return list(map(lightfm_mapping['items_inv_mapping'].get, top_cols))
