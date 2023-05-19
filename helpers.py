import pandas as pd
import numpy as np


TRAIN_BODY_PART = ['height', 'weight', 'level',
                   'trainHands', 'trainLegs', 'trainBack',
                   'trainPress', 'trainChest', 'trainShoulders']
AIM_HEALTH = 'health'
AIM_WEIGHT_LOSS = 'weight loss'
AIM_MUSCLE_GROUP = 'muscle group'

interactions_df = pd.read_csv("interactions.csv")
users_df = pd.read_csv("users.csv")
items_df = pd.read_csv("items.csv")


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


def get_user_item_features(users, items):
    """
    Метод для получения признаков пользователей и упражнений.
    :param users: Dataframe пользователей
    :param items: Dataframe упражнений
    :return: признаки пользователя, признаки упражнения
    """
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
    """
    Метод для преобразования признаков упражнения в словарь.
    :param row: Series, данные об одном упражнении
    :return: словарь, в котором ключом является имя признака, а значением - значение признака
    """
    feature_dct = {el: 1 if row['body_part'] == el else 0 for el in items_df['body_part'].unique()}
    feature_dct['difficulty'] = row['difficulty']
    return feature_dct


def make_user_features_dct(row):
    """
    Метод для преобразования признаков пользователя в словарь.
    :param row: Series, данные о пользователе
    :return: словарь, в котором ключом является имя признака, а значением - значение признака
    """
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
    """
    Метод для подготовки Dataframe в определённый формат( [(param1, param2, ..), ...]).
    :param df: Dataframe
    :return: zip объект в виде [(param1, param2, ..), ...]
    """
    return zip(*df.values.T)


def concat_last_to_list(t):
    """
    Метод для преобразования всех параметров iterable объекта в лист(кроме первого).
    :param t: Iterable объект
    :return: tuple
    """
    return t[0], list(t[1:])[0]


def df_to_tuple_list_iterator(df):
    """
    Метод для применения метода concat_last_to_list для всех рядов Dataframe.
    :param df: Dataframe
    :return: iterable объект, состоящий из tuple
    """
    return map(concat_last_to_list, zip(*df.values.T))


# def fill_dataset_create_mapper(dataset, users, items):
#     """
#     Метод для подготовки датасета для модели lightFM.
#     :param dataset: lightFM dataset
#     :param users: Dataframe пользователь
#     :param items: Dataframe упражнений
#     :return: словарь для преобразования id пользователя или упражнения в их индекс и наоборот
#     """
#     dataset.fit(users['id'],
#                 items['id'])  # add user, item mappings
#
#     users_features, items_features = get_user_item_features(users, items)
#
#     dataset.fit_partial(user_features=users_features,
#                         item_features=items_features)  # add user and item features to dataset
#
#     users['features'] = users.apply(make_user_features_dct, axis=1)
#     items['features'] = items.apply(make_item_features_dct, axis=1)
#
#     lightfm_mapping = dataset.mapping()
#     lightfm_mapping = {
#         'users_mapping': lightfm_mapping[0],
#         'user_features_mapping': lightfm_mapping[1],
#         'items_mapping': lightfm_mapping[2],
#         'item_features_mapping': lightfm_mapping[3],
#     }
#
#     lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
#     lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}
#
#     return lightfm_mapping


# def create_sparse_data(dataset, interactions, users, items):
#     """
#     Создание разреженных матриц для обучения модели lightFM, а также для предсказания.
#     :param dataset: lightFM dataset
#     :param interactions: Dataframe взаимодействий пользователя и упражнения
#     :param users: Dataframe пользователь
#     :param items: Dataframe упражнений
#     :return: разреженная матрица взаимодействий, разреженные матрицы признаков пользователей и упражнений
#     """
#     interactions_mat, _ = dataset.build_interactions(df_to_tuple_iterator(interactions[['user', 'item', 'rank']]))
#
#     known_users_filter = users['id'].isin(interactions['user'].unique())
#     users_features = dataset.build_user_features(
#         df_to_tuple_list_iterator(
#             users.loc[known_users_filter, ['id', 'features']]
#         )
#     )
#
#     known_items_filter = items['id'].isin(interactions['item'].unique())
#     items_features = dataset.build_item_features(
#         df_to_tuple_list_iterator(
#             items.loc[known_items_filter, ['id', 'features']]
#         )
#     )
#     return interactions_mat, users_features, items_features


# def predict_items_ids(model, users_features, items_features, lightfm_mapping, user_id, top_n=6):
#     """
#     Метод для рекомендации пользователю top_n продуктов
#     :param model: lightFM model
#     :param users_features: разреженная матрица признаков пользователя
#     :param items_features: разреженная матрица признаков упражнения
#     :param lightfm_mapping: словарь для преобразования id пользователя или упражнения в их индекс и наоборот
#     :param user_id: id пользователя
#     :param top_n: количество рекомендаций
#     :return: list длинной top_n, содержащий id рекомендованных упражнений
#     """
#     print(users_features.shape)
#     row_id = lightfm_mapping['users_mapping'][user_id]
#     all_cols = list(lightfm_mapping['items_mapping'].values())
#     pred = model.predict(row_id,
#                          all_cols,
#                          user_features=users_features,
#                          item_features=items_features)
#     top_cols = np.argpartition(pred, -np.arange(50))[-50:][::-1]
#     top_cols = np.random.choice(top_cols, top_n, replace=False)
#     return list(map(lightfm_mapping['items_inv_mapping'].get, top_cols))


# def predict_for_new_user(model, mapping, user_features, items_features):
#     user_features_arr = np.zeros(len(user_features))
#     for i in range(len(user_features)):
#         user_features_arr[i] = user_features[i]
#     pred = model.predict(0,
#                         list(mapping['items_mapping'].values()),
#                         user_features=sp.csr_matrix(user_features_arr),
#                         item_features=items_features,
#                         num_threads=1)
#     top_cols = np.argpartition(pred, -np.arange(50))[-50:][::-1]
#     top_cols = np.random.choice(top_cols, 6, replace=False)
#     return list(map(mapping['items_inv_mapping'].get, top_cols))


# def add_user(js):
#     row = pd.Series(js)
#     row['features'] = make_user_features_dct(row)
#     users_df.loc[len(users_df.index)] = row
#     #users_df.iloc[:, :-1].to_csv('users.csv', index=False)


# def train(lfm_model, interactions_mat, users_features, items_features, num_epochs=1):
#     lfm_model = LightFM(no_components=16, learning_rate=0.05, loss='logistic', max_sampled=5, random_state=23)
#     for _ in range(num_epochs):
#         lfm_model.fit(
#                 interactions_mat,
#                 user_features=users_features,
#                 item_features=items_features,
#         )


# def train_lightfm_model(lfm_model, interactions_mat, users_features, items_features, num_epochs=1):
#     new_thread = Thread(target=train,
#                         args=(lfm_model, interactions_mat, users_features, items_features, 1))
#     new_thread.start()
