import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from lightfm.data import Dataset
from lightfm import LightFM
from threading import Thread
from helpers import get_user_item_features, \
    make_user_features_dct, \
    make_item_features_dct, \
    df_to_tuple_iterator, \
    df_to_tuple_list_iterator


class LightFMModule:
    def __init__(self, lightfm_path='lightfm.pkl'):
        """
        Метод инициализации.
        :param lightfm_path: Путь к файлу с моделью LightFM
        """
        self.lightfm_path = lightfm_path

        self.dataset = Dataset()
        self.interactions_df = pd.read_csv("data/interactions.csv")
        self.users_df = pd.read_csv("data/users.csv")
        self.items_df = pd.read_csv("data/items.csv")

        self.lightfm_mapping = self.fill_dataset_create_mapper()
        self.interactions_mat, self.weights_mat, self.users_features, self.items_features = self.create_sparse_data()
        self.model_lightfm = pickle.load(open(lightfm_path, 'rb'))

    def fill_dataset_create_mapper(self):
        """
        Метод для подготовки датасета для модели lightFM.
        :return: Словарь для преобразования id пользователя или упражнения в их индекс и наоборот
        """
        self.dataset.fit(self.users_df['id'],
                         self.items_df['id'])  # add user, item mappings

        users_features, items_features = get_user_item_features(self.users_df, self.items_df)

        self.dataset.fit_partial(user_features=users_features,
                                 item_features=items_features)  # add user and item features to dataset

        self.users_df['features'] = self.users_df.apply(make_user_features_dct, axis=1)
        self.items_df['features'] = self.items_df.apply(make_item_features_dct, axis=1)

        lightfm_mapping = self.dataset.mapping()

        lightfm_mapping = {
            'users_mapping': lightfm_mapping[0],
            'user_features_mapping': lightfm_mapping[1],
            'items_mapping': lightfm_mapping[2],
            'item_features_mapping': lightfm_mapping[3],
        }

        lightfm_mapping['users_inv_mapping'] = {v: k for k, v in lightfm_mapping['users_mapping'].items()}
        lightfm_mapping['items_inv_mapping'] = {v: k for k, v in lightfm_mapping['items_mapping'].items()}

        return lightfm_mapping

    def create_sparse_data(self):
        """
        Создание Разреженных матриц для обучения модели lightFM, а также для предсказания.
        :return: Разреженная матрица взаимодействий, разреженные матрицы признаков пользователей и упражнений
        """
        interactions_mat, weights_mat = self.dataset.build_interactions(df_to_tuple_iterator(
            self.interactions_df[['user', 'item', 'rank']])
        )

        known_users_filter = self.users_df['id'].isin(self.interactions_df['user'].unique())
        users_features = self.dataset.build_user_features(
            df_to_tuple_list_iterator(
                self.users_df.loc[known_users_filter, ['id', 'features']]
            )
        )

        known_items_filter = self.items_df['id'].isin(self.interactions_df['item'].unique())
        items_features = self.dataset.build_item_features(
            df_to_tuple_list_iterator(
                self.items_df.loc[known_items_filter, ['id', 'features']]
            )
        )
        return interactions_mat, weights_mat, users_features, items_features

    def predict_items_ids(self, user_id, top_n=6):
        """
        Метод для рекомендации пользователю top_n продуктов
        :param user_id: id пользователя
        :param top_n: количество рекомендаций
        :return: list длинной top_n, содержащий id рекомендованных упражнений
        """
        print(self.users_features.shape)
        row_id = self.lightfm_mapping['users_mapping'][user_id]
        all_cols = list(self.lightfm_mapping['items_mapping'].values())
        pred = self.model_lightfm.predict(row_id,
                                          all_cols,
                                          user_features=self.users_features,
                                          item_features=self.items_features)
        top_cols = np.argpartition(pred, -np.arange(50))[-50:][::-1]
        top_cols = np.random.choice(top_cols, top_n, replace=False)
        return list(map(self.lightfm_mapping['items_inv_mapping'].get, top_cols))

    def predict_for_new_user(self, js, top_n=6):
        """
        Метод для рекомендации упражнений для нового пользователя.
        :param js: json объект с данными о пользователе
        :param top_n: количество рекомендаций
        :return: list объект, содержащий id рекомендованных упражнений
        """
        user_features = list(make_user_features_dct(pd.Series(js)).values())
        user_features_arr = np.zeros(len(user_features))
        for i in range(len(user_features)):
            user_features_arr[i] = user_features[i]
        pred = self.model_lightfm.predict(0,
                                          list(self.lightfm_mapping['items_mapping'].values()),
                                          user_features=sp.csr_matrix(user_features_arr),
                                          item_features=self.items_features,
                                          num_threads=1)
        top_cols = np.argpartition(pred, -np.arange(50))[-50:][::-1]
        top_cols = np.random.choice(top_cols, top_n, replace=False)
        return list(map(self.lightfm_mapping['items_inv_mapping'].get, top_cols))

    def add_user(self, js):
        """
        Метод для добавления нового пользователя.
        :param js: json объект с данными о пользователе
        :return: None
        """
        row = pd.Series(js)
        row['features'] = make_user_features_dct(row)
        self.users_df.loc[len(self.users_df.index)] = row
        self.users_df.iloc[:, :-1].to_csv('users.csv', index=False)

    def add_interactions(self, js):
        """
        Метод для добавления нового взаимодействия.
        :param js: json объект с данными о взаимодействии
        :return: None
        """
        row = pd.Series(js)
        self.interactions_df.loc[len(self.interactions_df.index)] = row
        self.interactions_df.to_csv('interactions.csv', index=False)

    def train(self, num_epochs=15):
        """
        Метод для обучения модели.
        :param num_epochs: Количество эпох обучения
        :return: None
        """
        self.model_lightfm = LightFM(no_components=16,
                                     learning_rate=0.025,
                                     loss='logistic',
                                     max_sampled=5,
                                     random_state=23)

        for _ in range(num_epochs):
            self.model_lightfm.fit(
                self.interactions_mat,
                sample_weight=self.weights_mat,
                user_features=self.users_features,
                item_features=self.items_features,
            )

        with open(self.lightfm_path, 'wb') as fin:
            pickle.dump(self.model_lightfm, fin, protocol=pickle.HIGHEST_PROTOCOL)

    def train_lightfm_model(self, num_epochs=15):
        """
        Метод, который запускает поток обучения модели
        :param num_epochs: Количество эпох обучения
        :return: None
        """
        new_thread = Thread(target=self.train,
                            args=(num_epochs,))
        new_thread.start()

    def update_data(self, num_epochs=15):
        """
        Метод для обновления данных о пользователях и модели.
        :param num_epochs: Количество эпох обучения.
        :return: None
        """
        self.dataset = Dataset()
        self.lightfm_mapping = self.fill_dataset_create_mapper()
        self.interactions_mat, self.weights_mat, self.users_features, self.items_features = self.create_sparse_data()
        self.train_lightfm_model(num_epochs)
