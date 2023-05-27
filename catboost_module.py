import pickle
from copy import deepcopy

from helpers import *


class CatBoostModule:
    def __init__(self, catboost_path):
        """
        Метод инициализации.
        :param catboost_path: Путь к файлу с моделью CatBoost
        """
        self.items_df = items_df
        self.mapping = AIM_TO_BODY_PART

        self.model_catboost = pickle.load(open(catboost_path, 'rb'))

    def prepare_data(self, df_req, aim, sex, weight):
        """
        Метод для подготовки DataFrame для модели CatBoost и отбора кандидатов.
        :param df_req: Pandas DataFrame с заявкой
        :param aim: Цель из заявки
        :param sex: Пол из заявки
        :param weight: Вес из заявки
        :return: Подготовленный DataFrame для модели CatBoost
        """
        full_items_df = deepcopy(self.items_df)
        trunc_items_df = None

        if aim == AIM_HEALTH:
            trunc_items_df = (full_items_df[(full_items_df["body_part"] == "cardio") |
                                            (full_items_df["name"].str.contains("stretch")) |
                                            (full_items_df["body_part"] == "neck")])
        elif aim == AIM_WEIGHT_LOSS:
            trunc_items_df = (full_items_df[(full_items_df["body_part"] == "cardio") |
                                            ((full_items_df["equipment"] == "body weight") &
                                            (full_items_df["difficulty"] < 8))])
        elif (sex == "m" and weight > 80) or sex == "f":
            trunc_items_df = (full_items_df[np.logical_not((full_items_df["equipment"] == "body weight") &
                                                           (full_items_df["difficulty"] > 7))])

        final_items_df = full_items_df if trunc_items_df is None else trunc_items_df
        df_req['key'] = 0
        final_items_df['key'] = 0

        full_df = df_req.merge(final_items_df, on='key', how='outer')
        final_df = full_df.drop(columns=["key", "equipment", "id_x", "name"])

        final_df["tmp"] = final_df["level"]
        final_df.drop(columns=["level"], inplace=True)
        final_df.rename(columns={'tmp': 'level'}, inplace=True)

        return final_df

    def ranking(self, final_df):
        """
        Метод для применения модели CatBoost.
        :param final_df: Подготовленный DataFrame для модели CatBoost
        :return: DataFrame с рассчитанным predict
        """
        final_df["predict"] = self.model_catboost.predict_proba(final_df.drop(columns=["id_y"]))[:, 1]

        return final_df

    def re_ranking(self, df_req, ranked_df, aim, rand_coef=2):
        """
        Метод для реранжирования и учета бизнес-логики.
        :param df_req: Pandas DataFrame с заявкой
        :param ranked_df: DataFrame с рассчитанным predict
        :param aim: Цель из заявки
        :param rand_coef: Коэффициент рандомизации предскзаний
        :return: Список, содержащий id рекомендованных упражнений
        """
        if aim != AIM_MUSCLE_GROUP:
            rec_ids = (np.random.choice(ranked_df.sort_values(by="predict", ascending=False)["id_y"]
                                                 .head(6*rand_coef).values, 6).reshape(6, ).tolist())

            return rec_ids

        ex_per_group = 6 // df_req[self.mapping.keys()].values.sum()

        rec_ids = []
        for group in self.mapping:
            if not df_req[group][0]:
                continue

            rec_ids_group = (np.random.choice(ranked_df[ranked_df["body_part"] == self.mapping[group]]
                             .sort_values(by="predict", ascending=False)["id_y"]
                             .head(ex_per_group*rand_coef).values, ex_per_group)
                             .reshape(ex_per_group, ).tolist())
            rec_ids.extend(rec_ids_group)

        return rec_ids
