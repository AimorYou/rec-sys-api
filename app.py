from flask import Flask, request

from lightfm_module import LightFMModule
from catboost_module import CatBoostModule
from helpers import *


app = Flask(__name__)

module_lightfm = LightFMModule('models/model_lightfm.pkl')
module_catboost = CatBoostModule('models/model_catboost.pkl')


@app.post('/catboost')
def predict_catboost():
    """
    API метод для рекомендации упражнений с помощью модели Catboost
    :return: list, содержащий id рекомендованных упражнений и код
    """
    req = request.get_json()
    aim = req.get("aim")
    sex = req.get("sex")
    weight = req.get("weight")
    df_req = pd.DataFrame(data=req, index=[0])

    try:
        final_df = module_catboost.prepare_data(df_req, aim, sex, weight)

    except Exception as e:
        return f"something went wrong in data preparation: {e}", 400

    try:
        ranked_df = module_catboost.ranking(final_df)
    except Exception as e:
        return f"something went wrong in ranking: {e}", 400

    try:
        rec_ids = module_catboost.re_ranking(df_req, ranked_df, aim)
    except Exception as e:
        return f"something went wrong in re-ranking: {e}", 400

    return rec_ids, 200


@app.post('/lightfm')
def predict_lightfm():
    """
    API метод для рекомендации упражнений с помощью модели LightFM
    :return: list, содержащий id рекомендованных упражнений и код
    """
    req = request.get_json()
    user_id = req.get("id")
    if not user_id:
        return "id not found", 404

    rec_ids = module_lightfm.predict_items_ids(user_id, top_n=6)
    return list(map(int, rec_ids)), 200


@app.post('/lightfm_new_user_rec')
def new_user_rec():
    """
    API метод для предсказания нового пользователя
    :return:Сообщение и код
    """
    js = request.get_json()
    try:
        rec_ids = module_lightfm.predict_for_new_user(js)
        return rec_ids, 200
    except Exception as e:
        return str(e), 500


@app.post('/add_user')
def add_new_user():
    """
    API метод для добавления нового пользователя.
    :return: Сообщение и код
    """
    req = request.get_json()
    try:
        module_lightfm.add_user(req)
        return 'OK!', 200
    except Exception as e:
        return str(e) + '. Send all users features', 500


@app.post('/add_interaction')
def add_interaction():
    """
    API метод для добавления взаимодействия
    :return: Сообщение и код
    """
    js = request.get_json()
    try:
        module_lightfm.add_interactions(js)
        return 'OK!', 200
    except Exception as e:
        return str(e) + '. Send all users features', 500


@app.post('/update_lightfm_model')
def update_data():
    """
    API метод для обновления данных и модели.
    :return: Сообщение и код
    """
    try:
        module_lightfm.update_data(num_epochs=15)
        return 'OK!', 200
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run()
