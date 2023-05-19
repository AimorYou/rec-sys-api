import pickle
from flask import Flask, request
from lightfm_module import LightFMModule
from helpers import *


app = Flask(__name__)

module_lightfm = LightFMModule('light_fm.pkl')
model_catboost = pickle.load(open('model_catboost_full.pkl', 'rb'))


@app.post('/catboost')
def predict_catboost():
    """
    API метод для рекомендации упражнений с помощью модели Catboost
    :return: list, содержащий id рекомендованных упражнений и код
    """
    req = request.get_json()
    aim = req.get("aim")

    if not aim:
        return "aim not found", 404

    df_req = pd.DataFrame(data=req, index=[0])
    final_df = prepare_data(df_req, items_df, aim)

    final_df["predict"] = model_catboost.predict_proba(final_df.drop(columns="id_y"))[:, 2]
    if aim == "muscle group":
        rec_ids = predict_muscle_group(df_req, final_df)
    else:
        rec_ids = final_df.sort_values(by="predict", ascending=False)[["id_y"]].head(6).values.reshape(6, ).tolist()

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
    :return:
    """
    js = request.get_json()
    try:
        rec_ids = module_lightfm.predict_for_new_user(js)
        return rec_ids, 200
    except Exception as e:
        return str(e), 500


@app.post('/add_user')
def add_new_user():
    req = request.get_json()
    try:
        module_lightfm.add_user(req)
        return 'OK!', 200
    except Exception as e:
        return str(e) + '. Send all users features', 500


@app.post('/update_lightfm_model')
def update_data():
    try:
        module_lightfm.update_data()
        return 'OK!', 200
    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run()
