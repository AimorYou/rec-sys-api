import pickle
from flask import Flask, request, jsonify
from lightfm.data import Dataset
from helpers import *


app = Flask(__name__)

dataset = Dataset()
lightfm_mapping = fill_dataset_create_mapper(dataset, interactions_df, users_df, items_df)
train_mat, users_features, items_features = create_sparse_data(dataset, interactions_df, users_df, items_df)
model_lightfm = pickle.load(open('light_fm.pkl', 'rb'))
model_catboost = pickle.load(open('model_catboost_full.pkl', 'rb'))


@app.post('/catboost')
def predict_catboost():
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
    req = request.get_json()

    user_id = req.get("id")
    if not user_id:
        return "id not found", 404

    rec_ids = predict_items_ids(model_lightfm, users_features, items_features, lightfm_mapping, user_id, top_n=6)
    return list(map(int, rec_ids)), 200


if __name__ == '__main__':
    app.run()
