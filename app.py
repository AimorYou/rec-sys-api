import pickle
import pandas as pd
from flask import Flask, request, jsonify

from helpers import prepare_data, predict_muscle_group


app = Flask(__name__)
items_df = pd.read_csv("items.csv")
model = pickle.load(open('model_catboost_full.pkl', 'rb'))


@app.post('/')
def hello_world():
    req = request.get_json()
    aim = req.get("aim")

    if not aim:
        return "aim not found", 404

    df_req = pd.DataFrame(data=req, index=[0])
    final_df = prepare_data(df_req, items_df, aim)

    final_df["predict"] = model.predict_proba(final_df.drop(columns="id_y"))[:, 2]
    if aim == "muscle group":
        rec_ids = predict_muscle_group(df_req, final_df)
    else:
        rec_ids = final_df.sort_values(by="predict", ascending=False)[["id_y"]].head(6).values.reshape(6, ).tolist()

    return rec_ids, 200


if __name__ == '__main__':
    app.run()
