#framework
import os
from flask import Flask, request, jsonify, json
from flask_cors import CORS
import pandas as pd
import numpy as np

#mlp package
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

#lstm package
TENSORFLOW_OK = True
try:
    import tensorflow as tf  # noqa
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception:
    TENSORFLOW_OK = False

import logging

#openai
_OPENAI_OK = True
try:
    from openai import OpenAI
    _client = OpenAI()
except Exception:
    _client = None
    _OPENAI_OK = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gdp-forecast")

app = Flask(__name__)
CORS(app)


# read dataset
df = pd.read_csv("GDP_by_Country_1999-2024_WB_CurrentUSD_OVERWRITE.csv")


#calcualte growth rate.
def calculate_growth(prev, cur):
    if prev is None or prev == 0 or cur is None:
        return None
    return round((cur - prev) / prev * 100, 1)


def append_growth(years, values):
    growth_data = []
    for i, (y, v) in enumerate(zip(years, values)):
        prev = None if i == 0 else values[i - 1]
        growth_data.append({"year": int(y), "gdp": float(v), "growth": calculate_growth(prev, v)})
    return growth_data


#################################MLP prediction##########################################
def predict_mlp(country_name, window_size=10, predict_years=5):

    country = country_name
    years = df["Year"].to_numpy(dtype=int)
    series = df[country].to_numpy(dtype=np.float32)

    if series.size <= window_size:
        return {f"insufficient data for {country} "}

    # 1) create a training dataset by sliding window.
    # list 'data'  is to collect data of GDP during ten years.
    #list 'label' is to collect data of GDP at next year.
    data, Label = [], []
    for i in range(len(series) - window_size):
        data.append(series[i : i + window_size])
        Label.append(series[i + window_size])
    data = np.asarray(data)
    Label = np.asarray(Label)

    # 2) 0-1 normalization
    scaler_data, scaler_Label = MinMaxScaler(), MinMaxScaler()
    datas = scaler_data.fit_transform(data)
    #Let the labels first be converted into two-dimensional column vectors and then into one-dimensional arrays
    Labels = scaler_Label.fit_transform(Label.reshape(-1, 1)).ravel()

    # 3) create a MLP model
    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=666)

    # training
    model.fit(datas, Labels)

    ################################fit evaluation################################################
    train_pred_scaled = model.predict(datas)
    train_pred = scaler_Label.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
    Label_train = scaler_Label.inverse_transform(Labels.reshape(-1, 1)).ravel()

    resid = Label_train - train_pred
    mae = float(np.mean(np.abs(resid))) #MAE(Mean absolute error)
    rmse = float(np.sqrt(np.mean(resid ** 2))) #RMSE(Root mean square error)
    den = float(np.sum((Label_train - Label_train.mean()) ** 2)) #Total sum of squared deviations
    if den == 0:
        r2 = float("Not a Number")
    else:
        squared_residuals = resid ** 2
        sum_squared_residuals = np.sum(squared_residuals) #Sum of squared residuals(SSE)
        sse = float(sum_squared_residuals)
        r2 = 1.0 - sse / den  #Degree of interpretation
    last_loss = None
    if hasattr(model, "loss_curve_") and model.loss_curve_:
        last_loss = model.loss_curve_[-1]

    # print the evaluation of model
    logger.info(
        f"[MLP] country={country} window={window_size} n_iter={getattr(model, 'n_iter_', 'NA')} "
        f"last_loss={last_loss} R2_train={r2:.4f} MAE_train={mae:.3f} RMSE_train={rmse:.3f}"
    )

########################################################################################################

    # 4) It is an iteration to predict next five years.
    recent = series[-window_size:].tolist()
    future_years, future_GDP = [], []
    last_year = int(years[-1])
    for i in range(predict_years):
        arr = np.array(recent[-window_size:]).reshape(1, -1)
        pred_scaled = model.predict(scaler_data.transform(arr))
        pred = scaler_Label.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
        future_GDP.append(float(pred))
        future_years.append(last_year + i + 1)
        recent.append(float(pred))

    # 5) growth
    full_years = years.tolist() + future_years
    full_vals = series.astype(float).tolist() + future_GDP
    growth = append_growth(full_years, full_vals)

    return {
        "country": country,
        "model": "mlp",
        "predicted_years": future_years,
        "predicted_gdps": future_GDP,
        "data": growth,
    }

# ##################################### LSTM prediction #################################

def predict_lstm(country_name, window_size=6, predict_years=5, epochs=300, batch_size=16):
  #  if not TENSORFLOW_OK:
  #      return {"LSTM unavailable`"}

 #   if country_name not in df.columns or country_name == "year":
  #      return {f"unknown country '{country_name}"}
    country = country_name

    years = df["Year"].to_numpy(dtype=int)
    series = df[country].to_numpy(dtype=np.float32).reshape(-1, 1)

    if len(series) <= window_size + 1:
        return {f"insufficient data for {country} "}

    # 1) 0-1 Standardization
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(series)

    # 2) training set by sliding window
    data, label = [], []
    for i in range(window_size, len(scaled)):
        data.append(scaled[i - window_size : i, 0])
        label.append(scaled[i, 0])
    data = np.asarray(data).reshape(-1, window_size, 1)
    label = np.asarray(label)

    if data.shape[0] == 0:
        return {"not enough sequences after windowing"}

    # 3) lstm model
    model = Sequential()
   #model configuration
    model.add(LSTM(64, input_shape=(window_size, 1))) #add lstm layer
    model.add(Dropout(0.2))# throw some neurons to prevent overfit
    model.add(Dense(1)) #output one in Fully connected layer
    model.compile(optimizer="adam", loss="mse") #training configuration
   # training
    model.fit(data, label, epochs=epochs, batch_size=batch_size, verbose=0)

    # 4) It is an iteration to predict next five years.
    recent = scaled[-window_size:].reshape(1, window_size, 1)
    future_years, future_gdp = [], []
    last_year = int(years[-1])

    for i in range(predict_years):
        pred_scaled = model.predict(recent, verbose=0)  # predict
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
        future_gdp.append(float(pred))
        future_years.append(last_year + i + 1)

        recent = np.concatenate([recent[:, 1:, :], pred_scaled.reshape(1, 1, 1)], axis=1)

    # 5) growth
    full_years = years.tolist() + future_years
    full_gdp = series.ravel().astype(float).tolist() + future_gdp
    growth_data = append_growth(full_years, full_gdp)

    return {
        "country": country,
        "model": "lstm",
        "predicted_years": future_years,
        "predicted_gdps": future_gdp,
        "data": growth_data,
    }


# ########################################API ####################################
@app.route("/api/predict", methods=["GET"])
def predict():

    country = request.args.get("country")
    model_type = request.args.get("model")

#    if not country:
#        return jsonify({"error": "country parameter is required"}), 400

    if model_type == "mlp":
        res = predict_mlp(country)
        if "error" in res:
            return jsonify(res)
        return jsonify(res)

    elif model_type == "lstm":
        res = predict_lstm(country)
#        if "error" in res:
#            return jsonify(res)
        return jsonify(res)

    elif model_type == "both":
        mlp_res = predict_mlp(country)
        lstm_res = predict_lstm(country)

        result = {
            "country": country,
            "mlp": mlp_res,
            "lstm": lstm_res,
        }


        if "error" in lstm_res:
            return jsonify(result)
        return jsonify(result)




#select country
@app.route("/api/countries", methods=["GET"])
def countries():

    columns = df.columns
    country_list = []
    for column in columns:
        # remove "year"
        if column != "Year":
            country_list.append(column)  # add to country list

    return jsonify(country_list)


@app.route("/api/compare", methods=["GET"])
def compare():

    country1 = request.args.get("country1")
    country2 = request.args.get("country2")
    range_years = request.args.get("range")

    if not country1 or not country2 or not range_years:
        return jsonify({"country1, country2 and range are required"})
    if country1 == country2:
        return jsonify({"cannot compare the same country"})
    if country1 not in df.columns or country1.lower() == "year":
        return jsonify({f"unknown country1 '{country1}'"})
    if country2 not in df.columns or country2.lower() == "year":
        return jsonify({f"unknown country2 '{country2}'"})

    years = df["Year"].to_numpy(dtype=int)
    last_year = int(years[-1])
    start_year = max(int(years[0]), last_year - int(range_years) + 1)
    mask = (years >= start_year) & (years <= last_year)

    ys = years[mask]
    v1 = df[country1].to_numpy(dtype=float)[mask]
    v2 = df[country2].to_numpy(dtype=float)[mask]

    data = []
    for i, y in enumerate(ys):
        cur1, cur2 = float(v1[i]), float(v2[i])
        prev1 = None if i == 0 else float(v1[i - 1])
        prev2 = None if i == 0 else float(v2[i - 1])
        data.append(
            {
                "year": int(y),
                "current": cur1,
                "compare": cur2,
                "current_growth": calculate_growth(prev1, cur1),
                "compare_growth": calculate_growth(prev2, cur2),
            }
        )

    return jsonify({"country1": country1, "country2": country2, "range": int(range_years), "data": data})

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    # 1) import OpenAI by environment variable
    os.getenv("OPENAI_API_KEY")

    payload = request.get_json(silent=True) or {}
    country = (payload.get("country") or "").strip()
    series = payload.get("series") or []

    full_GDP_data = []
    seen = set()
    for i in series:
        year = i.get("year")
        gdp = float(i.get("gdp"))
        if year in seen:
            continue
        full_GDP_data.append({"year": year, "gdp": gdp})
        seen.add(year)
    full_GDP_data.sort(key=lambda x: x["year"])
 #   if not GDP_data:
  #      return jsonify({"no valid points in series"})

    # 4) set fact
    facts = {
        "country": country,
        "series": full_GDP_data,
        "note": "GDP in current US dollars; UI shows 3-decimal rounding."
    }

    # 5) requiremnet
    requiremnet = (
        "You are a macroeconomic analyst. Please write an analysis of no more than 50 words in concise English based solely on the given prediction curve data."
        "The structure should include: 1) Current situation/recent trends; 2) Outlook/Possible direction (Note uncertainty)."
        "Don't fabricate specific event or policy names; Don't output the title or entry. Just one paragraph."
    )
    send = f"country：{country}\nforecast curve JSON：\n{json.dumps(facts, ensure_ascii=False)}"

    # 6)  OpenAI configuration
    resp = _client.chat.completions.create(
            model=os.getenv("OPENAI_GPT_MODEL", "gpt-3.5"),
            temperature=0.4,
            max_tokens=300,
            messages=[
                {"role": "system", "content": requiremnet},
                {"role": "user", "content": send},
            ],
        )
    text = (resp.choices[0].message.content or "").strip()


    return jsonify({"analysis": text})


if __name__ == "__main__":

    app.run(debug=True)
