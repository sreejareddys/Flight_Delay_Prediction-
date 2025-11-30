# -*- coding: utf-8 -*-

from datetime import datetime
import json
import pandas as pd
import pickle
import urllib
from sklearn.ensemble import RandomForestClassifier


API_KEY = '9d37eca700052d7261c474aff9a86ae3'


def get_prediction(origin, dest, airline, date, hour):
    query_date = datetime.strptime(date, '%Y-%m-%d')
    future_gap = query_date - datetime.now()
    if 0 <= future_gap.days <= 7:
        return get_prediction_w_weather(origin, dest, airline, date, hour)
    else:
        return get_prediction_wo_weather(origin, dest, airline, date, hour)

def fill_flight_data(data, origin, dest, airline, date, hour):
    data['month'] = int(date.split("-")[1])
    data['day'] = int(date.split("-")[2])
    data['day_of_week'] = int(datetime.strptime(date, '%Y-%m-%d').weekday())
    data['hour'] = int(hour)
    
    origin_col = 'ORIGIN_' + origin
    dest_col = 'DEST_' + dest
    carrier_col = 'OP_CARRIER_' + airline

    if origin_col in data.columns:
        data[origin_col] = 1
    if dest_col in data.columns:
        data[dest_col] = 1
    if carrier_col in data.columns:
        data[carrier_col] = 1

    return data
# def fill_flight_data(data, origin, dest, airline, date, hour):
#     data['month'] = int(date.split("-")[1])
#     data['day'] = int(date.split("-")[2])
#     data['day_of_week'] = int(datetime.strptime(date, '%Y-%m-%d').weekday())
#     data['hour'] = int(hour)
#     data['ORIGIN_' + origin] = 1
#     data['DEST_' + dest] = 1
#     data['OP_CARRIER_' + airline] = 1
#     return data


def fill_weather_data(data, origin_weather, dest_weather):
    if "dewPoint" in origin_weather:
        data["DewPointTemperature_origin"] = origin_weather["dewPoint"]
    if "temperature" in origin_weather:
        data["DryBulbTemperature_origin"] = origin_weather["temperature"]
    if "precipIntensity" in origin_weather:
        data["Precipitation_origin"] = origin_weather["precipIntensity"]
    if "humidity" in origin_weather:
        data["Humidity_origin"] = origin_weather["humidity"]
    if "pressure" in origin_weather:
        data["Pressure_origin"] = origin_weather["pressure"]
    if "visibility" in origin_weather:
        data["Visibility_origin"] = origin_weather["visibility"]
    if "windSpeed" in origin_weather:
        data["WindSpeed_origin"] = origin_weather["windSpeed"]
    if "precipType" in origin_weather:
        precipType = origin_weather["precipType"].lower()
        if precipType == "rain":
            data["PrecipType_origin_rain"] = 1
        elif precipType == "snow":
            data["PrecipType_origin_snow"] = 1
        else:
            data["PrecipType_origin_nan"] = 1
    else:
        data["PrecipType_origin_nan"] = 1
    if "temperatureHigh" in origin_weather:
        data["DryBulbTemperature_origin"] = dest_weather["temperatureHigh"]
    if "dewPoint" in dest_weather:
        data["DewPointTemperature_dest"] = dest_weather["dewPoint"]
    if "temperature" in dest_weather:
        data["DryBulbTemperature_dest"] = dest_weather["temperature"]
    if "precipIntensity" in dest_weather:
        data["Precipitation_dest"] = dest_weather["precipIntensity"]
    if "humidity" in dest_weather:
        data["Humidity_dest"] = dest_weather["humidity"]
    if "pressure" in dest_weather:
        data["Pressure_dest"] = dest_weather["pressure"]
    if "visibility" in dest_weather:
        data["Visibility_dest"] = dest_weather["visibility"]
    if "windSpeed" in dest_weather:
        data["WindSpeed_dest"] = dest_weather["windSpeed"]
    if "precipType" in dest_weather:
        precipType = dest_weather["precipType"].lower()
        if precipType == "rain":
            data["PrecipType_dest_rain"] = 1
        elif precipType == "snow":
            data["PrecipType_dest_snow"] = 1
        else:
            data["PrecipType_dest_nan"] = 1
    else:
        data["PrecipType_destn_nan"] = 1
    if "temperatureHigh" in dest_weather:
        data["DryBulbTemperature_dest"] = dest_weather["temperatureHigh"]
    return data
def get_prediction_wo_weather(origin, dest, airline, date, hour):
    model = pickle.load(open('Model/rf_model_wo_weather.sav', 'rb'))
    model.n_jobs = 1

    # Load the template which has the correct feature order and number of columns
    template = pd.read_csv("Data/data_template_wo_weather.csv")
    data = template.copy()
    data = fill_flight_data(data, origin, dest, airline, date, hour)
    
    # Ensure we only use the columns from the template
    expected_columns = template.columns
    data = data[expected_columns]
    
    prediction = model.predict(data.values)[0]
    result = "delayed" if prediction > 0 else "on-time"
    return result

# def get_prediction_wo_weather(origin, dest, airline, date, hour):
#     model = pickle.load(open('Model/rf_model_wo_weather.sav', 'rb'))
#     model.n_jobs = 1

#     data = pd.read_csv("Data/data_template_wo_weather.csv")
#     data = fill_flight_data(data, origin, dest, airline, date, hour)

#     prediction = model.predict(data.values)[0]
#     result = "delayed" if prediction > 0 else "on-time"
#     return result


def query_weather(lat, lon):
    url = 'https://api.darksky.net/forecast/' + API_KEY + '/' + str(lat) + ',' + str(lon)
    request = urllib.request.Request(url)
    r = urllib.request.urlopen(request)
    response = r.read().decode("utf-8")
    return json.loads(response)


def time_match(ts, dt, hr):
    t = datetime.fromtimestamp(ts)
    return t.strftime('%Y-%m-%d') == dt and t.hour == hr


def day_match(ts, dt):
    t = datetime.fromtimestamp(ts)
    return t.strftime('%Y-%m-%d') == dt


def get_prediction_w_weather(origin, dest, airline, date, hour):
    model = pickle.load(open('Model/rf_model_w_weather.sav', 'rb'))
    model.n_jobs = 1

    data = pd.read_csv("Data/data_template_w_weather.csv")
    data = fill_flight_data(data, origin, dest, airline, date, hour)

    airports = pd.read_csv("data/airports.csv")
    origin_lat = airports[airports["iata"] == origin]["latitude"].values[0]
    origin_lon = airports[airports["iata"] == origin]["longitude"].values[0]
    dest_lat = airports[airports["iata"] == dest]["latitude"].values[0]
    dest_lon = airports[airports["iata"] == dest]["longitude"].values[0]

    origin_json = query_weather(origin_lat, origin_lon)
    dest_json = query_weather(dest_lat, dest_lon)

    key, ind = None, None
    for i, w in enumerate(origin_json["hourly"]["data"]):
        if time_match(w["time"], date, hour):
            key, ind = "hourly", i
            break
    if key is None:
        for i, w in enumerate(origin_json["daily"]["data"]):
            if day_match(w["time"], date):
                key, ind = "daily", i
                break

    if key is None:
        return get_prediction_wo_weather(origin, dest, airline, date, hour)

    origin_weather = origin_json[key]["data"][ind]
    dest_weather = dest_json[key]["data"][ind]

    data = fill_weather_data(data, origin_weather, dest_weather)

    prediction = model.predict(data.values)[0]
    result = "delayed" if prediction > 0 else "on-time"
    return result


def generate_model():
    data_cleaned = pd.read_csv("data/flights_2009-2018_sample_clean.csv")
    data_expand = pd.concat(
        [data_cleaned[['year', 'delayed', 'month', 'day', 'day_of_week', 'hour', 'DewPointTemperature_origin',
                       'DewPointTemperature_dest', 'DryBulbTemperature_origin', 'DryBulbTemperature_dest',
                       'Precipitation_origin', 'Precipitation_dest', 'Humidity_origin', 'Humidity_dest',
                       'Pressure_origin', 'Pressure_dest', 'Visibility_origin', 'Visibility_dest',
                       'WindSpeed_origin', 'WindSpeed_dest']],
         pd.get_dummies(data_cleaned['ORIGIN'], dummy_na=True, prefix='ORIGIN'),
         pd.get_dummies(data_cleaned['DEST'], dummy_na=True, prefix='DEST'),
         pd.get_dummies(data_cleaned['OP_CARRIER'], dummy_na=True, prefix='OP_CARRIER'),
         pd.get_dummies(data_cleaned['PrecipType_origin'], dummy_na=True, prefix='PrecipType_origin'),
         pd.get_dummies(data_cleaned['PrecipType_dest'], dummy_na=True, prefix='PrecipType_dest')], axis=1)

    train_data = data_expand[data_expand['year'] < 2018].drop(columns=['year']).values

    rf = RandomForestClassifier(n_estimators=20, max_depth=10)
    rf.fit(train_data[0::, 1::], train_data[0::, 0])
    filename = 'rf_model_w_weather.sav'
    pickle.dump(rf, open(filename, 'wb'))

    data_cleaned2 = data_cleaned[
        ['year', 'delayed', 'month', 'day', 'day_of_week', 'hour', 'ORIGIN', 'DEST', 'OP_CARRIER']]
    data_expand2 = pd.concat([data_cleaned2[['year', 'delayed', 'month', 'day', 'day_of_week', 'hour']],
                              pd.get_dummies(data_cleaned2['ORIGIN'], dummy_na=True, prefix='ORIGIN'),
                              pd.get_dummies(data_cleaned2['DEST'], dummy_na=True, prefix='DEST'),
                              pd.get_dummies(data_cleaned2['OP_CARRIER'], dummy_na=True, prefix='OP_CARRIER')], axis=1)
    train_data2 = data_expand2[data_expand2['year'] < 2018].drop(columns=['year']).values

    rf2 = RandomForestClassifier(n_estimators=20, max_depth=10)
    rf2.fit(train_data2[0::, 1::], train_data2[0::, 0])
    filename2 = 'rf_model_wo_weather.sav'
    pickle.dump(rf2, open(filename2, 'wb'))
