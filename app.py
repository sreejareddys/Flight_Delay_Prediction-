# -*- coding: utf-8 -*-
import pickle
import datetime
import os
import pandas as pd    
import numpy as np
import scipy
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
import holidays
from flask import Flask, render_template, request, jsonify
from model import get_prediction
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine,inspect
from flask_sqlalchemy import SQLAlchemy
import sqlite3

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
db_dir = os.path.join(basedir, "db")
if not os.path.exists(db_dir):
    os.mkdir(db_dir)

db_path = os.path.join(db_dir, "flights_data.sqlite")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Now create the engine using the same URI
engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
conn = engine.connect()
session = Session(engine)

application = app

carriers_dict = {
    'American Airlines (AA)': 'AA',
    'Alaska Airlines (AS)': 'AS',
    'JetBlue (B6)': 'B6',
    'Delta Air Lines (DL)': 'DL',
    'Atlantic Southeast Airlines (EV)': 'EV',
    'Frontier Airlines (F9)': 'F9',
    'Hawaiian Airlines (HA)': 'HA',
    'Spirit Airlines (NK)': 'NK',
    'SkyWest Airlines (OO)': 'OO',
    'Virgin America (UA)': 'UA',
    'United Airlines (VX)': 'VX',
    'Southwest Airlines (WN)': 'WN', 
}

dep_time_dict = {
    0: '0001-0559',
    1: '0001-0559',
    2: '0001-0559',
    3: '0001-0559',
    4: '0001-0559',
    5: '0001-0559',
    6: '0600-0659',
    7: '0700-0759',
    8: '0800-0859',
    9: '0900-0959',
    10: '1000-1059',
    11: '1100-1159',
    12: '1200-1259',
    13: '1300-1359',
    14: '1400-1459',
    15: '1500-1559',
    16: '1600-1659',
    17: '1700-1759',
    18: '1800-1859',
    19: '1900-1959',
    20: '2000-2059',
    21: '2100-2159',
    22: '2200-2259',
    23: '2300-2359'
}

days_dict = {
    1: 'Monday',
    2: 'Tuesday',
    3: 'Wednesday',
    4: 'Thursday',
    5: 'Friday',
    6: 'Saturday',
    7: 'Sunday'
}   

bins = [-np.inf, 1, 21, 61, 121, 181, np.inf]

us_holidays = us_holidays = holidays.US()
X_cat_cols = ['CARRIER', 'ORIGIN', 'DEST', 'MONTH', 'DAY_OF_WEEK_H', 'DEP_TIME_BLK']


# open files
DIR = ""

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

reg_model = load_model("Model/regression_model.h5", custom_objects={'rmse': rmse})
classif_model = load_model("Model/classification_model.h5")
outlier_model = load_model("Model/outlier_model.h5")

#graph = tf.get_default_graph()
#graph = tf.compat.v1.get_default_graph()

with open("Model/OneHotEncoder.pkl", 'rb') as f:
    OHE = pickle.load(f)
with open("Model/Scaler.pkl", 'rb') as f:
    Scaler = pickle.load(f)
with open("Model/airports_dict.pkl", 'rb') as f:
    airports_dict = pickle.load(f)
connections = pd.read_csv("Data/connections.csv", index_col=[0, 1])
test_sample = pd.read_csv("Data/test_sample.csv")

inv_airports_dict = dict(zip(airports_dict.values(), airports_dict.keys()))
inv_carriers_dict = dict(zip(carriers_dict.values(), carriers_dict.keys()))

def fill_from_sample(df=test_sample):

    sample = df.sample(1)

    y = sample['ARR_DELAY'].values[0]

    return dict(
        carrier_name = inv_carriers_dict[sample['CARRIER'].values[0]],
        origin_name = inv_airports_dict[sample['ORIGIN'].values[0]],
        dest_name = inv_airports_dict[sample['DEST'].values[0]],
        dep_date = sample['FL_DATE'].values[0],
        dep_time = sample['DEP_TIME'].values[0],
        duration_hour = str(sample['DURATION_HOUR'].values[0]),
        duration_min = str(sample['DURATION_MIN'].values[0]),
        true_delay = delay_to_message(y)
    )

def get_results(
    carrier_name,
    origin_name,
    dest_name,
    dep_date,
    dep_time,
    duration_hour,
    duration_min
):

    X = transform_inputs(
        carrier_name,
        origin_name,
        dest_name,
        dep_date,
        dep_time,
        duration_hour,
        duration_min
    )
    
    with graph.as_default():
        reg = reg_model.predict(X)[0]
        outlier_pred = outlier_model.predict(X)[0]
        outlier_proba = get_outlier_as_proba(outlier_pred)
        classes_proba = class_predict(X, outlier_proba)[0]
        
    results = dict(
        reg = delay_to_message(reg) + ".",
        classes = {},
    )

    for i, pred in enumerate(classes_proba):
        results['classes'][str(i)] = f"{pred * 100:.1f}%"

    return results

def delay_to_message(delay):
    delay = round(int(delay))
    if delay < 0:
        message = f"{abs(delay)} min in advance"
    elif delay == 0:
        message = "on time"
    elif delay > 0:
        message = f"{delay} min late"
    else:
        raise ValueError("Delay is not a number") 

    return message

def transform_date(date):
    date_time = datetime.datetime.strptime(date, '%Y-%m-%d')
    month = date_time.month
    if date_time in us_holidays:
        day_of_week_h = "Holiday"
    else:
        day_of_week_h = days_dict[date_time.isoweekday()]
    return month, day_of_week_h

def transform_inputs(
    carrier_name,
    origin_name,
    dest_name,
    dep_date,
    dep_time,
    duration_hour,
    duration_min
):

    carrier_id = carriers_dict[carrier_name]
    origin_id = airports_dict[origin_name]
    dest_id = airports_dict[dest_name]

    dep_hour = int(dep_time[:2])
    dep_time_blk = dep_time_dict[dep_hour]

    distance, median_route_time = connections.loc[(origin_id, dest_id)]
    duration = int(duration_hour) * 60 + int(duration_min)
    
    diff_from_median_route_time = duration - median_route_time

    X = Scaler.transform([[distance, diff_from_median_route_time]])

    month, day_of_week_h = transform_date(dep_date)

    sample = np.array([[
        carrier_id,
        origin_id,
        dest_id,
        month,
        day_of_week_h,
        dep_time_blk
    ]], dtype=object)

    X = scipy.sparse.hstack((X, OHE.transform(sample)))

    return X.tocsr()

def class_predict(X, outliers_rate=0.023):
    classif_no_outliers = classif_model.predict(X)
    classif_no_outliers = classif_no_outliers * (1 - outliers_rate)
    return np.hstack((classif_no_outliers, np.array([outliers_rate])))

def get_outlier_as_proba(pred_rate, base_rate=0.023):
    min_pred = base_rate * 0.5
    max_pred = base_rate * 1.5
    return min_pred + pred_rate * (max_pred - min_pred)


@app.route("/")
def main():
    """Return the homepage."""
    return render_template("login.html")

@app.route("/index1")
def index1():
    """Return the homepage."""
    return render_template("index1.html")


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        random = request.form.get('random')

        if random:
            results = fill_from_sample()
            return jsonify(results)

        else:
            inputs = request.form
            results = get_results(**inputs)
            return jsonify(results)   

    return render_template("index.html")


@app.route("/index2")
def index2():
    """Return the homepage."""
    return render_template("index.html")

@app.route("/years")
def years():
    years = pd.read_sql("SELECT year FROM flights_data",engine)
    year_list = years['year'].unique()
    return jsonify(year_list.tolist())

@app.route("/topflights2018")
def topflights2018():
    """Return a list of sample names with their average delay time."""
    flight_data = pd.read_sql("SELECT * FROM flights_data",engine)
    flight_data_delay = flight_data[[ "year", "carrier_name", "arr_delay"]]
    flight_data_delay_carrier = flight_data_delay.loc[(flight_data_delay['year'] == 2018), :]
    flight_data_delay_grouped = flight_data_delay_carrier.groupby(['carrier_name'])
    top_flights = flight_data_delay_grouped["arr_delay"].mean()
    flights_2018 = top_flights.to_dict()
    flights_2018 = sorted(flights_2018.items(), key=lambda t: t[1])
    return jsonify(flights_2018)

@app.route("/topflights/<Inputyear>")
def topflights(Inputyear):
    """Return a list of sample names with their average delay time."""
    flight_data = pd.read_sql("SELECT * FROM flights_data",engine)
    flight_data_delay = flight_data[[ "year", "carrier_name", "arr_delay"]]
    Inputyear = int(Inputyear)
    flight_data_delay_carrier = flight_data_delay.loc[(flight_data_delay['year'] == Inputyear), :]
    flight_data_delay_grouped = flight_data_delay_carrier.groupby(['carrier_name'])
    top_flights = flight_data_delay_grouped["arr_delay"].mean()
    top_flights = top_flights.to_dict()
    top_flights = sorted(top_flights.items(), key=lambda t: t[1])
    return jsonify(top_flights)

@app.route("/topflightsName/<Airport>/<Inputyear>")
def topflightsName(Airport, Inputyear):
    """Return a list of sample names with their average delay time."""
    flight_data = pd.read_sql("SELECT * FROM flights_data",engine)
    flight_data_delay = flight_data[[ "year", "airport_name", "carrier_name", "arr_delay"]]
    flight_data_delay = flight_data_delay.replace("Dallas/Fort Worth, TX: Dallas/Fort Worth International", "Dallas Fort Worth International")
    flight_data_delay = flight_data_delay.replace("Houston, TX: George Bush Intercontinental/Houston", "George Bush Intercontinental Houston")
    flight_data_delay = flight_data_delay.replace("Atlanta, GA: Hartsfield-Jackson Atlanta International", "Hartsfield Jackson Atlanta International")
    Inputyear = int(Inputyear)
    #Inputyear = 2018
    flight_data_delay_carrier = flight_data_delay.loc[(flight_data_delay['year'] == Inputyear)&(flight_data_delay['airport_name'] == Airport), :]
    flight_data_delay_grouped = flight_data_delay_carrier.groupby(['carrier_name'])
    top_flights = flight_data_delay_grouped["arr_delay"].mean()
    top_flightsName = top_flights.to_dict()
    top_flightsName = sorted(top_flightsName.items(), key=lambda t: t[1])
    return jsonify(top_flightsName)


@app.route("/topflightsAll")
def topflightsAll():
    """Return a list of sample names with their average delay time."""
    flight_data = pd.read_sql("SELECT * FROM flights_data",engine)
    flight_data_delay = flight_data[[ "year", "carrier_name", "arr_delay"]]
    flight_data_delay_grouped = flight_data_delay.groupby(['carrier_name'])
    top_flights = flight_data_delay_grouped["arr_delay"].mean()
    top_flights_list = top_flights.to_dict()
    return jsonify(top_flights_list)

@app.route("/top_airports")
def airports():
    airports_data = pd.read_sql("SELECT airport_name,arr_flights FROM flights_data",engine)
    airports_data = airports_data.replace("Dallas/Fort Worth, TX: Dallas/Fort Worth International", "Dallas Fort Worth International")
    airports_data = airports_data.replace("Houston, TX: George Bush Intercontinental/Houston", "George Bush Intercontinental Houston")
    airports_data = airports_data.replace("Atlanta, GA: Hartsfield-Jackson Atlanta International", "Hartsfield Jackson Atlanta International")
    airport_grouped = airports_data.groupby(["airport_name"])
    top_airports = airport_grouped['arr_flights'].sum()
    top_airports = top_airports.sort_values(ascending = False)
    topten_airports = top_airports.head(10)
    top_airport_names = topten_airports.to_dict()
    return jsonify(top_airport_names)
    
@app.route("/monthly_count/<month>")
def month_count(month):
   airports_lat_lng = pd.read_sql("SELECT  airport_name,Latitude,Longitude,month,sum(arr_flights) sum_arr_flights,(sum(arr_del15)/sum(arr_flights))*100 del_pct FROM flights_data WHERE airport IN ('ATL', 'DFW', 'SFO', 'ORD', 'DEN', 'LAX', 'PHX', 'HOU', 'LAS', 'MSP') group by airport_name,month,Latitude,Longitude",engine)
   airports_lat_lng = airports_lat_lng.set_index('airport_name')
   month = int(month)
   print(month)
   test_data = airports_lat_lng.loc[airports_lat_lng['month']== month,:]
   air_dict = test_data.to_dict('index')
   print(air_dict)
   return jsonify(air_dict)

@app.route('/index')
def index():
   return render_template("index2.html")

@app.route('/input')
def flight_input():
    return render_template("input.html")
import pandas as pd
@app.route('/output')
def flight_output():
    origin = request.args.get('Origin','').upper()
    dest = request.args.get('Destination','').upper()
    airline = request.args.get('Airline','').upper()
    date = request.args.get('Date')
    hour = request.args.get('Hour')
    df = pd.read_csv("Data/Processed_data.csv")
    unique_origins = sorted(df['origin'].unique().tolist())
    unique_dests = sorted(df['dest'].unique().tolist())
    unique_carrier = sorted(df['carrier'].unique().tolist())
    if origin not in unique_origins or dest not in unique_dests or airline not in unique_carrier:
        warning_msg = "⚠️ Please enter a valid Origin and Destination airport code or airline code."
        return render_template("output.html", warning=warning_msg)
    else:
    # Inputs are valid — proceed with prediction
        prediction = get_prediction(origin, dest, airline, date, hour)
   
    return render_template("output.html", result=prediction)

@app.route("/signup")
def signup():
    name = request.args.get('username','')
    dob = request.args.get('DOB','')
    sex = request.args.get('Sex','')
    contactno = request.args.get('CN','')
    email = request.args.get('email','')
    martial = request.args.get('martial','')
    password = request.args.get('psw','')

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `accounts` (`name`, `dob`,`sex`,`contact`,`email`,`martial`, `password`) VALUES (?, ?, ?, ?, ?, ?, ?)",(name,dob,sex,contactno,email,martial,password))
    con.commit()
    con.close()

    return render_template("login.html")

@app.route("/signin")
def signin():
    mail1 = request.args.get('uname','')
    password1 = request.args.get('psw','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `email`, `password` from accounts where `email` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("login.html")

    elif mail1 == data[0] and password1 == data[1]:
        return render_template("index.html")

    
    else:
        return render_template("login.html")


@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/login')
def login():
    return render_template("login.html")
@app.route('/srija')
def srija():
    return render_template("srija.html")


if __name__ == "__main__":
    app.run(debug=True)