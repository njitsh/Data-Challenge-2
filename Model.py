import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from datetime import timedelta
import glob
import os
from tqdm import tqdm
import time
import seaborn as sns

import statsmodels.api as sm
from statsmodels.compat import lzip
from statsmodels.formula.api import ols
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from numpy import cumsum
from datetime import datetime as dt

from warnings import filterwarnings
filterwarnings('ignore')

def ARIMA_DATA(df, MSOA, category):
    ###Gets the data of a specific MSOA and category to use in the ARIMA_OPTIMAL function
    ###
    
    df = df[(df['MSOA'] == int(MSOA)) & (df['Crime type'] == category)]
    df = df[['Date', 'count']]
    df = df.set_index('Date')
    
    return df

def ARIMA_STATIONARY(df):
    ###Returns a stationary dataframe, created by ARIMA_DATA
    ###
    if adfuller(df['count'])[1] > 0.05:
        df = df.diff().dropna()
    
    return df

def ARIMA_OPTIMAL(stationary_data, MSOA, category):
    ### Looks for the best ARMA(p,q) + constant model according to MSOA and crime type
    ###
    
    order_aic_bic = list()

    # Loop over AR order
    for p in range(1,4):
        # Loop over MA order
        for q in range(1,4):
            #for d in range(3):
            try:
            # Fit model
                model = SARIMAX(stationary_data, order=(p,0,q), trend='c')
                results = model.fit(disp=0)
                # Add order and scores to list
                order_aic_bic.append((p, q, results.aic))
            except:
                continue
            
    order_df = pd.DataFrame(order_aic_bic, columns=['p','q','aic'])
    optimum = order_df[order_df['aic'] == order_df['aic'].min()]
    optimum.reset_index(inplace=True)
    return optimum['p'][0], optimum['q'][0], optimum['aic'][0]

def ARIMA_PREDICT(df, MSOA, category):
    ###Forecasts via ARIMA approach
    ###
    
    arima_data = ARIMA_DATA(df, MSOA, category)
    stationary_data = ARIMA_STATIONARY(arima_data)
    
    p,q = ARIMA_OPTIMAL(stationary_data, MSOA, category)[0:2]
    
    model = SARIMAX(stationary_data, order=(p,0,q), trend='c')
    results = model.fit()
    forecast = results.get_prediction(start=-25)
    mean_forecast = cumsum(forecast.predicted_mean) + stationary_data.iloc[-1,0]
    confidence_intervals = cumsum(forecast.conf_int())
    return arima_data, mean_forecast.to_frame(), confidence_intervals

def ARIMA_SUMMARY(df, MSOA, category):
    
    stationary_data = ARIMA_STATIONARY(ARIMA_DATA(df, MSOA, category))
    
    p,q = ARIMA_OPTIMAL(stationary_data, MSOA, category)[0:2]
    
    model = SARIMAX(stationary_data, order=(p,0,q), trend='c')
    results = model.fit()
    
    return results.summary()

def model_predict(df, msoa, category):
    data, mean_forecasts, confidence_intervals = ARIMA_PREDICT(df, msoa, category)
    
    lower_limits = confidence_intervals['lower count']
    upper_limits = confidence_intervals['upper count']
    
    return data, mean_forecasts, confidence_intervals, lower_limits, upper_limits

def get_best_models(df):
    param = list()
    
    for MSOA in tqdm(df['MSOA'].unique()):
        for category in df['Crime type'].unique():
            arima_data = ARIMA_DATA(df, MSOA, category)
            stationary_data = ARIMA_STATIONARY(arima_data)
            p, q, aic = ARIMA_OPTIMAL(stationary_data, MSOA, category)
            param.append((MSOA, category, p, q, aic))
            
    return pd.DataFrame(param, columns=['MSOA', 'Crime type', 'p','q', 'aic'])


if __name__ == "__main__":
    print("Open train set")
    Train = pd.read_csv("train_count_street_data.csv")
    Train.drop("Unnamed: 0", axis=1, inplace=True)

    print("Open test set with covid")
    TestCovid = pd.read_csv("test_covid_count_street_data.csv")
    TestCovid.drop("Unnamed: 0", axis=1, inplace=True)

    print("Open test set without covid")
    TestNoCovid = pd.read_csv("test_no_covid_count_street_data.csv")
    TestNoCovid.drop("Unnamed: 0", axis=1, inplace=True)

    print("Open train set without covid")
    TrainWithNoCovid = pd.read_csv("train_with_no_covid_count_street_data.csv")
    TrainWithNoCovid.drop("Unnamed: 0", axis=1, inplace=True)

    print("Train model on Train set")
    best_models_train = get_best_models(Train)
    best_models_train.to_csv("train_best_models.csv")

    print("Train model on TrainWithNoCovid set")
    best_models_train_with_no_covid = get_best_models(TrainWithNoCovid)
    best_models_train_with_no_covid.to_csv("train_with_no_covid_best_models.csv")