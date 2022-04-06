import pandas as pd
import numpy as np
from tqdm import tqdm

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from warnings import filterwarnings
filterwarnings('ignore')

def ARIMA_DATA(df, msoa, category):
    ###Gets the data of a specific MSOA and category to use in the ARIMA_OPTIMAL function
    ###

    df = df[(df['MSOA'] == msoa) & (df['Crime type'] == category)]
    df = df[['Date', 'count']]
    df = df.set_index('Date')
    
    return df

def ARIMA_STATIONARY(df):
    ###Returns a stationary dataframe, created by ARIMA_DATA
    ###
    if adfuller(df['count'])[1] > 0.05:
        df = df.diff().dropna()
    
    return df

def ARIMA_OPTIMAL(stationary_data, stationary_test):
    ### Looks for the best ARMA(p,q) + constant model according to MSOA and crime type
    ###
    
    order_aic_bic = list()

    # Loop over AR order
    for p in range(1, 3):
        # Loop over MA order
        for q in range(1, 3):
            # Fit model
            model = SARIMAX(stationary_data, order=(p,0,q), trend='c')
            results = model.fit(disp=0)

            # Add order and scores to list
            order_aic_bic.append((p, q, results.aic, results))
            
    order_df = pd.DataFrame(order_aic_bic, columns=['p', 'q', 'aic', 'results'])
    optimum = order_df[order_df['aic'] == order_df['aic'].min()]
    optimum.reset_index(inplace=True)

    # MASE
    forecast = results.get_forecast(steps=len(stationary_test.index) + 1)
    mean_forecast = forecast.predicted_mean.to_frame()['predicted_mean']
    mean_forecast.index = pd.to_datetime(mean_forecast.index, format = '%Y-%m-%d').strftime('%Y-%m')
    mase = mase_loss(y_train=stationary_data['count'], y_pred=mean_forecast, y_test=stationary_test['count'])

    return optimum['p'][0], optimum['q'][0], optimum['aic'][0], mase

# From sktime, as package could not be imported
def mase_loss(y_test, y_pred, y_train, sp=1):
    #  naive seasonal prediction
    y_train = np.asarray(y_train)
    y_pred_naive = y_train[:-sp]
    
    # mean absolute error of naive seasonal prediction
    mae_naive = np.mean(np.abs(y_train[sp:] - y_pred_naive))
    
    # if training data is flat, mae may be zero,
    # return np.nan to avoid divide by zero error
    # and np.inf values
    if mae_naive == 0:
        return np.nan
    else:
        return np.mean(np.abs(y_test - y_pred)) / mae_naive


def get_best_models(df, df_test):
    param = list()
    
    for msoa in tqdm(df['MSOA'].unique()):
        for category in df['Crime type'].unique():
            arima_data = ARIMA_DATA(df, msoa, category)
            stationary_data = ARIMA_STATIONARY(arima_data)
            arima_test = ARIMA_DATA(df_test, msoa, category)
            stationary_test = ARIMA_STATIONARY(arima_test)
            p, q, aic, mase = ARIMA_OPTIMAL(stationary_data, stationary_test)
            param.append((msoa, category, p, q, aic, mase))
            
    return pd.DataFrame(param, columns=['MSOA', 'Crime type', 'p', 'q', 'aic', 'MASE'])


if __name__ == "__main__":

    print("Open train set")
    train = pd.read_csv("datasets/train.csv")
    train.drop("Unnamed: 0", axis=1, inplace=True)

    print("Open test set")
    test = pd.read_csv("datasets/test.csv")
    test.drop("Unnamed: 0", axis=1, inplace=True)

    print("Find best models")
    best_models = get_best_models(train, test)
    best_models.to_csv("best_models.csv")