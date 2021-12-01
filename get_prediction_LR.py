from new_tools import Tools
import pandas as pd
from copy import deepcopy
from multiprocessing import Pool
import pickle
import os

from sklearn.preprocessing import MinMaxScaler

from statsmodels.formula.api import ols
import statsmodels.tsa.api as smt

import gc

from copy import deepcopy

from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

import pandas as pd
import numpy as np


def predictor(p):
    print(p)
    
    with open("data/%s_adj.txt" % p, 'rb') as f:
        data = pickle.load(f)
    trial = data.reset_index()
    
    indicator_list= ['last_hal', 'BBI',
       'vwapdiff', 'LR_lead_vwap', 'weighted_lead', 'Price_ratio', 'trend_LR',
       'TRIX', 'aaba', 'aroon_v', 'VPT', 'ROC', 'Elder_ray', 'trend_diff',
       'Bias', 'trend_diff_vwap', 'avg_boosting', 'CCI', 'aroon_osc', 'CR',
       'simple_lead', 'wmp_hal']

    td = data.TradingDay.unique()
    training_period = 10
    evaluate_period = 120
    prediction = []
    
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    trial[indicator_list] = scaler_x.fit_transform(trial[indicator_list].values)
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    trial['ret_1'] = scaler_y.fit_transform(trial['ret_1'].values.reshape(-1,1))
    
    
    for i in range(evaluate_period):
        dtdf = trial.query(' \'%s\'  <=TradingDay<=  \'%s\'  ' % (td[i], td[i + training_period - 1]))
        
        x = dtdf[indicator_list].values
        y = dtdf['ret_1'].values
        
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(x, y)
        
        testdf = trial.loc[trial.TradingDay == td[i+training_period]]
        pred = regr.predict(testdf[indicator_list].values)
        
        prediction += list(pred)


    with open("prediction/%s_LR.txt" % p, 'wb') as f:
        pickle.dump(prediction, f)

if __name__ == "__main__":
    #Product = list(Tools().qryWY("select distinct ProductID\
    #                              from bardata.minute_ext\
    #                              where ExchangeID = 'SHFE' and ProductID != 'wr'").ProductID)
    Product = ['cu', 'al', 'zn', 'ni', 'rb', 'hc', 'ag', 'au', 'bu', 'fu', 'ru', 'sp']
    pool = Pool(len(Product))
    run = [pool.apply_async(predictor, (p,)) for p in Product]
    pool.close()
    pool.join()

    res = [r.get() for r in run]
