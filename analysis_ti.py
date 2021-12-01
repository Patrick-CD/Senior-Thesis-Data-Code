from min_sig.new_tools import Tools
from min_sig.self_sig import Self_sig
import pandas as pd
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
import pickle
import os
from sklearn.preprocessing import MinMaxScaler



if not os.path.exists('preddf'):
    os.mkdir('preddf')
    os.mkdir('quantile')
    os.mkdir('daily')
    os.mkdir('analysis')


def lead_sig_Bach(p):
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    with open("../../self_sig/999_generation/%s.txt" % p, 'rb') as f:
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
    
    exp_data = trial.loc[(trial.TradingDay >= td[0])&(trial.TradingDay < td[training_period+evaluate_period])]
    used = exp_data[indicator_list].values
    used_normalized = scaler.fit_transform(used)

    ret = exp_data['ret_5'].values
    ret_normalized = scaler.fit_transform(ret.reshape(-1,1))

    with open("../../self_sig/999_generation/%s.txt" % p, 'rb') as f:
        data = pickle.load(f)
    
    td = data.TradingDay.unique()
    
    with open("/home/manager/research/min_study/fit_study/LSTM/prediction/%s_LR.txt" % p, 'rb') as f:
        prediction = pickle.load(f)
    actual_prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
    
    this_data = data.loc[(data.TradingDay>=td[training_period])
                         &(data.TradingDay<td[training_period+evaluate_period])].reset_index()
    this_data['pred'] = actual_prediction
    
    stats = {}
    stats[p] = {}
    
    for predictor in indicator_list+['pred']:

        back_test = Tools().get_min_ret(deepcopy(this_data), [predictor])
        
        ret_range = [5]


        combination = []
        for j in ret_range:
            combination.append([predictor, 'ret_%s' % j])

        sharp, rbl, md = Tools().ret_analysis(deepcopy(back_test),
                                         [predictor],
                                         [predictor + '_ret'],
                                         'daily/%s_adj' % p, False, False, False)
        stats[p][predictor] = [sharp, rbl, md]
    print('daily' + p)
    

#     back_test = Tools().get_quantile_ret(deepcopy(this_data), ['pred'])

#     with open("/home/manager/research/min_study/fit_study/LSTM/preddf/quan_%s_adj.txt" % p, 'wb') as f:
#         pickle.dump(back_test, f)
    
    return stats
    
if __name__ == "__main__":
    #Product = list(Tools().qryWY("select distinct ProductID\
    #                              from bardata.minute_ext\
    #                              where ExchangeID = 'SHFE' and ProductID != 'wr'").ProductID)
    Product = ['cu', 'al', 'zn', 'ni', 'rb', 'hc', 'ag', 'au', 'bu', 'fu', 'ru', 'sp']
    pool = Pool(len(Product))
    run = [pool.apply_async(lead_sig_Bach, (p,)) for p in Product]
    pool.close()
    pool.join()

    res = [r.get() for r in run]
    
    dic = {}
    for r in res:
        dic = {**dic, **r}

    with open("analysis/stats_ti.txt", 'wb') as f:
        pickle.dump(dic, f)
