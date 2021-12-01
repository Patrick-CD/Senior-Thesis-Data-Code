from new_tools import Tools
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
    with open("data/%s.txt" % p, 'rb') as f:
        data = pickle.load(f)
    trial = data.reset_index()
    
    td = data.TradingDay.unique()
    training_period = 10
    evaluate_period = 120
    
    exp_data = trial.loc[(trial.TradingDay >= td[0])&(trial.TradingDay < td[training_period+evaluate_period])]

    with open("data/%s.txt" % p, 'rb') as f:
        data = pickle.load(f)
    
    td = data.TradingDay.unique()
    
    this_data = data.loc[(data.TradingDay>=td[training_period])
                         &(data.TradingDay<td[training_period+evaluate_period])].reset_index()
    
    retlis = [1,3,5,15]
    epochlis = [10,30,50,70]
    for r in retlis:
        for e in epochlis:
            with open("prediction/%s_ret_%s_%sepoch.txt" % (p,r,e), 'rb') as f:
                prediction = pickle.load(f)
            ret = exp_data['ret_%s'%r].values
            ret_normalized = scaler.fit_transform(ret.reshape(-1,1))
            actual_prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
            this_data['pred_ret_%s_%sepoch'%(r,e)] = actual_prediction
            
    
    with open("prediction/%s_LR.txt" % p, 'rb') as f:
        prediction = pickle.load(f)
    ret = exp_data['ret_5'].values
    ret_normalized = scaler.fit_transform(ret.reshape(-1,1))
    actual_prediction = scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
    this_data['pred_LR'] = actual_prediction
    
    stats = {}
    stats[p] = {}
    
    for predictor in ['pred_LR']+['pred_ret_%s_%sepoch'%(r,e) for r in retlis for e in epochlis]:

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

    with open("analysis/stats_com.txt", 'wb') as f:
        pickle.dump(dic, f)
