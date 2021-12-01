from new_tools import Tools
import pandas as pd
from copy import deepcopy
from multiprocessing import Pool
import pickle
import os

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    
def create_inout_sequences(input_data, output_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = output_data[i+tw-1:i+tw]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

def predictor(p,r,e):
    print(p,r,e)
    
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

    scaler = MinMaxScaler(feature_range=(-1, 1))

    model = LSTM(22,48,1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_window = 10
    
    #50 original
    epochs = e

    exp_data = trial.loc[(trial.TradingDay >= td[0])&(trial.TradingDay < td[training_period+evaluate_period])]
    used = exp_data[indicator_list].values
    used_normalized = scaler.fit_transform(used)
    
    #5 original
    ret = exp_data['ret_%s'%r].values
    ret_normalized = scaler.fit_transform(ret.reshape(-1,1))

    train_tensor = torch.FloatTensor(used_normalized)
    ret_tensor = torch.FloatTensor(ret_normalized)

    train_inout_seq = create_inout_sequences(train_tensor, ret_tensor, train_window)

    last = 0

    for i in range(training_period):
        print(td[i])
        new_last = trial.loc[trial.TradingDay == td[i]].index[-1]

        for i in range(epochs):
            for seq, labels in train_inout_seq[last:new_last+1]:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            #if i%25 == 1:
                #print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        #print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
        last = new_last
    
    for i in range(evaluate_period):
        print(td[training_period+i])
        new_last = trial.loc[trial.TradingDay == td[training_period+i]].index[-1]

        model.eval()
        result = []
        for i in range(new_last-last):
            seq = torch.FloatTensor(train_tensor[last+i:last+train_window+i])
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
                prediction.append(model(seq).item())

        model.train()
        for i in range(epochs):
            for seq, labels in train_inout_seq[last:new_last+1]:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            #if i%25 == 1:
                #print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        #print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

        last = new_last


    with open("prediction/%s_ret_%s_%sepoch.txt" % (p,r,e), 'wb') as f:
        pickle.dump(prediction, f)

if __name__ == "__main__":
    #Product = list(Tools().qryWY("select distinct ProductID\
    #                              from bardata.minute_ext\
    #                              where ExchangeID = 'SHFE' and ProductID != 'wr'").ProductID)
    Product = ['cu', 'al', 'zn', 'ni', 'rb', 'hc', 'ag', 'au', 'bu', 'fu', 'ru', 'sp']
    retlis = [1,3,5,15]
    epochlis = [10,30,50,70]
    pool = Pool(len(Product))
    run = [pool.apply_async(predictor, (p,r,e,)) for p in Product for r in retlis for e in epochlis]
    pool.close()
    pool.join()

    res = [r.get() for r in run]
