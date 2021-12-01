import pandas as pd
import numpy as np

from numba import jit

import plotly
import plotly.graph_objects as go

import os

import gc

from datetime import datetime

from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from copy import deepcopy

@jit(nopython=False)
def ret_by_min(signal, vwap, close, percent=0):
    retarray = np.zeros(len(signal))
    previous = np.zeros(2)
    for s in range(len(signal)):
        retarray[s] = previous[0] * (vwap[s] - previous[1]) \
                      + signal[s] * (close[s] - vwap[s]) \
                      - abs(signal[s] - previous[0]) * vwap[s] * percent
        previous = np.array([signal[s], close[s]])
    return retarray

@jit(nopython=False)
def get_lot(signal, div):
    rate = np.abs(div/2/np.mean(np.abs(signal)))
    sigarray = np.zeros(len(signal))
    for s in range(len(signal)):
        now = signal[s] * rate
        if now > 0:
            sigarray[s] = np.floor(now)
        else:
            sigarray[s] = np.ceil(now)
    return sigarray

@jit(nopython=False)
def MaxDrawdown(return_list):
    maxi = np.zeros(len(return_list))
    maxi[0] = return_list[0]
    now = return_list[0]
    for i in range(len(return_list)):
        now = np.max([now, return_list[i]])
        #print(i, return_list[i])
        maxi[i] = now
    md = np.max(maxi-return_list)
    return np.mean(return_list)*242/md

class Tools():

    def __init__(self,  product='IC', exchange='cffex'):
        self.prod = product
        self.exchange = exchange

    #### data stats
    def startday(self, Prod =[]):
        dic = {}
        for p in Prod:
            df = self.qryHF("select ProductID, min(TradingDay) as td\
                                 from futures.minute_ext\
                                 where ProductID = '%s'\
                                 group by ProductID" % p)
            for i in df.iterrows():
                dic[i[1][0]] = i[1][1]
        return dic

    #### get_ret
    def get_ret(self, data, forward=[], use_vwap = True):
        if use_vwap:
            for f in forward:
                vwap = np.array(data.vwap)
                ret = np.zeros(len(vwap))
                ret[:-f] = (vwap[f:] - vwap[:-f]) / vwap[:-f]
                data['ret_%s' % f] = ret
        else:
            for f in forward:
                close = np.array(data.closewmp)
                ret = np.zeros(len(close))
                if f == 1:
                    ret[1:] = (close[1:] - close[:-1]) / close[:-1]
                    data['ret_%s' % f] = ret
                else:
                    ret[1:1 - f] = (close[f:] - close[:-f]) / close[:-f]
                    data['ret_%s' % f] = ret
        return data

    #### evaluation
    def quantile_map(self, df, combination, save_path, num = 20):
        pltdata = []
        for c in combination:
            current = df.sort_values(by=[c[0]], ascending=[True])
            sig = [current[c[0]][i * int(len(current[c[0]]) / num):(i + 1) * int(len(current[c[0]]) / num)]
                   for i in range(num)]
            ret = [current[c[1]][i * int(len(current[c[1]]) / num):(i + 1) * int(len(current[c[1]]) / num)]
                   for i in range(num)]
            pltdata.append(go.Scatter(x=[sum(i) / len(i) for i in sig],
                                      y=[sum(i) / len(i) for i in ret],
                                      mode='lines+markers', name="%s-%s"%(c[0],c[1])))
        plotly.offline.plot(pltdata, filename=save_path, auto_open=False)
        return(pltdata)

    def get_min_ret(self, df, siglis, save_path=False):
        if save_path:
            pltdata = []
            for s in siglis:
                retarr = ret_by_min(np.array(df[s]),
                                     np.array(df['vwap']),
                                     np.array(df['closewmp']))
                df[s+'_ret'] = retarr
                pltdata.append(go.Scatter(x=[i for i in range(len(retarr[::30]))],
                                          y=np.add.accumulate(retarr)[::30],
                                          mode='lines+markers', name=s))
            plotly.offline.plot(pltdata, filename=save_path, auto_open=False)
        else:
            for s in siglis:
                retarr = ret_by_min(np.array(df[s]),
                                     np.array(df['vwap']),
                                     np.array(df['closewmp']))
                df[s+'_ret'] = retarr
        return df

    def get_quantile_ret(self, df, siglis, div = 10, save_path=False):
        if save_path:
            pltdata = []
            for s in siglis:
                df[s] = get_lot(np.array(df[s]), div)
                retarr = ret_by_min(np.array(df[s]),
                                    np.array(df['vwap']),
                                    np.array(df['closewmp']))
                df[s+'_ret'] = retarr
                pltdata.append(go.Scatter(x=[i for i in range(len(retarr[::30]))],
                                          y=np.add.accumulate(retarr)[::30],
                                          mode='lines+markers', name=s))
            plotly.offline.plot(pltdata, filename=save_path, auto_open=False)
        else:
            for s in siglis:
                df[s] = get_lot(np.array(df[s]), div)
                retarr = ret_by_min(np.array(df[s]),
                                     np.array(df['vwap']),
                                     np.array(df['closewmp']))
                df[s+'_ret'] = retarr
        return df

    def ret_analysis(self, df, siglis, retlis, save_path, get_daily=False, lot=False, produce_plot=True):
        pltdata = []
        pltlot = []
        td = df.TradingDay.unique()
        sharp = {}
        md = {}
        ret_by_lot = {}
        for r in retlis:
            print(r)
            sig = np.array(df[siglis[retlis.index(r)]])
            shift = np.zeros(len(sig))
            shift[1:] = sig[:-1]
            ratio = (np.array(df.vwap)[1:] - np.array(df.vwap)[:-1]) / np.array(df.vwap)[:-1]
            ret_by_lot[r] = np.sum(sig[-1] * ratio) / (np.sum(np.abs(shift-sig))+np.abs(sig[0])+np.abs(sig[-1]))
            daily = []
            daily_lot = []
            for d in td:
                summy = sum(df.loc[df.TradingDay == d, r])
                daily.append(summy)
                if lot:
                    daily_lot.append(summy/
                             (np.dot(df.loc[df.TradingDay == d, 'vwap'],
                              df.loc[df.TradingDay == d, 'volume'])/
                              np.sum(df.loc[df.TradingDay == d, 'volume'])))

            daily = np.nan_to_num(np.array(daily))
            md[r] = MaxDrawdown(daily)
            print(md)
            sharp[r] = np.mean(daily) / np.std(daily) * 15.5
            pltdata.append(go.Scatter(x=list(td),
                                      y=np.add.accumulate(daily),
                                      mode='lines+markers', name=r))

            if lot:
                daily_lot = np.nan_to_num(np.array(daily_lot))
                pltlot.append(go.Scatter(x=list(td),
                                      y=np.add.accumulate(daily_lot),
                                      mode='lines+markers', name=r))

        start = list(df.loc[df.TradingDay == td[0], 'closewmp'])[-1]
        daily = []
        for d in td:
            daily.append(list(df.loc[df.TradingDay == d, 'closewmp'])[-1]-start)

        pltdata.append(go.Scatter(x=list(td),
                                  y=daily,
                                  mode='lines+markers', name='pricewave', xaxis='x', yaxis='y2'))

        if lot:
            pltlot.append(go.Scatter(x=list(td),
                                  y=daily,
                                  mode='lines+markers', name='pricewave', xaxis='x', yaxis='y2'))

        layout = go.Layout({"yaxis": {"title": {"text": ""}},
                            "yaxis2": {'anchor': 'x', "overlaying": 'y', "side": 'right'}})

        if lot:
            fig = go.Figure(pltlot, layout=layout)
            fig.write_html(save_path+'_daily_lot.html')
        if produce_plot:
            fig = go.Figure(pltdata, layout=layout)
            fig.write_html(save_path+'_daily.html')

        if get_daily:
            return sharp, ret_by_lot, md, daily
        else:
            return sharp, ret_by_lot, md




