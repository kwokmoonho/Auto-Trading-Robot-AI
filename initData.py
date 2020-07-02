#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:22:52 2020

@author: kwokmoonho
"""
import yfinance as yf
import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import time
import copy
import pandas as pd
from technical_indicator import TechnicalIndicator
import pandas_datareader.data as pdr
import mplfinance
import matplotlib.dates as mdates
from matplotlib import style
import datetime as dt
from magicFormular import MagicFormular
from fScore import FScore
from lstm import LSTM

class InitData:
    
    def __init__(self):
        pass
    
    def init_stocks(self,mylist):
        print("Initializing the stock list......")
        return mylist
    
    def get_stock_data(self,ohlc_intraday,tickers):
        print('Getting stocks data...')
        drop = [] # initializing list to store tickers whose close price was successfully extracted
        while len(tickers) != 0:
            tickers = [j for j in tickers if j not in drop]
            for i in range(len(tickers)):
                try:
                    ohlc_intraday[tickers[i]] = yf.Ticker(tickers[i]).history(period='1mo', interval='5m',actions=False)
                    ohlc_intraday[tickers[i]].columns = ["Open","High","Low","Adj Close","Volume"]
                    drop.append(tickers[i])    
                except:
                    print(tickers[i]," :failed to fetch data...deleting invalid ticker.....\n")
                    ohlc_intraday.pop(tickers[i])
                    tickers.pop(i)

            
    def time_zone_convert(self,ohlc_intraday, tickers):
        for s in tickers:
            ohlc_intraday[s].index = ohlc_intraday[s].index.tz_convert('US/Mountain').tz_localize(None)
    
        
    def load_SP500(self):
        print("Using default symbols....")
        SP500_symbol = ['AAPL','HD','FB','V','BABA','VZ','WMT','T','DXCM','ROST','JNJ','AMZN', 'MMM']
        return SP500_symbol

