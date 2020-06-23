# ============================================================================
# Back Testing Robot
# Author - KM
# =============================================================================
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
from lstm import LSTMml
from initData import InitData

#############################################################

def main():
    print("Back-Testing is running......\n")
    mylist = []
    ohlc_intraday = {} # directory with ohlc value for each stock    
    tickers_signal = {}
    tickers_ret = {} #return
    ti = TechnicalIndicator()
    impData = InitData()
    userStockInput = ""
    
    user_sp = input("Do you want to load all default sotcks?: y / n: ")
    if user_sp == 'y' or user_sp == 'Y':
        tickers = impData.init_stocks(impData.load_SP500())
    elif user_sp == 'n' or user_sp == 'N':
        while (userStockInput != 'done' and userStockInput != 'Done'):
            userStockInput = input("Type in the stock symbol (type done to quit):")
            if (userStockInput != 'done' and userStockInput != 'Done'):
                mylist.append(userStockInput)
        tickers = impData.init_stocks(mylist)
    impData.get_stock_data(ohlc_intraday,tickers)
    tickers = ohlc_intraday.keys() #redefine tickers variable after removing any tickers with corrupted data
    impData.time_zone_convert(ohlc_intraday,tickers) # convert time zone to MDT

    #####################BACKTESTING##############
    print("Back-Testing System is reday......\n")
    user_backTest_input = ""
    while (user_backTest_input != 'done' and user_backTest_input != 'Done'):
        print("Back-Test Option:")
        print("1. Renko + MACD")
        print("2. Portfolio Rebalancing")
        print("3. Resistance Breakout")
        print("4. Renko + OBV")
        print('\nValue Investment Info:')
        print("5. Joel Greenblatt's Magic Formular")
        print('6. Piotroski F-Score')
        print('\nMachine Learning Algorithms:')
        print('7. LSTM model: ')
        user_backTest_input = input("Please enter the number (type done to quit):")
        
        if user_backTest_input == '1':
            ohlc_renko = {}
            df = copy.deepcopy(ohlc_intraday)
            #Merging renko df with original ohlc df
            ti.merging_renko(tickers,ti,df,ohlc_renko,tickers_signal,tickers_ret)
            ti.signal_return(tickers, ohlc_intraday, tickers_signal, tickers_ret, ohlc_renko)
            ti.KPIs(ti, tickers, ohlc_renko)
            
        elif user_backTest_input == '2':
            # calculating return for each stock and consolidating return info by stock in a separate dataframe
            ohlc_dict = copy.deepcopy(ohlc_intraday)
            return_df = pd.DataFrame()
            ti.mReturn(tickers, ohlc_dict, return_df)
            #calculating overall strategy's KPIs
            print("Total CAGR: {}".format(ti.CAGR(ti.pflio(return_df,6,3))))
            print("Total sharpe ratio: {}".format(ti.sharpe(ti.pflio(return_df,6,3),0.025)))
            print("Total max drawdown: {}".format(ti.max_dd(ti.pflio(return_df,6,3))))
            #calculating KPIs for Index buy and hold strategy over the same period
            sp = ti.balancing_SP500(ti)
            ti.visBalancMethod(return_df,ti,sp)
            
        elif user_backTest_input == '3':
            ohlc_dict = copy.deepcopy(ohlc_intraday)
            tickers_signal = {}
            tickers_ret = {}
            ti.breakT(tickers, ohlc_dict, tickers_signal, tickers_ret, ti)
            ti.breakTSignal(tickers, ohlc_dict, tickers_signal, tickers_ret, ti)
            ti.KPIs(ti, tickers, ohlc_dict)
        
        elif user_backTest_input == '4':
            ohlc_renko = {}
            df = copy.deepcopy(ohlc_intraday)
            tickers_signal = {}
            tickers_ret = {}
            ti.mergining_renko_obv(tickers,ti,df,ohlc_renko,tickers_signal, tickers_ret)
            ti.signal_obv_renko(tickers, ohlc_intraday, ohlc_renko, tickers_signal, tickers_ret)
            ti.KPIs(ti,tickers,ohlc_renko)
        
        elif user_backTest_input == '5':
            mg = MagicFormular(tickers)
            mg.dataHandling()
        
        elif  user_backTest_input == '6':
            f = FScore(tickers)
            f.dataHandling()
            
        elif user_backTest_input == '7':
            lstm_ticker = []
            command = ""
            while command != 'done':
                print("For LSTM, you are allowed to do one stock at a time.")
                command = input("Please enter the stock ticker that you want to back-test (or type 'done' to quit): ")
                if command != 'done':
                    lstm_ticker.append(command)
                    lstm_intraday = {}
                    impData.get_stock_data(lstm_intraday,lstm_ticker)
                    impData.time_zone_convert(lstm_intraday,lstm_ticker)
                    lstm_intraday[command]['Date'] = lstm_intraday[command].index
                    lstm = LSTMml(lstm_intraday[command], command)
                    #type(lstm_intraday[command])
                    lstm.computing()
                else:
                    break
        
        elif user_backTest_input == 'q' or user_backTest_input == 'Q':
            print('Goodbye!')
            break
        else:
            print("Please enter 1 - 4 or q to q\n")
            

if __name__ == "__main__":
    main()
    