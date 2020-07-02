# ============================================================================
# Auto Trading Script
# Author - KM
# =============================================================================
import alpaca_trade_api as tradeapi
import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import time
import copy
import yfinance as yf
from lstm import LSTMml
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
#############################################################

def api_connection():
    print("Connecting to Alpaca trading account......")
    #setting api connection
    token = open('key.txt','r').read().splitlines()
    con = tradeapi.REST(token[0], token[1],token[2])
    #printing out account information
    #print("Account Summary:")
    #print(con.get_account())
    print()
    return con

def get_stock_data(ohlc_intraday,tickers):
    attempt = 0 # initializing passthrough variable
    drop = [] # initializing list to store tickers whose close price was successfully extracted
    while len(tickers) != 0 and attempt <=300:
        tickers = [j for j in tickers if j not in drop]
        for i in range(len(tickers)):
            try:
                ohlc_intraday[tickers[i]] = yf.Ticker(tickers[i]).history(period='1mo', interval='5m',actions=False)
                ohlc_intraday[tickers[i]].columns = ["Open","High","Low","Adj Close","Volume"]
                drop.append(tickers[i])    
            except:
                print(tickers[i]," :failed to fetch data...retrying")
                continue
        attempt+=1
        
def time_zone_convert(ohlc_intraday, tickers):
    for s in tickers:
        ohlc_intraday[s].index = ohlc_intraday[s].index.tz_convert('US/Mountain').tz_localize(None)
        
def MACD(DF,a,b,c):
    """function to calculate MACD typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    return (df["MACD"],df["Signal"])

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def slope(ser,n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df)
    df2.brick_size = max(0.5,round(ATR(DF,120)["ATR"][-1],0))
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    return renko_df

def renko_merge(DF):
    "function to merging renko df with original ohlc df"
    df = copy.deepcopy(DF)
    df["Date"] = df.index
    renko = renko_DF(df)
    renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
    merged_df = df.merge(renko.loc[:,["Date","bar_num"]],how="outer",on="Date")
    merged_df["bar_num"].fillna(method='ffill',inplace=True)
    merged_df["macd"]= MACD(merged_df,12,26,9)[0]
    merged_df["macd_sig"]= MACD(merged_df,12,26,9)[1]
    merged_df["macd_slope"] = slope(merged_df["macd"],5)
    merged_df["macd_sig_slope"] = slope(merged_df["macd_sig"],5)
    return merged_df

def trade_signal(MERGED_DF,l_s):
    "function to generate signal"
    signal = ""
    df = copy.deepcopy(MERGED_DF)
    #df = rm
    #l_s = long_short
    if l_s == "":
        if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Buy"
        elif df["bar_num"].tolist()[-1]<=-2 and df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Sell"
            
    elif l_s == "long":
        if df["bar_num"].tolist()[-1]<=-2 and df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Close_Sell"
        elif df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Close"
            
    elif l_s == "short":
        if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Close_Buy"
        elif df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Close"
    return signal

def ATRc(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2['ATR']


def breakT(ohlc_dict):
    df = copy.deepcopy(ohlc_dict)
    print("calculating ATR and rolling max price")
    df["ATR"] = ATRc(df, 20)
    df["roll_max_cp"] = df["High"].rolling(20).max()
    df["roll_min_cp"] = df["Low"].rolling(20).min()
    df["roll_max_vol"] = df["Volume"].rolling(20).max()
    df.dropna(inplace=True)
    return df

#predict the close price base on the lstm obj
def lstm_predict(l,ohlc_dict):
    df = copy.deepcopy(ohlc_dict)
    df['Date'] = ohlc_dict.index

    print("LSTM prediction for: ", l.tickers)
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Adj Close'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Adj Close'][i] = data['Adj Close'][i]
    #setting index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)
    dataset = new_data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    #predicting values, using past 60 from the train data
    inputs = new_data.values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)
    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)
    
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    prediction = l.model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    df['Prediction'] = np.nan
    for i in range(60,len(data)):
        df['Prediction'][i] = prediction[i-60]
        
    ##See the result:
    plt.figure(figsize=(16,8))
    plt.plot(df['Adj Close'], label='Test Data')
    plt.plot(df['Prediction'], label='Prediction')
    plt.xlabel("Date / Time")
    plt.ylabel("stock: " + l.tickers)
    plt.title("Prediction VS Real")
    plt.legend(title='Parameter where:')
    plt.show()        
    return df    

def breakTSignal(ohlc, l_s):
    # identifying signals and calculating daily return (stop loss factored in)
    signal = ""
    df = copy.deepcopy(ohlc)
    if l_s == "":
        if df['High'].tolist()[-2]>=df['roll_max_cp'].tolist()[-2] and df['Volume'].tolist()[-2]>1.5*df['roll_max_vol'].tolist()[-3]:
            signal = 'Buy'
        elif df['Low'].tolist()[-2]>=df['roll_min_cp'].tolist()[-2] and df['Volume'].tolist()[-2]>1.5*df['roll_max_vol'].tolist()[-3]:
            signal = 'Sell'
    elif l_s == "Buy":
        if df["Adj Close"].tolist()[-2]<df["Adj Close"].tolist()[-3] - df["ATR"].tolist()[-3]:
            #stop lost
            signal = "Close_Sell"
        #if it's buy signal and now go below the supporting point, then change to sell signal
        elif df["Low"].tolist()[-2]<=df["roll_min_cp"].tolist()[-2] and df["Volume"][-2]>1.5*df["roll_max_vol"][-3]:
            signal = 'Sell'
        else:
            pass
    elif l_s == 'Sell':
        if df["Adj Close"].tolist()[-2]>df["Adj Close"][-3] + df["ATR"][-3]:
            signal = "Close_Buy"
        elif df["High"][-2]>=df["roll_max_cp"][-2] and  df["Volume"][-2]>1.5*df["roll_max_vol"].tolist()[-3]:
            signal = 'Buy'
        else:
            pass
    return signal


def lstm_trade_signal(lstm_df, l_s):
    #lstm_df = lstm_predict(lstm_auto_list.get(stock), ohlc[stock])
    #l_s = long_short
    signal = ""
    df = copy.deepcopy(lstm_df)
    if l_s == "":
        if df["Prediction"].tolist()[-1] > df["Prediction"].tolist()[-2]:
            signal = "Buy"
            print()
            print(df["Prediction"].tolist()[-1])
            print(df["Prediction"].tolist()[-2])
            print()
        elif df["Prediction"].tolist()[-1] < df["Prediction"].tolist()[-2]:
            signal = 'Sell'
            print()
            print(df["Prediction"].tolist()[-1])
            print(df["Prediction"].tolist()[-2])
            print()
    elif l_s == "Buy":
        if df["Prediction"].tolist()[-1] < df["Prediction"].tolist()[-2]:
            #stop lost
            signal = "Close_Buy"
            print()
            print(df["Prediction"].tolist()[-1])
            print(df["Prediction"].tolist()[-2])
            print()
        else:
            print()
            print(df["Prediction"].tolist()[-1])
            print(df["Prediction"].tolist()[-2])
            print()
    elif l_s == 'Sell':
        if df["Prediction"].tolist()[-1] > df["Prediction"].tolist()[-2]:
            signal = "Close_Sell"
            print()
            print(df["Prediction"].tolist()[-1])
            print(df["Prediction"].tolist()[-2])
            print()
        else:
            print()
            print(df["Prediction"].tolist()[-1])
            print(df["Prediction"].tolist()[-2])
            print()
            
    return signal
        
        

def auto_trade(lstm_list):
    print("Auto-Testing Script is running......\n")
    lstm_auto_list = copy.deepcopy(lstm_list)
    con = api_connection()
    tickers_list = ['MSFT', 'HD', 'VIXY', 'SPXU','HD','ZM','V','FB','IBM','AAL']
    
    """
    auto trading method name
    """
    try:
        open_pos = con.list_positions()     #return a list of position
        for stock in tickers_list:
            print('Computing for stock :', stock)
            long_short = ""
            open_pos_sk = [sk for sk in open_pos if sk.symbol == stock]   #stock will change each time | list of position obj
            if len(open_pos) > 0:
                if len(open_pos_sk) > 0:    #if the list is not empty (handle case when stock is not on the position list)
                    print('Open Position stocks:', stock)
                    if open_pos_sk[0].side == 'long':
                        long_short = 'Buy'
                    elif open_pos_sk[0].side == 'short':
                        long_short = 'Sell'
            ohlc = {} # directory with ohlc value for each stock   
            tick = [stock] 
            get_stock_data(ohlc,tick)
            tickers = ohlc.keys() #redefine tickers variable after removing any tickers with corrupted data
            time_zone_convert(ohlc,tickers) # convert time zone to MDT
            
        
            ##using renko + macd
            #signal = trade_signal(renko_merge(ohlc[stock]), long_short)

            ##using resistance break out method
            #signal = breakTSignal(breakT(ohlc[stock]), long_short)
            
            
            ##using lstm
            
            #*********************************************************************************
            signal = lstm_trade_signal(lstm_predict(lstm_auto_list.get(stock), ohlc[stock]), long_short)
            print('Signal:',signal)
            if signal == 'Buy':
                if float(con.get_account().buying_power) > float(ohlc[stock].High.tolist()[-1] * 100) and float(con.get_account().daytrading_buying_power) > float(ohlc[stock].High.tolist()[-1] * 100):
                    con.submit_order(symbol = stock, qty = '100', side='buy',type='market',time_in_force='gtc')
                    print("New long position initiated for ", stock)
                    print()
                else:
                    print('Insufficient buying power or day trade power\n Buying power: ${}\n Day trade power: ${}\n'.format(con.get_account().buying_power, con.get_account().daytrading_buying_power))
            elif signal == 'Sell':
                if float(con.get_account().buying_power) > float(ohlc[stock].High.tolist()[-1] * 100) and float(con.get_account().daytrading_buying_power) > float(ohlc[stock].High.tolist()[-1] * 100):
                    con.submit_order(symbol = stock, qty = '100', side='sell',type='market',time_in_force='gtc')
                    print('New Short order initiated for ', stock)
                    print()
                else:
                     print('Insufficient buying power or day trade power\n Buying power: ${}\n Day trade power: ${}\n'.format(con.get_account().buying_power, con.get_account().daytrading_buying_power))
            elif signal == 'Close_Buy':
                con.submit_order(symbol = stock, qty = open_pos_sk[0].qty, side='sell',type='market',time_in_force='gtc')
                print('Existing Short position closed for ', stock)
                print()
            elif signal == 'Close_Sell':
                con.submit_order(symbol = stock, qty = int(open_pos_sk[0].qty) * -1, side='buy',type='market',time_in_force='gtc')
                print('Existing long position closed for ', stock)
                print()
            else:
                print('No action taken for ', stock)
                print()

    except: 
        print("error encountered....skipping this iteration\n")
        print()
        print()

# Continuous execution        
starttime=time.time()               #stock market open at 9:30am to 4:00pm Easten time on weekday
timeout = time.time() + 60*60*5.5  # 60 seconds times 60 meaning the script will run for 1 hr, * 5.5 will be 5.5hrs
print('Training LSTM data')
stock_list = ['MSFT', 'HD', 'VIXY', 'SPXU','HD','ZM','V','FB','IBM','AAL']
lstm_list = {}
for command in stock_list:
    print('Computin LSTM training for: ', command)
    lstm_intraday = {}
    get_stock_data(lstm_intraday,[command])
    time_zone_convert(lstm_intraday,[command])
    lstm_intraday[command]['Date'] = lstm_intraday[command].index
    lstm = LSTMml(lstm_intraday[command],command)
    lstm.computing()
    lstm_list[command] = lstm

while time.time() <= timeout:
    try:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        auto_trade(lstm_list)
        time.sleep(300 - ((time.time() - starttime) % 300.0)) # 5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()
