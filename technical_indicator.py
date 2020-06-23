# ============================================================================
# Technical indicator
# Author - KM
# =============================================================================
import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import copy
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

class TechnicalIndicator:
    
    def __init__(self):
        print("Initializing Technical Indicator class......")
    

    def slope(self,ser,n):
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
    
    def ATR(self,DF,n):
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
    
    def ATRc(self,DF,n):
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
    
    def renko_DF(self,DF):
        "function to convert ohlc data into renko bricks"
        df = DF.copy()
        df.reset_index(inplace=True)
        df = df.iloc[:,[0,1,2,3,4,5]]
        df.columns = ["date","open","high","low","close","volume"]
        df2 = Renko(df)
        df2.brick_size = max(0.5,round(self.ATR(DF,120)["ATR"][-1],0))
        renko_df = df2.get_ohlc_data()
        renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
        for i in range(1,len(renko_df["bar_num"])):
            if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
                renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
            elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
                renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
        return renko_df
    
    def renko_merge(self,DF):
        "function to merging renko df with original ohlc df"
        df = copy.deepcopy(DF)
        df["Date"] = df.index
        renko = self.renko_DF(df)
        renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
        merged_df = df.merge(renko.loc[:,["Date","bar_num"]],how="outer",on="Date")
        merged_df["bar_num"].fillna(method='ffill',inplace=True)
        merged_df["macd"]= self.MACD(merged_df,12,26,9)[0]
        merged_df["macd_sig"]= self.MACD(merged_df,12,26,9)[1]
        merged_df["macd_slope"] = self.slope(merged_df["macd"],5)
        merged_df["macd_sig_slope"] = self.slope(merged_df["macd_sig"],5)
        return merged_df
    
    def OBV(self,DF):
        """function to calculate On Balance Volume"""
        df = DF.copy()
        df['daily_ret'] = df['Adj Close'].pct_change()
        df['direction'] = np.where(df['daily_ret']>=0,1,-1)
        df['direction'][0] = 0
        df['vol_adj'] = df['Volume'] * df['direction']
        df['obv'] = df['vol_adj'].cumsum()
        return df['obv']
    
    def CAGR(self,DF):
        "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
        df = DF.copy()
        df["cum_return"] = (1 + df["ret"]).cumprod()
        n = len(df)/(252*78) #1 trading day has 78 5 mins section
        CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
        return CAGR
    
    def volatility(self,DF):
        "function to calculate annualized volatility of a trading strategy"
        df = DF.copy()
        vol = df["ret"].std() * np.sqrt(252*78) #1 trading day has 78 5 mins section
        return vol
    
    def sharpe(self,DF,rf):
        "function to calculate sharpe ratio ; rf is the risk free rate"
        df = DF.copy()
        sr = (self.CAGR(df) - rf)/self.volatility(df)
        return sr
        
    
    def max_dd(self,DF):
        "function to calculate max drawdown"
        df = DF.copy()
        df["cum_return"] = (1 + df["ret"]).cumprod()
        df["cum_roll_max"] = df["cum_return"].cummax()
        df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
        df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
        max_dd = df["drawdown_pct"].max()
        return max_dd
    
    def MACD(self,DF,a,b,c):
        """function to calculate MACD typical values a = 12; b =26, c =9"""
        df = DF.copy()
        df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
        df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
        df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
        df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
        df.dropna(inplace=True)
        return (df["MACD"],df["Signal"])
    
    def pflio(self,DF,m,x):
        """Returns cumulative portfolio return
        DF = dataframe with monthly return info for all stocks
        m = number of stock in the portfolio
        x = number of underperforming stocks to be removed from portfolio monthly"""
        df = DF.copy()
        portfolio = []
        ret = [0] #the %change will be 0 for the first row
        for i in range(1,len(df)):
            if len(portfolio) > 0:
                ret.append(df[portfolio].iloc[i,:].mean())
                bad_stocks = df[portfolio].iloc[i,:].sort_values(ascending=True)[:x].index.values.tolist()
                portfolio = [t for t in portfolio if t not in bad_stocks]
            fill = m - len(portfolio)
            new_picks = df.iloc[i,:].sort_values(ascending=False)[:fill].index.values.tolist()
            portfolio = portfolio + new_picks
            #print(portfolio) #only for the strategy return is bigger than the index return.
        ret_df = pd.DataFrame(np.array(ret),columns=["ret"])
        return ret_df
        
    def merging_renko(self,tickers,ti,df,ohlc_renko,tickers_signal,tickers_ret):
        for ticker in tickers:
            print("merging for ",ticker)
            renko = self.renko_DF(df[ticker])
            renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
            df[ticker]["Date"] = df[ticker].index
            ohlc_renko[ticker] = df[ticker].merge(renko.loc[:,["Date","bar_num"]],how="outer",on="Date")
            ohlc_renko[ticker]["bar_num"].fillna(method='ffill',inplace=True)
            ohlc_renko[ticker]["macd"]= self.MACD(ohlc_renko[ticker],12,26,9)[0]
            ohlc_renko[ticker]["macd_sig"]= self.MACD(ohlc_renko[ticker],12,26,9)[1]
            ohlc_renko[ticker]["macd_slope"] = self.slope(ohlc_renko[ticker]["macd"],5)
            ohlc_renko[ticker]["macd_sig_slope"] = self.slope(ohlc_renko[ticker]["macd_sig"],5)
            tickers_signal[ticker] = ""
            tickers_ret[ticker] = []
    
    def mergining_renko_obv(self,tickers,ti,df,ohlc_renko,tickers_signal,tickers_ret):
        for ticker in tickers:
            print("merging for ",ticker)
            renko = self.renko_DF(df[ticker])
            renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
            df[ticker]["Date"] = df[ticker].index
            ohlc_renko[ticker] = df[ticker].merge(renko.loc[:,["Date","bar_num"]],how="outer",on="Date")
            ohlc_renko[ticker]["bar_num"].fillna(method='ffill',inplace=True)
            ohlc_renko[ticker]["obv"]= self.OBV(ohlc_renko[ticker])
            ohlc_renko[ticker]["obv_slope"]= self.slope(ohlc_renko[ticker]["obv"],5)
            tickers_signal[ticker] = ""
            tickers_ret[ticker] = []
            
    def signal_return(self,tickers, ohlc_intraday, tickers_signal, tickers_ret, ohlc_renko):
        #Identifying signals and calculating daily return
        for ticker in tickers:
            print("calculating daily returns for ",ticker)
            for i in range(len(ohlc_intraday[ticker])):
                if tickers_signal[ticker] == "":
                    tickers_ret[ticker].append(0)
                    if i > 0:
                        if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["macd"][i]>ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]>ohlc_renko[ticker]["macd_sig_slope"][i]:
                            tickers_signal[ticker] = "Buy"
                        elif ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["macd"][i]<ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]<ohlc_renko[ticker]["macd_sig_slope"][i]:
                            tickers_signal[ticker] = "Sell"
                
                elif tickers_signal[ticker] == "Buy":
                    tickers_ret[ticker].append((ohlc_renko[ticker]["Adj Close"][i]/ohlc_renko[ticker]["Adj Close"][i-1])-1)
                    if i > 0:
                        if ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["macd"][i]<ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]<ohlc_renko[ticker]["macd_sig_slope"][i]:
                            tickers_signal[ticker] = "Sell"
                        elif ohlc_renko[ticker]["macd"][i]<ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]<ohlc_renko[ticker]["macd_sig_slope"][i]:
                            tickers_signal[ticker] = ""
                        
                elif tickers_signal[ticker] == "Sell":
                    tickers_ret[ticker].append((ohlc_renko[ticker]["Adj Close"][i-1]/ohlc_renko[ticker]["Adj Close"][i])-1)
                    if i > 0:
                        if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["macd"][i]>ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]>ohlc_renko[ticker]["macd_sig_slope"][i]:
                            tickers_signal[ticker] = "Buy"
                        elif ohlc_renko[ticker]["macd"][i]>ohlc_renko[ticker]["macd_sig"][i] and ohlc_renko[ticker]["macd_slope"][i]>ohlc_renko[ticker]["macd_sig_slope"][i]:
                            tickers_signal[ticker] = ""
            ohlc_renko[ticker]["ret"] = np.array(tickers_ret[ticker])
            
    def signal_obv_renko(self,tickers, ohlc_intraday, ohlc_renko, tickers_signal, tickers_ret):
        #Identifying signals and calculating daily return
        for ticker in tickers:
            print("calculating daily returns for ",ticker)
            for i in range(len(ohlc_intraday[ticker])):
                if tickers_signal[ticker] == "":
                    tickers_ret[ticker].append(0)
                    if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["obv_slope"][i]>30:
                        tickers_signal[ticker] = "Buy"
                    elif ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["obv_slope"][i]<-30:
                        tickers_signal[ticker] = "Sell"
                
                elif tickers_signal[ticker] == "Buy":
                    tickers_ret[ticker].append((ohlc_renko[ticker]["Adj Close"][i]/ohlc_renko[ticker]["Adj Close"][i-1])-1)
                    if ohlc_renko[ticker]["bar_num"][i]<=-2 and ohlc_renko[ticker]["obv_slope"][i]<-30:
                        tickers_signal[ticker] = "Sell"
                    elif ohlc_renko[ticker]["bar_num"][i]<2:
                        tickers_signal[ticker] = ""
                        
                elif tickers_signal[ticker] == "Sell":
                    tickers_ret[ticker].append((ohlc_renko[ticker]["Adj Close"][i-1]/ohlc_renko[ticker]["Adj Close"][i])-1)
                    if ohlc_renko[ticker]["bar_num"][i]>=2 and ohlc_renko[ticker]["obv_slope"][i]>30:
                        tickers_signal[ticker] = "Buy"
                    elif ohlc_renko[ticker]["bar_num"][i]>-2:
                        tickers_signal[ticker] = ""
            ohlc_renko[ticker]["ret"] = np.array(tickers_ret[ticker])
            
    def KPIs(self,ti, tickers, ohlc_renko):
        #calculating overall strategy's KPIs
        strategy_df = pd.DataFrame()
        for ticker in tickers:
            strategy_df[ticker] = ohlc_renko[ticker]["ret"]
        strategy_df["ret"] = strategy_df.mean(axis=1)
        print("Total CAGR: {}".format(self.CAGR(strategy_df)))
        print("Total sharpe ratio: {}".format(self.sharpe(strategy_df,0.025)))
        print("Total max drawdown: {}".format(self.max_dd(strategy_df)))
        
        #visualizing strategy returns
        plt.plot((1+strategy_df["ret"]).cumprod())
        plt.title('overall %return summary')
        plt.ylabel('%return')
        plt.xlabel('#of trade')
        plt.show()
    
        #calculating individual stock's KPIs
        cagr = {}
        sharpe_ratios = {}
        max_drawdown = {}
        for ticker in tickers:
            print("calculating KPIs for ",ticker)      
            cagr[ticker] =  self.CAGR(ohlc_renko[ticker])
            sharpe_ratios[ticker] =  self.sharpe(ohlc_renko[ticker],0.025)
            max_drawdown[ticker] =  self.max_dd(ohlc_renko[ticker])
        
        
        KPI_df = pd.DataFrame([cagr,sharpe_ratios,max_drawdown],index=["Return","Sharpe Ratio","Max Drawdown"])      
        print(KPI_df.T)
        print('DONE\n')
        print()
    
            
    def mReturn(self,tickers, ohlc_dict, return_df):
        for ticker in tickers:
            print("calculating return for ",ticker)
            ohlc_dict[ticker]["ret"] = ohlc_dict[ticker]["Adj Close"].pct_change()
            return_df[ticker] = ohlc_dict[ticker]["ret"]
            
    def balancing_SP500(self,ti):
        sp = yf.Ticker('^GSPC').history(period='1mo', interval='5m',actions=False)
        sp.columns = ["Open","High","Low","Adj Close","Volume"]
        sp.index = sp.index.tz_convert('US/Mountain').tz_localize(None)
        sp['ret'] = sp["Adj Close"].pct_change()
        print("CAGR for SP500: {}".format(self.CAGR(sp)))
        print("Total sharpe ratio for sp500: {}".format(self.sharpe(sp,0.025)))
        print("Total max drawdown for sp500: {}".format(self.max_dd(sp)))
        return sp
        
    def visBalancMethod(self,return_df,ti,sp):
        #visualization
        fig, ax = plt.subplots()
        plt.plot((1+self.pflio(return_df,6,3)).cumprod())
        plt.plot((1+sp["ret"][2:].reset_index(drop=True)).cumprod())
        plt.title("Index Return vs Strategy Return")
        plt.ylabel("cumulative return")
        plt.xlabel("trade")
        ax.legend(["Strategy Return","Index Return"])
        plt.show()
        
    def breakT(self,tickers, ohlc_dict, tickers_signal, tickers_ret, ti):
        for ticker in tickers:
            print("calculating ATR and rolling max price for ",ticker)
            #print(ohlc_dict[ticker])
            ohlc_dict[ticker]["ATR"] = self.ATRc(ohlc_dict[ticker], 20)
            ohlc_dict[ticker]["roll_max_cp"] = ohlc_dict[ticker]["High"].rolling(20).max()
            ohlc_dict[ticker]["roll_min_cp"] = ohlc_dict[ticker]["Low"].rolling(20).min()
            ohlc_dict[ticker]["roll_max_vol"] = ohlc_dict[ticker]["Volume"].rolling(20).max()
            ohlc_dict[ticker].dropna(inplace=True)
            tickers_signal[ticker] = ""
            tickers_ret[ticker] = []
            
    def breakTSignal(self,tickers, ohlc_dict, tickers_signal, tickers_ret, ti):
        # identifying signals and calculating daily return (stop loss factored in)
        for ticker in tickers:
            print("calculating returns for ",ticker)
            for i in range(len(ohlc_dict[ticker])):
                if tickers_signal[ticker] == "":
                    tickers_ret[ticker].append(0)
                    if ohlc_dict[ticker]["High"][i]>=ohlc_dict[ticker]["roll_max_cp"][i] and \
                       ohlc_dict[ticker]["Volume"][i]>1.5*ohlc_dict[ticker]["roll_max_vol"][i-1]:
                        tickers_signal[ticker] = "Buy"
                    elif ohlc_dict[ticker]["Low"][i]<=ohlc_dict[ticker]["roll_min_cp"][i] and \
                       ohlc_dict[ticker]["Volume"][i]>1.5*ohlc_dict[ticker]["roll_max_vol"][i-1]:
                        tickers_signal[ticker] = "Sell"
                
                elif tickers_signal[ticker] == "Buy":
                    if ohlc_dict[ticker]["Adj Close"][i]<ohlc_dict[ticker]["Adj Close"][i-1] - ohlc_dict[ticker]["ATR"][i-1]:
                        tickers_signal[ticker] = ""
                        #stop lost of %return
                        tickers_ret[ticker].append(((ohlc_dict[ticker]["Adj Close"][i-1] - ohlc_dict[ticker]["ATR"][i-1])/ohlc_dict[ticker]["Adj Close"][i-1])-1)
                    #if it's buy signal and now go below the supporting point, then change to sell signal
                    elif ohlc_dict[ticker]["Low"][i]<=ohlc_dict[ticker]["roll_min_cp"][i] and \
                       ohlc_dict[ticker]["Volume"][i]>1.5*ohlc_dict[ticker]["roll_max_vol"][i-1]:
                        tickers_signal[ticker] = "Sell"
                        tickers_ret[ticker].append(((ohlc_dict[ticker]["Adj Close"][i-1] - ohlc_dict[ticker]["ATR"][i-1])/ohlc_dict[ticker]["Adj Close"][i-1])-1)
                    else:
                        tickers_ret[ticker].append((ohlc_dict[ticker]["Adj Close"][i]/ohlc_dict[ticker]["Adj Close"][i-1])-1)
                        
                elif tickers_signal[ticker] == "Sell":
                    if ohlc_dict[ticker]["Adj Close"][i]>ohlc_dict[ticker]["Adj Close"][i-1] + ohlc_dict[ticker]["ATR"][i-1]:
                        tickers_signal[ticker] = ""
                        tickers_ret[ticker].append((ohlc_dict[ticker]["Adj Close"][i-1]/(ohlc_dict[ticker]["Adj Close"][i-1] + ohlc_dict[ticker]["ATR"][i-1]))-1)
                    elif ohlc_dict[ticker]["High"][i]>=ohlc_dict[ticker]["roll_max_cp"][i] and \
                       ohlc_dict[ticker]["Volume"][i]>1.5*ohlc_dict[ticker]["roll_max_vol"][i-1]:
                        tickers_signal[ticker] = "Buy"
                        tickers_ret[ticker].append((ohlc_dict[ticker]["Adj Close"][i-1]/(ohlc_dict[ticker]["Adj Close"][i-1] + ohlc_dict[ticker]["ATR"][i-1]))-1)
                    else:
                        tickers_ret[ticker].append((ohlc_dict[ticker]["Adj Close"][i-1]/ohlc_dict[ticker]["Adj Close"][i])-1)
                        
            ohlc_dict[ticker]["ret"] = np.array(tickers_ret[ticker])
        
            