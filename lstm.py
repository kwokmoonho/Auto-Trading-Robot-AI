"""
@author: kwokmoonho
Stock prediction by using LSTM
"""

#import library
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import rcParams

class LSTMml:
    def __init__(self, df, tick):
        self.df = df
        self.tickers = tick
        self.model = Sequential()
    
    def computing(self):
        rcParams['figure.figsize'] = 20,10
        scaler = MinMaxScaler(feature_range=(0, 1))
        pd.options.mode.chained_assignment = None  # default='warn'
        
        #reading data
        #df = pd.read_csv('sp500.csv')
        #overlook the data
        #df.head()
        
        #setting index as date
        #self.df['Date'] = pd.to_datetime(self.df.Datetime,format='%Y-%m-%d')
        #self.df.index = self.df['Date']
        
        #plot
        plt.figure(figsize=(16,8))
        plt.plot(self.df['Adj Close'], label='Close Price history')
        plt.xlabel("Date / Time")
        plt.title("Date overview")
        plt.ylabel("stock: " + self.tickers)
        plt.show()
        
        """
        LSTM
        """
        #creating dataframe
        data = self.df.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(0,len(self.df)),columns=['Date', 'Adj Close'])
        for i in range(0,len(data)):
            new_data['Date'][i] = data['Date'][i]
            new_data['Adj Close'][i] = data['Adj Close'][i]
        
        #setting index
        new_data.index = new_data.Date
        new_data.drop('Date', axis=1, inplace=True)
        
        #creating train and test sets
        dataset = new_data.values
        
        #80% for train
        row80P = int(len(self.df)*0.8)
        train = dataset[0:row80P,:]
        valid = dataset[row80P:,:]
        
        #converting dataset into x_train and y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        x_train, y_train = [], []
        #using past 60 data to predict next data
        for i in range(60,len(train)):
            x_train.append(scaled_data[i-60:i,0])
            y_train.append(scaled_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
        
        """
        #check for best units
        
        #myRMS = []
        #using for loop to check the best units value, so far 57 is the best for sp500 index
        #for p in range (40,60):
        model = Sequential()
        model.add(LSTM(units=57, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(units=57))
        model.add(Dense(1))
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)
        
        #predicting values, using past 60 from the train data
        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)
        
        X_test = []
        for i in range(60,inputs.shape[0]):
            X_test.append(inputs[i-60:i,0])
        X_test = np.array(X_test)
        
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)
        
        rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
        print("RMS for the LSTM is :{}".format(rms))
        #myRMS.append(rms)
        #print(rms)
        """
            
        """
        print("Dimensionality of the output space for different units values:")
        for i in range (len(myRMS)):
            print("units = {} , rms = {}".format(40+i,myRMS[i]))
        """
        # create and fit the LSTM network
        self.model.add(LSTM(units=57, return_sequences=True, input_shape=(x_train.shape[1],1))) #units 57 base on sp500 index result
        self.model.add(LSTM(units=57))
        self.model.add(Dense(1))
        
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)
        
        #predicting values, using past 60 from the train data
        inputs = new_data[len(new_data) - len(valid) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)
        
        X_test = []
        for i in range(60,inputs.shape[0]):
            X_test.append(inputs[i-60:i,0])
        X_test = np.array(X_test)
        
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = self.model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)
        
        rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
        print("RMS for the LSTM is :{}".format(rms))
        print()
        
        #plotting
        train = new_data[:row80P]
        valid = new_data[row80P:]
        valid['Predictions'] = closing_price
        plt.figure(figsize=(16,8))
        plt.plot(train['Adj Close'], label='Train Data')
        plt.plot(valid['Adj Close'], label='Test Data')
        plt.plot(valid['Predictions'], label='Prediction')
        plt.xlabel("Date / Time")
        plt.title("LSTM Result")
        plt.ylabel("stock: " + self.tickers)
        plt.legend(title='Parameter where:')
        plt.show()
        
        #zoom in
        plt.figure(figsize=(16,8))
        plt.plot(valid['Adj Close'], label='Test Data')
        plt.plot(valid['Predictions'], label='Prediction')
        plt.xlabel("Date / Time")
        plt.ylabel("stock: " + self.tickers)
        plt.title("Zoom in the test result")
        plt.legend(title='Parameter where:')
        plt.show()
