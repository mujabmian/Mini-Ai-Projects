import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

stock = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
stock['Date'] = stock.index

X = stock.index.factorize()[0].reshape(-1,1)
y = stock['Close'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(stock['Date'].iloc[len(X_train):], y_test, label='Actual')
plt.plot(stock['Date'].iloc[len(X_train):], predictions, label='Predicted')
plt.legend()
plt.title("Stock Price Prediction - AAPL")
plt.show()