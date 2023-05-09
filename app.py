from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # get user input
        stock_symbol = request.form['stock_symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        # get stock data
        df = pd.read_csv('./symbols_valid_meta.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # train prediction model
        X = df[['open', 'high', 'low']].values
        y = df['close'].values
        model = LinearRegression()
        model.fit(X, y)
        
        # make prediction
        today = datetime.today().strftime('%Y-%m-%d')
        prediction = model.predict([[df['open'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]]])[0]
        
        # render prediction result
        return render_template('prediction.html', stock_symbol=stock_symbol, prediction=prediction, today=today)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run()
