from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import datetime
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Dictionary pemetaan nama dropdown ke kode Yahoo Finance
KODE_YF = {
    "IHSG": "^JKSE",
    "BBCA": "BBCA.JK",
    "BBRI": "BBRI.JK",
    "GOTO": "GOTO.JK",
    "BUMI": "BUMI.JK",
    "GOLD": "GOLD"
}

def prepare_and_predict(nama_saham):
    ticker = KODE_YF[nama_saham]
    
    # 1. Load Model dan Scaler secara dinamis sesuai pilihan dropdown
    # 1. Load Model dan Scaler dari dalam folder 'models'
    model = load_model(f'Models/model_{nama_saham}.keras')
    scaler = joblib.load(f'Models/scaler_{nama_saham}.pkl')
    
    # 2. Tarik data dari yfinance
    df = yf.download(ticker, period="3mo")
    
    if df.empty:
        raise ValueError(f"Data untuk {nama_saham} tidak ditemukan.")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Outlier Treatment
    filt = df.index.isin(df[(np.abs(stats.zscore(df['High'])) > 3)].index.values)
    df.loc[filt, ['Open','High','Low','Close']] = df.loc[filt, ['Open','High','Low','Close']].apply(lambda x: x/100)
    
    prices_historis = [float(x) for x in df['Close'].tolist()]
    dates_historis = df.index.strftime('%Y-%m-%d').tolist()
    last_price = prices_historis[-1]
    
    # Proses Prediksi
    df_20_days = df.iloc[-20:].copy()
    data_array = df_20_days.values.flatten().reshape(1, -1) 
    x_scaled = scaler.transform(data_array)
    x_ready = x_scaled.reshape((1, 20, 5))
    
    ypred = model.predict(x_ready)
    predicted_prices = ypred[0].tolist() 
    
    return dates_historis, prices_historis, last_price, predicted_prices, df.index[-1]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Tangkap 'value' dari dropdown (contoh: "BBCA" atau "IHSG")
        pilihan = request.form.get('pilihan_saham')
            
        try:
            dates_historis, prices_historis, last_price, predicted_prices, last_date = prepare_and_predict(pilihan)
            
            dates_prediksi = []
            next_date = last_date
            for _ in range(len(predicted_prices)):
                next_date += datetime.timedelta(days=1)
                while next_date.weekday() > 4: 
                    next_date += datetime.timedelta(days=1)
                dates_prediksi.append(next_date.strftime('%Y-%m-%d'))
                
            labels = dates_historis + dates_prediksi
            data_historis = prices_historis + [None] * len(predicted_prices)
            null_padding = [None] * (len(prices_historis) - 1)
            data_prediksi = null_padding + [last_price] + predicted_prices
            
            return render_template('index.html', 
                                   ticker=pilihan, # Kirim nama pilihan untuk auto-select di dropdown
                                   labels=labels,
                                   data_historis=data_historis,
                                   data_prediksi=data_prediksi)
                                   
        except Exception as e:
            return render_template('index.html', error=f"Terjadi kesalahan: {str(e)}", ticker='', labels=[], data_historis=[], data_prediksi=[])
            
    return render_template('index.html', ticker='IHSG', labels=[], data_historis=[], data_prediksi=[])

if __name__ == '__main__':
    app.run(debug=True)