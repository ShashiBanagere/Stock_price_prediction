import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from keras.models import load_model
import yfinance as yf

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #43A047;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-weight: bold;
        color: #1565C0;
    }
    .metric-value {
        font-size: 1.2rem;
        color: #2E7D32;
    }
    .chart-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

import numpy as np
# FUNCTION TO CREATE TIME SERIES DATASET WITH 5-DAY LOOK-BACK
def new_dataset(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset) - step_size - 1):
        a = dataset[i:(i + step_size), 0]  # Take 5 days of data
        data_X.append(a)
        data_Y.append(dataset[i + step_size, 0])  # Predict the next day
    return np.array(data_X), np.array(data_Y)


step_size = 5

# Main app
def main():
    st.markdown("<h1 class='main-header'>Price Pal - Stock Price Predictor</h1>", unsafe_allow_html=True)

    model_path = "stock model.keras"
    try:
        model = load_model(model_path)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return

    # File uploader
    st.header('Stock Market Predictor')

    stock =st.text_input('Enter Stock Symnbol', 'GOOG')
    start = '2012-01-01'
    end = '2024-11-29'

    data = yf.download(stock, start ,end)

    st.subheader('Stock Data')
    st.write(data)
    
    close_prices = data["Close"].values

        # CREATING OWN INDEX FOR FLEXIBILITY
    obs = np.arange(1, len(data) + 1, 1)

        # TAKING DIFFERENT INDICATORS FOR PREDICTION
    OHLC_avg = data[['Open','High','Low','Close']].mean(axis=1)
    HLC_avg = data[['High', 'Low', 'Close']].mean(axis=1)
    close_val = data[['Close']]

        # Plotting
    st.markdown("<h2 class='sub-header'>Stock Price Indicators</h2>", unsafe_allow_html=True)
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(obs, OHLC_avg, 'r', label='OHLC avg')
    ax1.plot(obs, HLC_avg, 'b', label='HLC avg')
    ax1.plot(obs, close_val, 'g', label='Closing price')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Time in Days')
    ax1.set_ylabel('Price')
    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # PREPARATION OF TIME SERIES DATASE
    OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
    scaler = MinMaxScaler(feature_range=(0, 1))
    OHLC_avg = scaler.fit_transform(OHLC_avg)

    # TRAIN-TEST SPLIT
    train_OHLC = int(len(OHLC_avg) * 0.75)
    test_OHLC = len(OHLC_avg) - train_OHLC
    train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]
    
    # PREPARATION OF TIME SERIES DATASET
    trainX, trainY = new_dataset(train_OHLC, step_size)
    testX, testY = new_dataset(test_OHLC, step_size)

    # RESHAPING TRAIN AND TEST DATA
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    # PREDICTION
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    # DE-NORMALIZING FOR PLOTTING
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    # CREATING SIMILAR DATASET TO PLOT TEST PREDICTIONS
    trainPredictPlot = np.empty_like(OHLC_avg)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

    # CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
    testPredictPlot = np.empty_like(OHLC_avg)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict
    
    # DE-NORMALIZING MAIN DATASET
    OHLC_avg = scaler.inverse_transform(OHLC_avg)
    
     # PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
    st.markdown("<h2 class='sub-header'>Stock Price Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(OHLC_avg, 'g', label='Original dataset')
    ax2.plot(trainPredictPlot, 'r', label='Training set')
    ax2.plot(testPredictPlot, 'b', label='Predicted stock price/test set')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Time in Days')
    ax2.set_ylabel('OHLC Value')
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)
    
    last_5_vals = testPredict[-step_size:]
    last_5_vals_scaled = scaler.transform(last_5_vals.reshape(-1, 1))
    next_val = model.predict(np.reshape(last_5_vals_scaled, (1, 1, step_size)))
    #print("Last Day Value:", last_5_vals[-1].item())
    #print("Next Day Value:", scaler.inverse_transform(next_val).item())
    
    st.markdown("<h2 class='sub-header'>Next Day Value Prediction</h2>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Last Day Value</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{last_5_vals[-1].item():.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col6:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Next Day Value (Predicted)</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='metric-value'>{scaler.inverse_transform(next_val).item():.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
