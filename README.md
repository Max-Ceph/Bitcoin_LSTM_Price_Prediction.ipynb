# Bitcoin Price Forecasting Using LSTM Neural Networks

## 📌 Project Overview

This research project explores the use of **Long Short-Term Memory (LSTM)** neural networks for forecasting **Bitcoin (BTC)** price trends. Given the high volatility and non-linear behavior of cryptocurrency markets, traditional models such as ARIMA or GARCH often underperform. This project demonstrates how LSTM models can capture long-term temporal dependencies and provide accurate, trend-aligned predictions.

---

## 🧠 Objectives

- Implement an LSTM-based time-series model for Bitcoin price prediction
- Compare LSTM performance with traditional models (ARIMA) and alternatives (GRU, Transformer)
- Evaluate using MSE, MAE, and MAPE metrics
- Explore real-world applications for investors, traders, and analysts

---

## 📅 Dataset

- **Source**: [Binance API](https://www.binance.com)
- **Interval**: 30-minute historical BTC/USDT trading data
- **Fields**: `Timestamp`, `Open`, `High`, `Low`, `Close`, `Volume`
- Dataset was selected over alternatives (CoinGecko, CoinMarketCap) due to Binance's:
  - High liquidity
  - Granular intervals
  - Reliable and real-time API access

---

## ⚙️ Methods Used

### 🔧 Preprocessing
- Missing value handling
- `MinMaxScaler` for normalization
- Look-back window of 60 time steps
- Feature engineering (moving averages, volume trends, etc.)

### 🧠 Model Architecture
- 2 stacked LSTM layers with 50 units each
- Dropout layers (20%) to reduce overfitting
- Dense layers with ReLU and linear activation
- Optimizer: Adam (lr=0.001)
- Loss Function: MSE

### 🧪 Evaluation Metrics
- **Mean Squared Error (MSE)**: 197,761.60  
- **Mean Absolute Error (MAE)**: 425.99  
- **Mean Absolute Percentage Error (MAPE)**: 2.32%

---

## 🔬 Experimental Results

- The LSTM model captured overall BTC price trends with **high temporal consistency**
- Performed well on both 3-year and 5-month datasets
- Slight underestimation of price volatility (smoother predictions)
- Best suited for **medium- to long-term forecasting**
- Outperformed ARIMA and GRU; transformer models were not implemented due to computational cost

---

## 📉 Limitations & Future Work

- Under-response to sudden price spikes or crashes
- Slight prediction lag
- Potential improvements:
  - Add features (volume, on-chain metrics, sentiment)
  - Try hybrid or ensemble models
  - Experiment with transformer-based architectures (Informer, TFT)
  - Hyperparameter tuning via grid or Bayesian search

---

## 💻 Code Snippets

Example model training (see notebook for full implementation):

```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=50)
