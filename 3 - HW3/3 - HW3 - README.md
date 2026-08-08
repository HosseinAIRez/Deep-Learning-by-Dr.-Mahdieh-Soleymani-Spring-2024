# Chapter 3: RNNs, LSTMs, GRU, ARIMA, GPT-2, PEFT, and Reasoning

In this chapter, we explore the application of deep learning to **Natural Language Processing (NLP)** and **Time Series Forecasting**. The chapter covers recurrent neural networks for sequential data, statistical time series models, GPT-2, Parameter-Efficient Fine-Tuning (PEFT), and reasoning capabilities in modern language models.

> **Note**
>
> This chapter is divided into two sections:
>
> - Theoretical
> - Practical

---

# ✒️ Theoretical Part

The theoretical section covers the following topics:

- Backpropagation Through Time (BPTT) in LSTMs
- Attention mechanism and its computational & memory complexity
- Positional Encoding
- Mathematical formulation of Self-Attention and Multi-Head Attention
- Technical analysis of Encoder-Only language models (BERT)
- Gaussian Encoding and its characteristics
- The relationship between CNNs and Attention mechanisms

---

# 👨‍💻 Practical Part

The practical section consists of several hands-on projects covering both **Deep Learning** and **Classical Time Series Forecasting**.

## 1. Crude Oil Price Forecasting

In this project, historical crude oil prices from **Yahoo Finance** (2010–Present) are used to predict future oil prices.

The workflow includes:

- Downloading historical oil price data
- Filling missing dates using interpolation methods
- Temporal train/validation/test splitting
- Data normalization
- Sliding-window sequence generation
- Training and evaluating:
  - RNN
  - LSTM
  - GRU
  - ARIMA
  - SARIMA

Model performance is evaluated using several regression metrics, including:

- RMSE
- MAE
- MAPE
- R² Score

The trained models, generated tensors, processed datasets, and prediction visualizations are stored in the `Results` directory.

# 📂 Repository Contents

The `Results` directory contains the following artifacts generated throughout the project.

## `Q1_RNN`:
### 🤖 Trained Models

- `ARIMA.pkl` — Serialized ARIMA model with the optimal hyperparameters.
- `SARIMA.pkl` — Serialized SARIMA model selected using AIC-based grid search.
- `RNN_oil_checkpoint.pt` — Trained RNN model checkpoint.
- `LSTM_oil_checkpoint.pt` — Trained LSTM model checkpoint.
- `GRU_oil_checkpoint.pt` — Trained GRU model checkpoint.

---

### 📊 Processed Dataset

- `full_data_filled.csv` — Final cleaned dataset after interpolation and missing value handling.
- `train_data.csv` — Training split used for model fitting.
- `val_data.csv` — Validation split used for hyperparameter tuning.
- `test_data.csv` — Test split used for final model evaluation.
- `oil_dataset.pt` — Preprocessed PyTorch tensors generated using sliding-window sequences.

---

### 📈 Prediction Results

- `ARIMA Predictions vs Actual Values.png` — Comparison between ARIMA predictions and ground-truth values.
- `SARIMA Predictions vs Actual Values.png` — Comparison between SARIMA predictions and ground-truth values.
- `RNN Predictions vs Actual Values.png` — Prediction results of the RNN model.
- `LSTM Predictions vs Actual Values.png` — Prediction results of the LSTM model.
- `GRU Predictions vs Actual Values.png` — Prediction results of the GRU model.

---

### 📉 Data Visualization

- `Histogram of Close.png` — Distribution of the normalized **Close** price values.
---

# 🌳 Project Structure

```text
3 - HW3/
├── Practical/
│   ├── Results/
│   │   ├── CSV Files/
│   │   ├── Models/
│   │   ├── Graphs/
│   │   └── Tensors/
│   │
│   └── Q1_RNN.ipynb
│
└── Theoretical/
```

---

# 🛠 Requirements

Install the required dependencies before running the notebooks:

```bash
pip install pmdarima yfinance
```
