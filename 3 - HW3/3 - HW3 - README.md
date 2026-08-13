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

## 2. Building a Small Language Model with GPT-2

In this project, we built a small local language model based on **GPT-2 Small (124M parameters)** and trained it on the [`Snappfood Persian Sentiment Analysis`](https://www.kaggle.com/datasets/mohammad1ziyar/cleaned-snappfood-persian-sentiment-analysis) dataset.

The main objective was to build and train a lightweight decoder-only Transformer language model capable of generating **Positive** and **Negative** Persian reviews based on sentiment control tokens.

> **Note:**
>
> The original assignment recommended using either [**Llama-3.3-70B-Instruct**](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) or [**Gemma-2-27B-it**](https://huggingface.co/google/gemma-2-27b-it).
>
> Access to the Llama repository was denied due to its gated access policy. Although access to **Gemma-2-27B-it** was granted, its computational requirements were far beyond the available local hardware.
>
> Initially, a smaller Gemma model was also considered. However, experiments showed that even with a limited GPU, training a larger language model was impractical in terms of memory consumption and training time.
>
> Therefore, to ensure reproducibility and allow the complete training pipeline to run on consumer-grade hardware, the project was migrated to **GPT-2 Small (124M parameters)**.
>
> GPT-2 provides a lightweight decoder-only Transformer architecture that is well suited for educational experimentation and language-modeling tasks. This modification affects only the backbone language model and does not change the overall training pipeline or the learning objectives.

### The workflow includes:

- Downloading and preprocessing the `Snappfood` labeled dataset
- Splitting the dataset into training, validation, and test sets
- Preparing the GPT-2 tokenizer and adding sentiment control tokens
- Building a lightweight GPT-2 configuration with custom architectural parameters
- Implementing the Transformer architecture and attention mechanism
- Defining the multi-head self-attention process manually
- Training the model using causal language modeling
- Evaluating the model on the validation set during training
- Saving the trained model weights and tokenizer
- Generating sentiment-controlled text using `<POS>` and `<NEG>` control tokens

### GPT-2 Architecture

The overall architecture used in this project follows the decoder-only Transformer structure:

```text
                    Input IDs
                        │
                        ▼
               Word Embedding (wte)
                        │
                        │
Position IDs ─────► Position Embedding (wpe)
                        │
                        ▼
                 Sum Embeddings
                        │
                        ▼
                    Dropout
                        │
                        ▼
        ┌────────────────────────────────┐
        │         Transformer Block 1    │
        └────────────────────────────────┘
                        │
        ┌────────────────────────────────┐
        │         Transformer Block 2    │
        └────────────────────────────────┘
                        │
                     ...
                        │
        ┌────────────────────────────────┐
        │        Transformer Block N     │
        └────────────────────────────────┘
                        │
                        ▼
                 Final LayerNorm
                        │
                        ▼
                  LM Head (Linear)
                        │
                        ▼
                  Vocabulary Logits
                        │
                        ▼
              Sampling / Decoding
                        │
                        ▼
                  Next Token
```
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

## `Q2_GPT2.ipynb`

### 🤖 Trained Model & Tokenizer

- `model_weights.pt` - weights of the best-performing locally trained GPT-2 model, selected based on validation loss.
- `tokenizer_config.json` - tokenizer configuration used for tokenization and sentiment control tokens.

---

### 📈 Generated Texts

- `ten_generated_examples.xlsx` - 10 generated examples containing both positive and negative sentiment-controlled reviews produced by the local GPT-2 model.

---

### 📉 Data Visualization & Training Graphs

- `Data_visualization-1.jpg` - visualization of labeled comments and their character lengths in the training and validation datasets.

  <p align="center">
    <img src="Data_visualization-1.jpg" alt="Labeled Snappfood comments and their character lengths" width="600">
    <br>
    <em>Examples of labeled comments from the Snappfood dataset</em>
  </p>

- `training_loss_graphs.png` - training and validation loss curves across the 3 training epochs. The relatively long and oscillating curves reflect the computational limitations and the limited training budget used for this experiment.
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
│   └── Q2_GPT2.ipynb
│
└── Theoretical/
```

---

# 🛠 Requirements

Install the required dependencies before running the notebooks:

```bash
pip install pmdarima yfinance accelerate transformers
```
