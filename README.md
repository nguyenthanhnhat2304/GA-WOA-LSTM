# 📈 GA-WOA-LSTM: Improving LSTM Hyperparameter Optimization for Stock Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**GA-WOA-LSTM** is a hybrid deep learning framework that combines the **Genetic Algorithm (GA)** and the **Whale Optimization Algorithm (WOA)** to optimize the hyperparameters of a **Long Short-Term Memory (LSTM)** neural network. This project is specifically applied to forecast the closing prices of major stocks in the Vietnamese market.

---

## 🎯 Overview & Motivation

Financial time-series data in emerging markets like Vietnam exhibit strong non-linearity, high volatility, and non-stationarity. While LSTM networks excel at capturing long-term dependencies, their performance is highly sensitive to hyperparameter configurations.

This project introduces a dual-stage evolutionary optimization strategy:

1. **Global Exploration (GA):** Utilizes selection, crossover, and mutation to broadly explore the hyperparameter search space and prevent premature convergence.
2. **Local Exploitation (WOA):** Refines elite candidate solutions using mechanisms inspired by the bubble-net hunting strategy of humpback whales.

By dynamically fusing GA and WOA, this model effectively balances exploration and exploitation, outperforming standalone optimization techniques.

---

## 🔄 Framework / Workflow

The flowchart below illustrates the complete pipeline of the proposed GA-WOA-LSTM model, from data preprocessing to hyperparameter optimization and final prediction:

![GA-WOA-LSTM Framework](images/flowchart.png)


---

## 📊 Dataset

The model is trained and evaluated on historical daily stock data from three leading companies on the Vietnam Stock Exchange, spanning 15 years from **March 15, 2010, to March 13, 2025**:

- **VIC:** Vingroup Joint Stock Company.
- **HPG:** Hoa Phat Group.
- **DPM:** Petrovietnam Fertilizer & Chemicals Corporation.

**Preprocessing pipeline:**

- **Features:** Open, High, Low, Close, Volume.
- **Scaling:** Min-Max scaling (fitted only on the training set to prevent data leakage).
- **Data Split:** 80% Training, 20% Testing.
- **Sliding Window:** Sequence length of $w = 8$ trading days to predict the closing price of day $T+1$.

---

## 🧠 Model Architecture & Optimization

### Hyperparameter Search Space

The cooperative GA-WOA optimizer searches for the optimal LSTM configuration within the following bounds:

- **LSTM Hidden Units:** [64, 256]
- **Dropout Rate:** [0.05, 0.3]
- **Batch Size:** [16, 128]
- **Learning Rate:** $[1 \times 10^{-4}, 1 \times 10^{-2}]$

### Fitness Function

The objective is to minimize a composite fitness function that balances the Weighted Mean Squared Error (WMSE) and the Goodness-of-Fit ($R^2$):
$$F(\theta) = WMSE(\theta) + (1 - R^2(\theta))$$

---

## 🏆 Experimental Results

The proposed **GA-WOA-LSTM** was benchmarked against standard models including BP, CNN, RNN, standard LSTM, WOA-LSTM, and GA-LSTM.

It consistently achieved the best generalization on the Test Set across all three stocks.

| Stock | MAE | MAPE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **VIC** | 403.61 | 1.51% | 623.08 | 0.9888 |
| **HPG** | 317.71 | 1.73% | 431.95 | 0.9848 |
| **DPM** | 387.76 | 2.01% | 589.08 | 0.9670 |

*Highlights:* For HPG, GA-WOA-LSTM reduced RMSE by 11.89% compared to the standalone WOA-LSTM. For DPM, it reduced MAPE by 24.15% compared to GA-LSTM.

---

## 📂 Project Structure

```text
GA-WOA-LSTM/
├── data/                   # (Ignored in git) Store raw and processed data
├── models/                 # (Ignored in git) Saved model weights
├── images/                 # Architecture flowcharts and result plots
│   └── flowchart.png       # Framework image
├── utils.py                # Helper functions for metrics, plotting, and processing
├── crawl_data.py           # Script to fetch stock data via yfinance
├── data_preprocessing.py   # Data cleaning, scaling, and sliding window generation
├── BP.py                   # Backpropagation Baseline
├── CNN.py                  # 1D-CNN Baseline
├── RNN.py                  # Standard RNN Baseline
├── LSTM.py                 # Standard LSTM Baseline
├── WOA_LSTM.py             # LSTM optimized by Whale Optimization Algorithm
├── GA_LSTM.py              # LSTM optimized by Genetic Algorithm
├── GA_WOA_LSTM.py          # 🌟 Main Hybrid Model
├── main.py                 # Entry point to execute various models
└── README.md

