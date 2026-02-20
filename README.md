# 📈 GA-WOA-LSTM: Improving LSTM Hyperparameter Optimization for Stock Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

[cite_start]**GA-WOA-LSTM** is a hybrid deep learning framework that combines the **Genetic Algorithm (GA)** and the **Whale Optimization Algorithm (WOA)** to optimize the hyperparameters of a **Long Short-Term Memory (LSTM)** neural network[cite: 60]. [cite_start]This project is specifically applied to forecast the closing prices of major stocks in the Vietnamese market[cite: 60, 62].

---

## 🎯 Overview & Motivation

[cite_start]Financial time-series data in emerging markets like Vietnam exhibit strong non-linearity, high volatility, and non-stationarity[cite: 71, 121]. [cite_start]While LSTM networks excel at capturing long-term dependencies, their performance is highly sensitive to hyperparameter configurations[cite: 76, 77]. 

This project introduces a dual-stage evolutionary optimization strategy:
1. [cite_start]**Global Exploration (GA):** Utilizes selection, crossover, and mutation to broadly explore the hyperparameter search space and prevent premature convergence[cite: 140, 419].
2. [cite_start]**Local Exploitation (WOA):** Refines elite candidate solutions using mechanisms inspired by the bubble-net hunting strategy of humpback whales[cite: 140, 342, 419].

[cite_start]By dynamically fusing GA and WOA, this model effectively balances exploration and exploitation, outperforming standalone optimization techniques[cite: 85, 413, 419].

---

## 📊 Dataset

[cite_start]The model is trained and evaluated on historical daily stock data from three leading companies on the Vietnam Stock Exchange, spanning 15 years from **March 15, 2010, to March 13, 2025**[cite: 510, 512]:
* [cite_start]**VIC:** Vingroup Joint Stock Company[cite: 510].
* [cite_start]**HPG:** Hoa Phat Group[cite: 510].
* [cite_start]**DPM:** Petrovietnam Fertilizer & Chemicals Corporation[cite: 510].

**Preprocessing pipeline:**
* [cite_start]**Features:** Open, High, Low, Close, Volume[cite: 513].
* [cite_start]**Scaling:** Min-Max scaling (fitted only on the training set to prevent data leakage)[cite: 560, 561].
* [cite_start]**Data Split:** 80% Training, 20% Testing[cite: 557, 558].
* [cite_start]**Sliding Window:** Sequence length of $w = 8$ trading days to predict the closing price of day $T+1$[cite: 565, 566].

---

## 🧠 Model Architecture & Optimization

### Hyperparameter Search Space
[cite_start]The cooperative GA-WOA optimizer searches for the optimal LSTM configuration within the following bounds[cite: 669, 672]:
* [cite_start]**LSTM Hidden Units:** [64, 256] [cite: 672]
* [cite_start]**Dropout Rate:** [0.05, 0.3] [cite: 672]
* [cite_start]**Batch Size:** [16, 128] [cite: 672]
* [cite_start]**Learning Rate:** $[1 \times 10^{-4}, 1 \times 10^{-2}]$ [cite: 672]

### Fitness Function
[cite_start]The objective is to minimize a composite fitness function that balances the Weighted Mean Squared Error (WMSE) and the Goodness-of-Fit ($R^2$)[cite: 475, 476]:
[cite_start]$$F(\theta) = WMSE(\theta) + (1 - R^2(\theta))$$ [cite: 475]

---

## 🏆 Experimental Results

[cite_start]The proposed **GA-WOA-LSTM** was benchmarked against standard models including BP, CNN, RNN, standard LSTM, WOA-LSTM, and GA-LSTM[cite: 716]. 

[cite_start]It consistently achieved the best generalization on the Test Set across all three stocks[cite: 730].

| Stock | MAE | MAPE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **VIC** | 403.61 | 1.51% | 623.08 | [cite_start]0.9888 | [cite: 731]
| **HPG** | 317.71 | 1.73% | 431.95 | [cite_start]0.9848 | [cite: 733]
| **DPM** | 387.76 | 2.01% | 589.08 | [cite_start]0.9670 | [cite: 735]

[cite_start]*Highlights:* For HPG, GA-WOA-LSTM reduced RMSE by 11.89% compared to the standalone WOA-LSTM[cite: 734]. [cite_start]For DPM, it reduced MAPE by 24.15% compared to GA-LSTM[cite: 736].

---

## 📂 Project Structure

```text
GA-WOA-LSTM/
├── data/                   # (Ignored in git) Store raw and processed data
├── models/                 # (Ignored in git) Saved model weights
├── images/                 # Architecture flowcharts and result plots
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
└── README.md# 📈 GA-WOA-LSTM: Improving LSTM Hyperparameter Optimization for Stock Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

[cite_start]**GA-WOA-LSTM** is a hybrid deep learning framework that combines the **Genetic Algorithm (GA)** and the **Whale Optimization Algorithm (WOA)** to optimize the hyperparameters of a **Long Short-Term Memory (LSTM)** neural network[cite: 60]. [cite_start]This project is specifically applied to forecast the closing prices of major stocks in the Vietnamese market[cite: 60, 62].

---

## 🎯 Overview & Motivation

[cite_start]Financial time-series data in emerging markets like Vietnam exhibit strong non-linearity, high volatility, and non-stationarity[cite: 71, 121]. [cite_start]While LSTM networks excel at capturing long-term dependencies, their performance is highly sensitive to hyperparameter configurations[cite: 76, 77]. 

This project introduces a dual-stage evolutionary optimization strategy:
1. [cite_start]**Global Exploration (GA):** Utilizes selection, crossover, and mutation to broadly explore the hyperparameter search space and prevent premature convergence[cite: 140, 419].
2. [cite_start]**Local Exploitation (WOA):** Refines elite candidate solutions using mechanisms inspired by the bubble-net hunting strategy of humpback whales[cite: 140, 342, 419].

[cite_start]By dynamically fusing GA and WOA, this model effectively balances exploration and exploitation, outperforming standalone optimization techniques[cite: 85, 413, 419].

---

## 📊 Dataset

[cite_start]The model is trained and evaluated on historical daily stock data from three leading companies on the Vietnam Stock Exchange, spanning 15 years from **March 15, 2010, to March 13, 2025**[cite: 510, 512]:
* [cite_start]**VIC:** Vingroup Joint Stock Company[cite: 510].
* [cite_start]**HPG:** Hoa Phat Group[cite: 510].
* [cite_start]**DPM:** Petrovietnam Fertilizer & Chemicals Corporation[cite: 510].

**Preprocessing pipeline:**
* [cite_start]**Features:** Open, High, Low, Close, Volume[cite: 513].
* [cite_start]**Scaling:** Min-Max scaling (fitted only on the training set to prevent data leakage)[cite: 560, 561].
* [cite_start]**Data Split:** 80% Training, 20% Testing[cite: 557, 558].
* [cite_start]**Sliding Window:** Sequence length of $w = 8$ trading days to predict the closing price of day $T+1$[cite: 565, 566].

---

## 🧠 Model Architecture & Optimization

### Hyperparameter Search Space
[cite_start]The cooperative GA-WOA optimizer searches for the optimal LSTM configuration within the following bounds[cite: 669, 672]:
* [cite_start]**LSTM Hidden Units:** [64, 256] [cite: 672]
* [cite_start]**Dropout Rate:** [0.05, 0.3] [cite: 672]
* [cite_start]**Batch Size:** [16, 128] [cite: 672]
* [cite_start]**Learning Rate:** $[1 \times 10^{-4}, 1 \times 10^{-2}]$ [cite: 672]

### Fitness Function
[cite_start]The objective is to minimize a composite fitness function that balances the Weighted Mean Squared Error (WMSE) and the Goodness-of-Fit ($R^2$)[cite: 475, 476]:
[cite_start]$$F(\theta) = WMSE(\theta) + (1 - R^2(\theta))$$ [cite: 475]

---

## 🏆 Experimental Results

[cite_start]The proposed **GA-WOA-LSTM** was benchmarked against standard models including BP, CNN, RNN, standard LSTM, WOA-LSTM, and GA-LSTM[cite: 716]. 

[cite_start]It consistently achieved the best generalization on the Test Set across all three stocks[cite: 730].

| Stock | MAE | MAPE | RMSE | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **VIC** | 403.61 | 1.51% | 623.08 | [cite_start]0.9888 | [cite: 731]
| **HPG** | 317.71 | 1.73% | 431.95 | [cite_start]0.9848 | [cite: 733]
| **DPM** | 387.76 | 2.01% | 589.08 | [cite_start]0.9670 | [cite: 735]

[cite_start]*Highlights:* For HPG, GA-WOA-LSTM reduced RMSE by 11.89% compared to the standalone WOA-LSTM[cite: 734]. [cite_start]For DPM, it reduced MAPE by 24.15% compared to GA-LSTM[cite: 736].

---

## 📂 Project Structure

```text
GA-WOA-LSTM/
├── data/                   # (Ignored in git) Store raw and processed data
├── models/                 # (Ignored in git) Saved model weights
├── images/                 # Architecture flowcharts and result plots
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
