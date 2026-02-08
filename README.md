# GA-WOA-LSTM: Hybrid Deep Learning Framework for Stock Price Prediction 📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Introduction
This repository contains the implementation of the **GA-WOA-LSTM** model, developed as part of my Bachelor's Thesis: *"Improving LSTM Hyperparameter Optimization through Genetic and Whale Algorithms: Application to Vietnamese Stock Price Prediction"*.

The project addresses the challenge of hyperparameter tuning in deep learning by introducing a **dual-stage evolutionary strategy**:
1.  **Genetic Algorithm (GA):** Performs global exploration to identify promising hyperparameter regions.
2.  **Whale Optimization Algorithm (WOA):** Conducts local refinement to pinpoint the optimal configuration.
3.  **LSTM Network:** Utilizes the optimized parameters to forecast daily closing prices.

## 🚀 Key Features
- [cite_start]**Hybrid Optimization:** Seamlessly integrates GA (global search) and WOA (local refinement) to prevent premature convergence and stagnation in local optima[cite: 63, 64].
- [cite_start]**Robustness:** Validated on 15 years of historical data (2010–2025) from the Vietnamese stock market (**VIC, HPG, DPM**)[cite: 515, 517].
- [cite_start]**High Performance:** Achieved **$R^2$ scores up to 0.9888** (VIC stock), significantly outperforming standard LSTM, CNN, and RNN models[cite: 710, 716].
- [cite_start]**Comprehensive Benchmarking:** Includes implementations of 6 baseline models for rigorous comparison[cite: 701].

## 🛠 Technologies
- **Language:** Python 3.x
- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** Pandas, NumPy, Scikit-learn, Openpyxl
- **Data Source:** Yfinance (Yahoo Finance API)
- **Visualization:** Matplotlib

## 📂 Project Structure
This repository is organized as follows:

### Core Implementation
- `GA-WOA-LSTM.py`: **Main Model**. The hybrid framework combining GA, WOA, and LSTM.
- `crawl_data.py`: Script to fetch historical stock data (VIC, HPG, DPM) from Yahoo Finance.
- [cite_start]`data_preprocessing.py`: Handles data cleaning, MinMax scaling, and sliding window segmentation ($w=8$)[cite: 571].

### Baseline Models (For Comparison)
- `GA-LSTM.py`: LSTM optimized by Genetic Algorithm only.
- `WOA-LSTM.py`: LSTM optimized by Whale Optimization Algorithm only.
- `LSTM.py`: Standard Long Short-Term Memory network.
- `RNN.py`: Recurrent Neural Network.
- `CNN.py`: 1D Convolutional Neural Network.
- `BP.py`: Backpropagation Neural Network.

## 📊 Results
The proposed GA-WOA-LSTM model demonstrated superior performance across all metrics (MAE, RMSE, MAPE, $R^2$).

**Performance on Vingroup (VIC) Test Set:**

| Model | MAE | RMSE | MAPE | $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **GA-WOA-LSTM** | **403.61** | **623.08** | **1.51%** | **0.9888** |
| GA-LSTM | 415.20 | 663.10 | 1.53% | 0.9873 |
| LSTM | 688.46 | 947.41 | 2.59% | 0.9742 |
| CNN | 1749.45 | 1988.51 | 7.22% | 0.8862 |
[cite_start]*(Data source: Thesis Evaluation Results [cite: 710])*

### Visualization
> *Comparison of Actual vs. Predicted Prices for VIC stock:*

![Prediction Result](images/result_vic.png)
*(Please upload your best prediction chart to an `images` folder and name it `result_vic.png`)*

## ⚡ How to Run

### 1. Clone the repository
```bash
git clone [https://github.com/nguyenthanhnhat2304/GA-WOA-LSTM.git](https://github.com/nguyenthanhnhat2304/GA-WOA-LSTM.git)
cd GA-WOA-LSTM
