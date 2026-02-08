# GA-WOA-LSTM Hybrid Model for Stock Price Prediction 📈

This repository contains the implementation of a hybrid deep learning framework combining **Genetic Algorithm (GA)**, **Whale Optimization Algorithm (WOA)**, and **Long Short-Term Memory (LSTM)** networks for forecasting stock prices.

This project was developed as part of my Bachelor's Thesis in Mathematical Economics.

##  Key Features
- **Hybrid Optimization:** Uses GA for global search and WOA for local refinement to optimize LSTM hyperparameters.
- **Robustness:** Overcomes the limitations of manual tuning and local optima stagnation.
- **Tested Data:** Validated on 15 years of data (2010-2025) from the Vietnamese stock market (VIC, HPG, DPM).
- **Performance:** Achieved **R² scores up to 0.9888** and significantly reduced MAE/RMSE compared to standard LSTM, CNN, and RNN models.

##  Technologies
- **Language:** Python 3.x
- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib

##  Project Structure
- `crawl_data.py`: Script to scrape/fetch historical stock data.
- `data_preprocessing.py`: Data cleaning, normalization (MinMax), and sliding window segmentation.
- `GA-WOA-LSTM.py`: The core hybrid model implementation.
- `GA-LSTM.py` / `WOA-LSTM.py`: Baseline optimization models for comparison.
- `BP.py`, `CNN.py`, `RNN.py`, `LSTM.py`: Baseline neural network models.

##  Results
*(Bạn hãy chụp màn hình các biểu đồ đẹp nhất trong luận văn - ví dụ hình 3.7, 3.16 - và chèn vào đây. Hình ảnh quan trọng hơn lời nói!)*

![Prediction Result Example](link_to_your_image.png)

##  How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
