import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os


data_dir = "./processed_data"
os.makedirs(data_dir, exist_ok=True)
os.makedirs("train_results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


X_train = np.load(os.path.join(data_dir, "X_train.npy"))
X_test = np.load(os.path.join(data_dir, "X_test.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))
y_test = np.load(os.path.join(data_dir, "y_test.npy"))
scaler = joblib.load(os.path.join(data_dir, "scaler.save"))


model = Sequential(
    [
        LSTM(
            64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])
        ),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1),
    ]
)
model.compile(optimizer="adam", loss="mean_squared_error")


early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
history = model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)


plt.figure(figsize=(16, 6))
plt.plot(history.history["loss"], label="Train Loss", color="blue")
plt.plot(history.history["val_loss"], label="Validation Loss", color="orange")
plt.title("Training & Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("figures/LSTM_training_loss.png")
plt.show()


y_train_pred = model.predict(X_train).flatten()
y_train_pred_inv = (
    y_train_pred * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
)
y_train_inv = (
    y_train * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
)
train_mae = mean_absolute_error(y_train_inv, y_train_pred_inv)
train_mape = np.mean(np.abs((y_train_inv - y_train_pred_inv) / y_train_inv)) * 100
train_rmse = np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))
train_r2 = r2_score(y_train_inv, y_train_pred_inv)

print("Kết quả đánh giá tập dữ liệu huấn luyện:")
print(f"Train MAE  = {train_mae:.4f}")
print(f"Train MAPE = {train_mape:.2f}%")
print(f"Train RMSE = {train_rmse:.4f}")
print(f"Train R²   = {train_r2:.4f}")


train_results = pd.DataFrame(
    {"Actual_Close_Train": y_train_inv, "Predicted_Close_Train": y_train_pred_inv}
)
train_results.to_csv("train_results/LSTM_train_prediction.csv", index=False)


plt.figure(figsize=(16, 6))
plt.plot(y_train_inv, label="Actual Train", color="green")
plt.plot(y_train_pred_inv, label="Predicted Train", color="purple", alpha=0.7)
plt.title("LSTM Training Prediction Performance")
plt.xlabel("Time Step")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.savefig("figures/LSTM_train_fit_plot.png")
plt.show()


y_pred = model.predict(X_test).flatten()
y_pred_inv = (
    y_pred * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
)
y_test_inv = (
    y_test * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]
)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print("Kết quả đánh giá tập dữ liệu kiểm tra:")
print(f"Test MAE  = {mae:.4f}")
print(f"Test MAPE = {mape:.2f}%")
print(f"Test RMSE = {rmse:.4f}")
print(f"Test R²   = {r2:.4f}")


test_results = pd.DataFrame({"Actual_Close": y_test_inv, "Predicted_Close": y_pred_inv})
test_results.to_csv("train_results/LSTM_test_prediction.csv", index=False)


plt.figure(figsize=(16, 6))
plt.plot(y_test_inv, label="Actual Test", color="blue")
plt.plot(y_pred_inv, label="Predicted Test", color="red", alpha=0.7)
plt.title("LSTM Test Prediction Performance")
plt.xlabel("Time Step")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.savefig("figures/LSTM_test_fit_plot.png")
plt.show()


evaluation_metrics = pd.DataFrame(
    {
        "Dataset": ["Train", "Test"],
        "MAE": [train_mae, mae],
        "MAPE (%)": [train_mape, mape],
        "RMSE": [train_rmse, rmse],
        "R2": [train_r2, r2],
    }
)
evaluation_metrics.to_csv("train_results/LSTM_evaluation_metrics.csv", index=False)


model.save("train_results/LSTM_model.h5")


np.save("train_results/y_pred_train_LSTM.npy", y_train_pred_inv)
np.save("train_results/y_train_LSTM.npy", y_train_inv)
np.save("train_results/y_pred_LSTM.npy", y_pred_inv)
np.save("train_results/y_test_LSTM.npy", y_test_inv)
