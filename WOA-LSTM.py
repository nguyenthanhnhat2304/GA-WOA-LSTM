import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import random

data_dir = "./processed_data"
os.makedirs(data_dir, exist_ok=True)
os.makedirs("train_results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

X_train = np.load(os.path.join(data_dir, "X_train.npy"))
X_test = np.load(os.path.join(data_dir, "X_test.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))
y_test = np.load(os.path.join(data_dir, "y_test.npy"))
scaler = joblib.load(os.path.join(data_dir, "scaler.save"))


def build_lstm_model(units, dropout_rate, learning_rate, input_shape):
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(units // 2, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units // 2),
            Dropout(dropout_rate),
            Dense(1),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error"
    )
    return model


def decode_solution(sol):
    units = int(round(sol[0] / 16) * 16)
    dropout = sol[1]
    batch_size = int(round(sol[2] / 8) * 8)
    learning_rate = sol[3]
    return units, dropout, batch_size, learning_rate


# 适应度函数
def evaluate_model(units, dropout_rate, batch_size, learning_rate):
    model = build_lstm_model(
        units, dropout_rate, learning_rate, (X_train.shape[1], X_train.shape[2])
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=80,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
    )
    return min(history.history["val_loss"])


def whale_optimization(objective, dim, bounds, num_whales=20, max_iter=15):
    population = [
        np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dim)])
        for _ in range(num_whales)
    ]
    fitness = [objective(*decode_solution(ind)) for ind in population]

    elite_idx = np.argmin(fitness)
    elite_solution = population[elite_idx].copy()
    elite_fitness = fitness[elite_idx]

    history = []

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)
        for i in range(num_whales):
            r = random.random()
            A = 2 * a * r - a
            C = 2 * r
            p = random.random()
            D = abs(C * elite_solution - population[i])
            if p < 0.5:
                if abs(A) < 1:
                    population[i] = elite_solution - A * D
                else:
                    rand_idx = random.randint(0, num_whales - 1)
                    rand_sol = population[rand_idx]
                    D = abs(C * rand_sol - population[i])
                    population[i] = rand_sol - A * D
            else:
                distance_to_best = abs(elite_solution - population[i])
                population[i] = (
                    distance_to_best
                    * np.exp(-2 * random.random())
                    * np.cos(2 * np.pi * random.random())
                    + elite_solution
                )
            population[i] = np.clip(
                population[i], [b[0] for b in bounds], [b[1] for b in bounds]
            )

        fitness = [objective(*decode_solution(ind)) for ind in population]
        gen_best_idx = np.argmin(fitness)
        gen_best_solution = population[gen_best_idx].copy()
        gen_best_fitness = fitness[gen_best_idx]

        if gen_best_fitness < elite_fitness:
            elite_solution = gen_best_solution.copy()
            elite_fitness = gen_best_fitness

        decoded = decode_solution(elite_solution)
        history.append(
            {
                "Generation": t + 1,
                "units": decoded[0],
                "dropout": decoded[1],
                "batch_size": decoded[2],
                "learning_rate": decoded[3],
                "val_loss(RMSE)": elite_fitness,
            }
        )

        print(
            f"Thế hệ {t+1}/{max_iter} Thế hệ tốt nhất hiện tại RMSE: {elite_fitness:.4f}"
        )

    pd.DataFrame(history).to_csv(
        "train_results/WOA_LSTM_optimization_history.csv", index=False
    )
    return decode_solution(elite_solution), [row["val_loss(RMSE)"] for row in history]


bounds = [(64, 256), (0.05, 0.3), (16, 128), (1e-4, 1e-2)]


print("Đang sử dụng WOA nâng cao để tối ưu hóa siêu tham số LSTM...")
best_params, losses = whale_optimization(evaluate_model, dim=4, bounds=bounds)
units, dropout, batch_size, lr = best_params

print("\n Các tham số tối ưu:")
print(f"units         = {units}")
print(f"dropout       = {dropout:.3f}")
print(f"batch_size    = {batch_size}")
print(f"learning_rate = {lr:.6f}")
pd.DataFrame(
    [
        {
            "units": units,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": lr,
        }
    ]
).to_csv("train_results/WOA_LSTM_best_params.csv", index=False)

model = build_lstm_model(units, dropout, lr, (X_train.shape[1], X_train.shape[2]))
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
history = model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("WOA-LSTM Training Loss")
plt.tight_layout()
plt.savefig("figures/WOA_LSTM_training_loss.png")
plt.close()


def inverse_transform(y):
    return y * (scaler.data_max_[-1] - scaler.data_min_[-1]) + scaler.data_min_[-1]


y_train_pred = model.predict(X_train).flatten()
y_train_inv = inverse_transform(y_train)
y_train_pred_inv = inverse_transform(y_train_pred)

y_test_pred = model.predict(X_test).flatten()
y_test_inv = inverse_transform(y_test)
y_test_pred_inv = inverse_transform(y_test_pred)

plt.figure(figsize=(16, 6))
plt.plot(y_train_inv, label="Actual Train", color="green")
plt.plot(y_train_pred_inv, label="Predicted Train", color="purple")
plt.legend()
plt.title("WOA-LSTM Train Prediction")
plt.tight_layout()
plt.savefig("figures/WOA_LSTM_train_fit_plot.png")
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(y_test_inv, label="Actual Test", color="blue")
plt.plot(y_test_pred_inv, label="Predicted Test", color="red")
plt.legend()
plt.title("WOA-LSTM Test Prediction")
plt.tight_layout()
plt.savefig("figures/WOA_LSTM_test_fit_plot.png")
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(losses, marker="o")
plt.title("WOA-LSTM Convergence Curve")
plt.xlabel("Generation")
plt.ylabel("Validation RMSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/WOA_LSTM_convergence_curve.png")
plt.show()

train_mae = mean_absolute_error(y_train_inv, y_train_pred_inv)
train_mape = np.mean(np.abs((y_train_inv - y_train_pred_inv) / y_train_inv)) * 100
train_rmse = np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))
train_r2 = r2_score(y_train_inv, y_train_pred_inv)

test_mae = mean_absolute_error(y_test_inv, y_test_pred_inv)
test_mape = np.mean(np.abs((y_test_inv - y_test_pred_inv) / y_test_inv)) * 100
test_rmse = np.sqrt(mean_squared_error(y_test_inv, y_test_pred_inv))
test_r2 = r2_score(y_test_inv, y_test_pred_inv)

print("\nKết quả đánh giá tập huấn luyện：")
print(f"Train MAE  = {train_mae:.4f}")
print(f"Train MAPE = {train_mape:.2f}%")
print(f"Train RMSE = {train_rmse:.4f}")
print(f"Train R²   = {train_r2:.4f}")

print("\nKết quả đánh giá tập kiểm tra：")
print(f"Test MAE  = {test_mae:.4f}")
print(f"Test MAPE = {test_mape:.2f}%")
print(f"Test RMSE = {test_rmse:.4f}")
print(f"Test R²   = {test_r2:.4f}")
pd.DataFrame(
    {
        "Dataset": ["Train", "Test"],
        "MAE": [train_mae, test_mae],
        "MAPE (%)": [train_mape, test_mape],
        "RMSE": [train_rmse, test_rmse],
        "R2": [train_r2, test_r2],
    }
).to_csv("train_results/WOA_LSTM_evaluation_metrics.csv", index=False)


pd.DataFrame(
    {"Actual_Close_Train": y_train_inv, "Predicted_Close_Train": y_train_pred_inv}
).to_csv("train_results/WOA_LSTM_train_prediction.csv", index=False)
pd.DataFrame({"Actual_Close": y_test_inv, "Predicted_Close": y_test_pred_inv}).to_csv(
    "train_results/WOA_LSTM_test_prediction.csv", index=False
)


model.save("train_results/WOA_LSTM_model.h5")
np.save("train_results/y_pred_train_WOA_LSTM.npy", y_train_pred_inv)
np.save("train_results/y_train_WOA_LSTM.npy", y_train_inv)
np.save("train_results/y_pred_WOA_LSTM.npy", y_test_pred_inv)
np.save("train_results/y_test_WOA_LSTM.npy", y_test_inv)
