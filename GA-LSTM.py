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


def initialize_population(size, bounds):
    return [
        np.array([random.uniform(low, high) for (low, high) in bounds])
        for _ in range(size)
    ]


def crossover(parent1, parent2):
    alpha = random.random()
    return alpha * parent1 + (1 - alpha) * parent2


def mutate(solution, bounds, mutation_rate=0.1):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = random.uniform(bounds[i][0], bounds[i][1])
    return np.clip(solution, [b[0] for b in bounds], [b[1] for b in bounds])


def genetic_algorithm(objective, bounds, pop_size=20, generations=15):
    population = initialize_population(pop_size, bounds)
    fitness = [objective(*decode_solution(ind)) for ind in population]

    best_idx = np.argmin(fitness)
    elite_solution = population[best_idx].copy()
    elite_fitness = fitness[best_idx]

    history = []

    for gen in range(generations):
        new_population = []

        new_population.append(elite_solution.copy())

        while len(new_population) < pop_size:
            parents = random.sample(population, 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child, bounds)
            new_population.append(child)

        population = new_population
        fitness = [objective(*decode_solution(ind)) for ind in population]

        gen_best_idx = np.argmin(fitness)
        gen_best_solution = population[gen_best_idx]
        gen_best_fitness = fitness[gen_best_idx]

        if gen_best_fitness < elite_fitness:
            elite_solution = gen_best_solution.copy()
            elite_fitness = gen_best_fitness

        decoded = decode_solution(elite_solution)
        history.append(
            {
                "Generation": gen + 1,
                "units": decoded[0],
                "dropout": decoded[1],
                "batch_size": decoded[2],
                "learning_rate": decoded[3],
                "val_loss(RMSE)": elite_fitness,
            }
        )

        print(f"Generation {gen + 1}/{generations}, Best RMSE: {elite_fitness:.4f}")

    history_df = pd.DataFrame(history)
    history_df.to_csv("train_results/GA_LSTM_optimization_history.csv", index=False)

    return decode_solution(elite_solution), [row["val_loss(RMSE)"] for row in history]


bounds = [(64, 256), (0.05, 0.3), (16, 128), (1e-4, 1e-2)]
print("\nTối ưu hóa siêu tham số LSTM bằng thuật toán di truyền (GA)....")
best_params, losses = genetic_algorithm(evaluate_model, bounds)
units, dropout, batch_size, lr = best_params
print(
    f"\n Các tham số tối ưu: units={units}, dropout={dropout:.2f}, batch_size={batch_size}, learning_rate={lr:.5f}"
)

pd.DataFrame(
    {
        "units": [units],
        "dropout": [dropout],
        "batch_size": [batch_size],
        "learning_rate": [lr],
    }
).to_csv("train_results/GA_LSTM_best_params.csv", index=False)

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

plt.figure(figsize=(12, 4))
plt.plot(history.history["loss"], label="Train Loss", color="blue")
plt.plot(history.history["val_loss"], label="Validation Loss", color="orange")
plt.title("GA-LSTM Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("figures/GA_LSTM_training_loss.png")
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

print("\nKết quả đánh giá bộ dữ liệu đào tạo:")
print(f"Train MAE  = {train_mae:.4f}")
print(f"Train MAPE = {train_mape:.2f}%")
print(f"Train RMSE = {train_rmse:.4f}")
print(f"Train R²   = {train_r2:.4f}")

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

print("\Kết quả đánh giá bộ dữ liệu thử nghiệm:")
print(f"Test MAE  = {mae:.4f}")
print(f"Test MAPE = {mape:.2f}%")
print(f"Test RMSE = {rmse:.4f}")
print(f"Test R²   = {r2:.4f}")

plt.figure(figsize=(16, 6))
plt.plot(y_train_inv, label="Actual Train", color="green")
plt.plot(y_train_pred_inv, label="Predicted Train", color="purple")
plt.title("GA-LSTM Train Prediction")
plt.legend()
plt.tight_layout()
plt.savefig("figures/GA_LSTM_train_fit_plot.png")
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(y_test_inv, label="Actual Test", color="blue")
plt.plot(y_pred_inv, label="Predicted Test", color="red")
plt.title("GA-LSTM Test Prediction")
plt.legend()
plt.tight_layout()
plt.savefig("figures/GA_LSTM_test_fit_plot.png")
plt.show()

pd.DataFrame(
    {"Actual_Close_Train": y_train_inv, "Predicted_Close_Train": y_train_pred_inv}
).to_csv("train_results/GA_LSTM_train_prediction.csv", index=False)
pd.DataFrame({"Actual_Close": y_test_inv, "Predicted_Close": y_pred_inv}).to_csv(
    "train_results/GA_LSTM_test_prediction.csv", index=False
)
pd.DataFrame(
    {
        "Dataset": ["Train", "Test"],
        "MAE": [train_mae, mae],
        "MAPE (%)": [train_mape, mape],
        "RMSE": [train_rmse, rmse],
        "R2": [train_r2, r2],
    }
).to_csv("train_results/GA_LSTM_evaluation_metrics.csv", index=False)

model.save("train_results/GA_LSTM_model.h5")
np.save("train_results/y_pred_train_GA_LSTM.npy", y_train_pred_inv)
np.save("train_results/y_train_GA_LSTM.npy", y_train_inv)
np.save("train_results/y_pred_GA_LSTM.npy", y_pred_inv)
np.save("train_results/y_test_GA_LSTM.npy", y_test_inv)
