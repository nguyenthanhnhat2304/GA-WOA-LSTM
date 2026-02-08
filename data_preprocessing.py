import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump


class SlidingWindowDataLoaderSimple:
    def __init__(
        self, file_path, target_col="Close", window_size=8, save_dir="processed_data"
    ):
        self.file_path = file_path
        self.target_col = target_col
        self.window_size = window_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.data = pd.read_csv(file_path)
        if "Date" not in self.data.columns:
            raise ValueError(" Thiếu cột ngày tháng")
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.data = self.data.sort_values("Date").reset_index(drop=True)

        self.feature_cols = [c for c in self.data.columns if c != "Date"]

        total_len = len(self.data)
        train_size = int(total_len * 0.8)
        self.train_df = self.data.iloc[:train_size].reset_index(drop=True)
        self.test_df = self.data.iloc[train_size:].reset_index(drop=True)

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def _create_sliding_window(self, X: np.ndarray, y: np.ndarray):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.window_size):
            X_seq.append(X[i : i + self.window_size])
            y_seq.append(y[i + self.window_size])
        return np.array(X_seq), np.array(y_seq)

    def process_and_save(self):

        X_train_raw = self.scaler_X.fit_transform(self.train_df[self.feature_cols])
        y_train_raw = self.scaler_y.fit_transform(
            self.train_df[[self.target_col]]
        ).flatten()

        X_test_raw = self.scaler_X.transform(self.test_df[self.feature_cols])
        y_test_raw = self.scaler_y.transform(self.test_df[[self.target_col]]).flatten()

        X_train, y_train = self._create_sliding_window(X_train_raw, y_train_raw)
        X_test, y_test = self._create_sliding_window(X_test_raw, y_test_raw)

        np.save(f"{self.save_dir}/X_train.npy", X_train)
        np.save(f"{self.save_dir}/y_train.npy", y_train)
        np.save(f"{self.save_dir}/X_test.npy", X_test)
        np.save(f"{self.save_dir}/y_test.npy", y_test)
        dump(self.scaler_y, f"{self.save_dir}/scaler.save")

        print(
            f" Việc thi công cửa sổ trượt IMF đã hoàn tất (không bao gồm các hạng mục phát sinh vào ngày T+1)! window_size = {self.window_size}"
        )
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test  shape: {X_test.shape},  y_test  shape: {y_test.shape}")
        print(f" Dữ liệu và scaler_y.pkl đã được lưu vào {self.save_dir}/")


if __name__ == "__main__":
    loader = SlidingWindowDataLoaderSimple(
        file_path="", target_col="Close", window_size=8, save_dir="processed_data"
    )
    loader.process_and_save()
