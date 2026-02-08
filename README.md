# GA-WOA-LSTM: Mô hình lai tối ưu hóa hyperparameter cho dự báo giá cổ phiếu 📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Giới thiệu

**GA-WOA-LSTM** là một khung (framework) kết hợp **Genetic Algorithm (GA)** và **Whale Optimization Algorithm (WOA)** để tối ưu siêu tham số cho mạng **LSTM**, phục vụ dự báo giá đóng cửa cổ phiếu hàng ngày. Dự án này là phần của luận văn Cử nhân: *"Improving LSTM Hyperparameter Optimization through Genetic and Whale Algorithms: Application to Vietnamese Stock Price Prediction"*.

Mục tiêu chính:

* Giảm thiểu việc dính vào cực trị cục bộ khi tinh chỉnh hyperparameter.
* Kết hợp ưu thế khám phá toàn cục của GA và tinh chỉnh cục bộ của WOA nhằm tìm cấu hình tối ưu cho LSTM.

---

## 🚀 Điểm nổi bật

* **Chiến lược lai (Hybrid):** GA thực hiện khám phá toàn cục; WOA thực hiện khai thác cục bộ để tinh chỉnh.
* **Độ tin cậy:** Đã kiểm thử trên ~15 năm dữ liệu (2010–2025) từ thị trường Việt Nam (VIC, HPG, DPM).
* **Hiệu năng cao:** $R^2$ đạt đến **0.9888** cho mã VIC (theo kết quả luận văn).
* **So sánh toàn diện:** Bao gồm 6 mô hình chuẩn làm benchmark: LSTM, GA-LSTM, WOA-LSTM, RNN, CNN, BP.

---

## 🧰 Công nghệ

* Ngôn ngữ: **Python 3.8+**
* Deep Learning: **TensorFlow / Keras**
* Xử lý dữ liệu: **pandas, numpy, scikit-learn**
* Lấy dữ liệu: **yfinance** (Yahoo Finance)
* Visualizations: **matplotlib**

---

## 📂 Cấu trúc repository (gợi ý)

```
GA-WOA-LSTM/
├─ crawl_data.py            # script crawl dữ liệu (VIC, HPG, DPM)
├─ data_preprocessing.py    # làm sạch, scaling, tạo sliding windows
├─ GA-WOA-LSTM.py           # file chính triển khai pipeline GA -> WOA -> LSTM
├─ GA-LSTM.py               # baseline: LSTM tối ưu bởi GA
├─ WOA-LSTM.py              # baseline: LSTM tối ưu bởi WOA
├─ LSTM.py                  # baseline: LSTM chuẩn
├─ RNN.py                   # baseline: RNN
├─ CNN.py                   # baseline: CNN 1D
├─ BP.py                    # baseline: MLP/BP
├─ requirements.txt         # dependencies
└─ images/
   ├─ flowchart.png         # sơ đồ quy trình (đặt ảnh luồng ở đây)
   └─ result_vic.png        # ảnh kết quả dự báo (ví dụ)
```

> **Lưu ý:** file ảnh luồng quy trình gốc đang có tại: `/mnt/data/fec99984-597f-4324-9ad3-e5e74a11c83e.png`. Bạn có thể copy nó vào `images/flowchart.png` trong repo để hiển thị trong README.

---

## 🔧 Cài đặt

1. Clone repo:

```bash
git clone https://github.com/nguyenthanhnhat2304/GA-WOA-LSTM.git
cd GA-WOA-LSTM
```

2. Tạo virtual environment (khuyến nghị) và cài dependencies:

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

*Nếu chưa có `requirements.txt`, cài tối thiểu:*

```bash
pip install yfinance pandas numpy scikit-learn matplotlib tensorflow openpyxl
```

---

## ⚡ Ví dụ chạy

### 1) Crawl dữ liệu (VIC, HPG, DPM)

```bash
python crawl_data.py --tickers VIC.VN HPG.VN DPM.VN --start 2010-03-13 --end 2025-03-13
```

Kết quả: tạo 3 file CSV `VIC.csv`, `HPG.csv`, `DPM.csv` trong thư mục làm việc.

### 2) Tiền xử lý dữ liệu

```bash
python data_preprocessing.py --input data/VIC.csv --window 8 --output processed/VIC_processed.npz
```

* `window` là kích thước cửa sổ trượt (ví dụ w=8).

### 3) Huấn luyện mô hình GA-WOA-LSTM

```bash
python GA-WOA-LSTM.py --config configs/vic_config.json
```

* `config` chứa siêu tham số ban đầu, kích thước quần thể (GA), số thế hệ, tham số WOA, giới hạn epoch cho LSTM, v.v.

### 4) Lưu và đánh giá

* Kết quả huấn luyện: model đã học, dự báo trên test set (20%), metrics: MAE, RMSE, MAPE, R².
* Các biểu đồ: lưu vào `images/` (ví dụ `result_vic.png`).

---

## 📊 Kết quả mẫu (tóm tắt)

*Bảng ví dụ — kết quả trên tập test của mã VIC (theo báo cáo luận văn):*

| Model           |     MAE    |    RMSE    |    MAPE   |    $R^2$   |
| :-------------- | :--------: | :--------: | :-------: | :--------: |
| **GA-WOA-LSTM** | **403.61** | **623.08** | **1.51%** | **0.9888** |
| GA-LSTM         |   415.20   |   663.10   |   1.53%   |   0.9873   |
| LSTM            |   688.46   |   947.41   |   2.59%   |   0.9742   |
| CNN             |   1749.45  |   1988.51  |   7.22%   |   0.8862   |

> **Chú ý:** Bảng trên là minh họa kết quả luận văn — nếu bạn chạy lại thí nghiệm trên dữ liệu thực tế, kết quả có thể khác (tùy preprocessing, khoảng thời gian, seed ngẫu nhiên...).

---

## 🖼 Hình minh họa

* Đặt ảnh sơ đồ quy trình vào `images/flowchart.png` để hiển thị quy trình data → huấn luyện → dự báo.
* Kết quả dự báo minh họa (ví dụ `images/result_vic.png`).

---

## 📚 Tài liệu tham khảo & chú thích

* Mô tả thuật toán GA và WOA, cũng như tài liệu tham khảo dùng trong luận văn nên được đưa vào file `REFERENCES.md` hoặc phần `docs/` nếu cần trích dẫn chi tiết.

---

## 🤝 Đóng góp

* Hoan nghênh PRs để:

  * Cải thiện pipeline tiền xử lý
  * Thêm mô hình baseline mới
  * Tự động hoá chạy nhiều mã cùng lúc

Vui lòng mở issue trước khi làm PR lớn.

---

## 📝 License

Bộ mã nguồn được cấp phép theo **MIT License**.

---

## ✉ Liên hệ

Nguyễn Thành Nhật — [GitHub](https://github.com/nguyenthanhnhat2304)

---

*Bạn muốn mình thêm hướng dẫn chi tiết cho `configs/*.json` hay mẫu file `requirements.txt` vào README không?*
