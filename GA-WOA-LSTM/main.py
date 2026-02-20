"""
main.py - Diem bat dau chay toan bo pipeline du bao gia co phieu.

Cach dung:
    python main.py [--ticker TICKER] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                   [--window N] [--model TEN_MODEL] [--skip-data]

Vi du:
    # Chay toan bo pipeline voi ma co VIC
    python main.py --ticker VIC

    # Chi train GA-WOA-LSTM, bo qua buoc tai du lieu
    python main.py --ticker VIC --model GA-WOA-LSTM --skip-data

    # Xem tat ca tuy chon
    python main.py --help
"""

import argparse
import os
import sys
import subprocess


# --------------------------------------------------------------------
# Thu muc lam viec
# --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for _d in [DATA_RAW_DIR, DATA_PROC_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------
# Danh sach mo hinh ho tro
# --------------------------------------------------------------------
AVAILABLE_MODELS = ["BP", "LSTM", "RNN", "CNN", "GA-LSTM", "WOA-LSTM", "GA-WOA-LSTM"]

# Ten mo hinh -> ten file script tuong ung
MODEL_SCRIPTS = {model: f"{model}.py" for model in AVAILABLE_MODELS}


# --------------------------------------------------------------------
# Ham tien ich
# --------------------------------------------------------------------


def separator(title=""):
    """In mot dong ke ngan de phan chia cac buoc trong console."""
    line = "-" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


def run_script(script_name, extra_env=None):
    """
    Chay mot script Python khac (cung cap trong cung thu muc) duoi dang subprocess.

    Truyen them bien moi truong qua extra_env de script con biet duong dan
    den du lieu va ket qua.
    """
    script_path = os.path.join(BASE_DIR, script_name)

    if not os.path.isfile(script_path):
        print(f"  [SKIP] Khong tim thay file: {script_path}")
        return

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    separator(f"Dang chay: {script_name}")
    result = subprocess.run([sys.executable, script_path], env=env)

    if result.returncode != 0:
        print(f"\n  [LOI] {script_name} ket thuc voi ma loi {result.returncode}.")
        sys.exit(result.returncode)


# --------------------------------------------------------------------
# Cac buoc chinh trong pipeline
# --------------------------------------------------------------------


def step_download(ticker, start_date, end_date):
    """
    Buoc 1: Tai du lieu gia co phieu tu Yahoo Finance va luu vao data/raw/.

    Tra ve duong dan file CSV vua luu.
    """
    separator(f"Buoc 1 - Tai du lieu: {ticker} ({start_date} den {end_date})")

    try:
        import yfinance as yf
    except ImportError:
        print("  [LOI] Chua cai yfinance. Chay: pip install yfinance")
        sys.exit(1)

    vn_ticker = f"{ticker}.VN"
    print(f"  Dang tai {vn_ticker} ...")
    data = yf.download(vn_ticker, start=start_date, end=end_date)

    if data.empty:
        print(
            f"  [CANH BAO] Khong co du lieu cho {vn_ticker}. Thu lai voi ma {ticker} ..."
        )
        data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        print(f"  [LOI] Khong the tai du lieu cho ma co '{ticker}'.")
        sys.exit(1)

    data = data.round(3)
    output_path = os.path.join(DATA_RAW_DIR, f"{ticker}.csv")
    data.to_csv(output_path)
    print(f"  Da luu: {output_path}  ({len(data)} dong)")
    return output_path


def step_preprocess(csv_path, window_size):
    """
    Buoc 2: Tien xu ly du lieu bang sliding window va luu ket qua vao data/processed/.
    """
    file_name = os.path.basename(csv_path)
    separator(f"Buoc 2 - Tien xu ly: {file_name}  (window = {window_size})")

    try:
        from data_preprocessing import SlidingWindowDataLoaderSimple
    except ImportError as e:
        print(f"  [LOI] Khong the import data_preprocessing: {e}")
        sys.exit(1)

    loader = SlidingWindowDataLoaderSimple(
        file_path=csv_path,
        target_col="Close",
        window_size=window_size,
        save_dir=DATA_PROC_DIR,
    )
    loader.process_and_save()
    print(f"  Du lieu da duoc luu vao: {DATA_PROC_DIR}")


def step_train(models_to_run):
    """
    Buoc 3: Train lan luot tung mo hinh trong danh sach.

    Cac script con nhan duong dan qua bien moi truong de khong phu thuoc
    vao thu muc lam viec hien tai.
    """
    extra_env = {
        "GAWOA_DATA_DIR": DATA_PROC_DIR,
        "GAWOA_MODELS_DIR": MODELS_DIR,
        "GAWOA_RESULTS_DIR": RESULTS_DIR,
    }
    for model_name in models_to_run:
        script = MODEL_SCRIPTS[model_name]
        run_script(script, extra_env=extra_env)


# --------------------------------------------------------------------
# Xu ly tham so dong lenh
# --------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline du bao gia co phieu bang GA-WOA-LSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ticker",
        default="VIC",
        help="Ma co phieu (mac dinh: VIC)",
    )
    parser.add_argument(
        "--start",
        default="2010-03-13",
        help="Ngay bat dau tai du lieu, dinh dang YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        default="2025-03-13",
        help="Ngay ket thuc tai du lieu, dinh dang YYYY-MM-DD",
    )
    parser.add_argument(
        "--window",
        default=8,
        type=int,
        help="Kich thuoc cua so truot (mac dinh: 8)",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all"] + AVAILABLE_MODELS,
        help="Mo hinh can train (mac dinh: all)",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Bo qua buoc tai va tien xu ly du lieu, dung du lieu da co san",
    )
    return parser.parse_args()


# --------------------------------------------------------------------
# Ham chinh
# --------------------------------------------------------------------


def main():
    args = parse_args()

    models_to_run = AVAILABLE_MODELS if args.model == "all" else [args.model]

    print("\nGA-WOA-LSTM - Pipeline du bao gia co phieu")
    separator()
    print(f"  Ma co phieu : {args.ticker}")
    print(f"  Giai doan   : {args.start} -> {args.end}")
    print(f"  Cua so truot: {args.window}")
    print(f"  Mo hinh     : {', '.join(models_to_run)}")
    print(f"  Bo qua data : {'Co' if args.skip_data else 'Khong'}")

    # Buoc 1 & 2: Tai du lieu va tien xu ly
    if not args.skip_data:
        csv_path = step_download(args.ticker, args.start, args.end)
        step_preprocess(csv_path, args.window)
    else:
        print("\n  [THONG TIN] Bo qua buoc tai va tien xu ly (--skip-data).")
        required_files = [
            "X_train.npy",
            "X_test.npy",
            "y_train.npy",
            "y_test.npy",
            "scaler.save",
        ]
        missing = [
            f
            for f in required_files
            if not os.path.isfile(os.path.join(DATA_PROC_DIR, f))
        ]
        if missing:
            print(f"  [LOI] Thieu cac file sau trong {DATA_PROC_DIR}:")
            for f in missing:
                print(f"         - {f}")
            print("  Hay chay lai ma khong co --skip-data de tao cac file nay.")
            sys.exit(1)

    # Buoc 3: Train cac mo hinh
    separator(f"Buoc 3 - Train {len(models_to_run)} mo hinh")
    step_train(models_to_run)

    separator()
    print("Pipeline hoan tat.")
    print(f"Ket qua duoc luu tai: {RESULTS_DIR}")
    print()


if __name__ == "__main__":
    main()
