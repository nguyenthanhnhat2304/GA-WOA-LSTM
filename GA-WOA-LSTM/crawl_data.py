"""
crawl_data.py - Tai du lieu gia co phieu tu Yahoo Finance va luu vao data/raw/.

Co the chay doc lap:
    python crawl_data.py

Hoac duoc goi tu main.py thong qua step_download().
"""

import os
import yfinance as yf


# Danh sach ma co phieu can tai (khong co duoi .VN)
TICKERS = ["VIC", "HPG", "DPM"]
START_DATE = "2010-03-13"
END_DATE = "2025-03-13"

# Thu muc luu du lieu thu
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def download_ticker(ticker, start, end):
    """
    Tai du lieu cua mot ma co phieu va luu thanh file CSV.

    Ham nay thu tai voi duoi .VN truoc, neu that bai thi thu lai voi
    ma goc (khong duoi) de tuong thich voi mot so ma co phieu quoc te.
    """
    vn_ticker = f"{ticker}.VN"
    print(f"Dang tai {vn_ticker} ...")

    data = yf.download(vn_ticker, start=start, end=end)

    if data.empty:
        print(
            f"  [CANH BAO] Khong co du lieu cho {vn_ticker}, thu lai voi {ticker} ..."
        )
        data = yf.download(ticker, start=start, end=end)

    if data.empty:
        print(f"  [BO QUA] Khong the tai du lieu cho ma co '{ticker}'.")
        return

    data = data.round(3)
    output_path = os.path.join(RAW_DIR, f"{ticker}.csv")
    data.to_csv(output_path)
    print(f"  Da luu: {output_path}  ({len(data)} dong)")


if __name__ == "__main__":
    print(f"Bat dau tai du lieu: {START_DATE} -> {END_DATE}")
    print(f"Thu muc luu: {RAW_DIR}")
    print("-" * 50)

    for ticker in TICKERS:
        download_ticker(ticker, START_DATE, END_DATE)

    print("-" * 50)
    print("Hoan tat.")
