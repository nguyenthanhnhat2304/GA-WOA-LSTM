import yfinance as yf

tickers = ["VIC.VN", "HPG.VN", "DPM.VN"]

start_date = "2010-03-13"
end_date = "2025-03-13"

for ticker in tickers:
    print(f"Downloading {ticker}...")

    data = yf.download(ticker, start=start_date, end=end_date)

    data = data.round(3)

    filename = f"{ticker.replace('.VN','')}.csv"
    data.to_csv(filename)

    print(f"Saved {filename}")
