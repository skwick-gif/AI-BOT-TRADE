import os
import sys
import subprocess
import json
import time
import argparse
from io import StringIO
from datetime import datetime, timedelta
import re
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
warnings.filterwarnings("ignore", message="Unknown datetime string format", category=UserWarning)

import yfinance as yf

DATA_FOLDER = "stock_data"
os.makedirs(DATA_FOLDER, exist_ok=True)
START_DATE = "2020-01-01"
MAX_TICKERS_PER_BATCH = 50
TODO_FILE = 'todo_tickers.json'
COMPLETED_FILE = 'completed_tickers.json'
HEADERS = {"User-Agent": "Mozilla/5.0"}

session = requests.Session()
session.headers.update(HEADERS)
session.verify = True

def check_internet_connection():
    """בדיקת חיבור אינטרנט"""
    try:
        response = session.get("https://www.google.com", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def check_internet_connection():
    """בדיקת חיבור אינטרנט"""
    try:
        response = session.get("https://www.google.com", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def find_ticker_column(table):
    possible_cols = ['Symbol', 'Ticker symbol', 'Ticker', 'Code']
    for col in possible_cols:
        if col in table.columns:
            return col
    return None

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        response = session.get(url)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        ticker_col = find_ticker_column(tables[0])
        if not ticker_col:
            raise ValueError("Ticker column not found in S&P 500 Wikipedia table")
        tickers = tables[0][ticker_col].tolist()
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        print(f"Error getting S&P 500 tickers: {e}")
        return []

def get_nasdaq100_tickers():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    try:
        response = session.get(url)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        for table in tables:
            ticker_col = find_ticker_column(table)
            if ticker_col:
                tickers = table[ticker_col].tolist()
                return [t.replace('.', '-') for t in tickers]
        raise ValueError("Ticker column not found in NASDAQ-100 Wikipedia tables")
    except Exception as e:
        print(f"Error getting NASDAQ 100 tickers: {e}")
        return []

def get_dowjones_tickers():
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    try:
        response = session.get(url)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        for table in tables:
            ticker_col = find_ticker_column(table)
            if ticker_col:
                tickers = table[ticker_col].tolist()
                return [t.replace('.', '-') for t in tickers]
        raise ValueError("Ticker column not found in Dow Jones Wikipedia tables")
    except Exception as e:
        print(f"Error getting Dow Jones tickers: {e}")
        return []

def fetch_text(url, retries=4, backoff=1.8, timeout=45):
    last = None
    for i in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(backoff ** i)
    raise RuntimeError(f"Failed to fetch {url}: {last}")

def clean_tickers(seq):
    """Normalize and filter ticker symbols.

    - Uppercase and strip whitespace
    - Replace '.' with '-' to match Yahoo format
    - Drop empty, too-short, or invalid symbols
    - Keep only alphanumeric plus '-'
    - Filter out warrants (ending with W), test stocks, and Z-stocks
    """
    cleaned = []
    for s in seq:
        if s is None:
            continue
        t = str(s).strip().upper()
        if not t:
            continue
        # remove trailing markers like '^' or spaces, and unify dotted tickers
        if t.endswith('^'):
            t = t[:-1]
        t = t.split()[0]
        t = t.replace('.', '-')
        # filter invalid
        if len(t) < 2:
            continue
        if not all(ch.isalnum() or ch == '-' for ch in t):
            continue
        # Filter out warrants (ending with W) and test stocks, Z-stocks
        if t.endswith('W') and len(t) > 1:  # Warrants like AAPLW, GOOGW
            continue
        if 'TEST' in t or t.startswith('TEST'):  # Test stocks
            continue
        if t.startswith('Z') or 'ZZ' in t:  # Z-stocks like ZAZZT, ZVZZT
            continue
        cleaned.append(t)
    # unique + stable order
    return sorted(set(cleaned))
def try_one_url(txt):
    try:
        df = pd.read_csv(StringIO(txt), sep="|")
    except Exception as e:
        print(f"Error parsing NYSE csv: {e}")
        return []
    if "ACT Symbol" in df.columns:
        tickers = clean_tickers(df["ACT Symbol"].astype(str).tolist())
    elif "Symbol" in df.columns:
        tickers = clean_tickers(df["Symbol"].astype(str).tolist())
    else:
        print(f"Symbol column not found in NYSE data columns: {df.columns.tolist()}")
        return []
    if len(tickers) > 100:
        return tickers
    print("NY resolver returned too few tickers")
    return []

def get_nasdaq_tickers():
    mirrors = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
    ]
    last_err = None
    for url in mirrors:
        try:
            txt = fetch_text(url)
            try:
                df = pd.read_csv(StringIO(txt), sep="|")
            except Exception as e:
                print(f"Error parsing NASDAQ csv: {e}")
                continue
            if "Symbol" in df.columns:
                tickers = clean_tickers(df["Symbol"].astype(str).tolist())
            else:
                print(f"Symbol column not found in NASDAQ data columns: {df.columns.tolist()}")
                continue
            if len(tickers) > 100:
                print(f"NASDAQ tickers ({len(tickers)}) downloaded from {url}")
                return tickers
            print("NASDAQ resolver returned too few tickers")
        except Exception as e:
            last_err = e
            print(f"Failed to download NASDAQ from {url}: {e}")
            continue
    print(f"Failed all NASDAQ mirrors. Last error: {last_err}")
    return []

def get_nyse_tickers():
    mirrors = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    last_err = None
    for url in mirrors:
        try:
            txt = fetch_text(url)
            tickers = try_one_url(txt)
            if tickers:
                with open("nyse.txt", "w", encoding="utf-8") as f:
                    f.write(", ".join(tickers))
                print(f"NYSE tickers ({len(tickers)}) created from {url}")
                return tickers
        except Exception as e:
            last_err = e
            print(f"Failed to download NYSE from {url}: {e}")
            continue
    print(f"Failed all NYSE mirrors. Last error: {last_err}")
    return []

def find_header_line(csv_text):
    for i, line in enumerate(csv_text.splitlines()):
        if "Ticker" in line.split(",") or "Ticker" in line:
            return i
    return -1

def sniff_delimiter(sample):
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        return dialect.delimiter
    except Exception:
        return ","

def get_russell2000_tickers():
    URL_RUSSELL2000 = ("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
                       "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund")
    try:
        print("Downloading Russell 2000 tickers via official iShares CSV...")
        csv_text = fetch_text(URL_RUSSELL2000)
        with open("IWM_raw.csv", "w", encoding="utf-8") as f:
            f.write(csv_text)
        header_idx = find_header_line(csv_text)
        if header_idx < 0:
            print("Header line with 'Ticker' not found in IWM CSV")
            return []
        trimmed = "\n".join(csv_text.splitlines()[header_idx:])
        delim = sniff_delimiter("\n".join(csv_text.splitlines()[header_idx:header_idx+5]))
        df = pd.read_csv(StringIO(trimmed), sep=delim, engine="python")
        ticker_col = None
        for c in df.columns:
            if str(c).strip().lower() == "ticker":
                ticker_col = c
                break
        if ticker_col is None:
            for c in df.columns:
                if "ticker" in str(c).lower():
                    ticker_col = c
                    break
        if ticker_col is None:
            print(f"Could not find 'Ticker' column in IWM CSV. Columns: {list(df.columns)}")
            return []
        tickers = clean_tickers(df[ticker_col].astype(str).tolist())
        print(f"Russell 2000 tickers downloaded: {len(tickers)}")
        return tickers
    except Exception as e:
        print(f"Error downloading Russell 2000 tickers: {e}")
        return []
    out = []
    blacklisted_patterns = [
        'TEST', 'DUMMY', 'ZZZ', 'XXX', 'TEMP', 'BLANK',
    ]
    blacklisted_exact = [
        'ZAZZT', 'ZBZX', 'ZCZZT', 'ZBZZT', 'ZEXIT', 'ZIEXT', 'ZTEST',
        'ZXIET', 'ZZAZT', 'ZZINT', 'ZZEXT', 'ZZTEST', 'ZZDIV', 'XTSLA',
        'ADRO', 'CAD', 'CBO', 'CBX', 'CRDA', 'FILE', 'GEFB', 'GTXI', 'IGZ', 'INH',
        'MOGA', 'MSFUT', 'P5N994', 'PDLI', 'RTYZ5', 'SBT', 'THE', 'THRD', 'XTSLA',
        'DTSQR', 'IPCXU', 'NOEMR', 'QSEAR', 'WTGUR'
    ]
    for x in seq:
        t = str(x).strip().upper().replace(".", "-")
        if not t or not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z]{1,3})?", t):
            continue
        is_blacklisted = False
        for pattern in blacklisted_patterns:
            if pattern in t:
                is_blacklisted = True
                break
        if is_blacklisted:
            continue
        if t in blacklisted_exact:
            continue
        if t.endswith('W'):
            continue
        if '-' in t:
            continue
        if len(t) < 2:
            continue
        if t.startswith('Z') and len(t) >= 4 and t.endswith('T'):
            continue
        out.append(t)
    filtered_count = len(seq) - len(out)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} problematic tickers")
    return sorted(set(out))

def get_all_tickers():
    sp500 = get_sp500_tickers()
    nasdaq100 = get_nasdaq100_tickers()
    dow30 = get_dowjones_tickers()
    russell2000 = get_russell2000_tickers()
    nyse = get_nyse_tickers()
    nasdaq = get_nasdaq_tickers()
    combined = set(sp500 + nasdaq100 + dow30 + russell2000 + nyse + nasdaq)
    additional_etfs = [
        'DIA', 'QQQ', 'IWM', 'SPY',
        'SDOW', 'SPXL', 'SPXU', 'SQQQ', 'TNA', 'TQQQ', 'TZA', 'UDOW', 'UPRO'
    ]
    combined.update(additional_etfs)
    print(f"Total tickers before filter: {len(combined)}")
    print(f"  S&P 500: {len(sp500)}, NASDAQ-100: {len(nasdaq100)}, Dow: {len(dow30)}")
    print(f"  Russell 2000: {len(russell2000)}, NYSE: {len(nyse)}, NASDAQ: {len(nasdaq)}")
    print(f"  Additional ETFs: {len(additional_etfs)}")
    filtered = [t for t in combined if '-' not in t and len(t) >= 2 and t.isalnum() and 
                not (t.endswith('W') and len(t) > 1) and 'TEST' not in t and not t.startswith('TEST') and
                not t.startswith('Z') and not 'ZZ' in t]
    removed = len(combined) - len(filtered)
    print(f"Tickers after filter: {len(filtered)} (Removed {removed} - filtered out tickers with '-', warrants (W), test stocks, Z-stocks, or invalid)")
    return filtered

def update_price_data(ticker, start_date, folder):
    file_path = os.path.join(folder, ticker, f"{ticker}_price.csv")
    os.makedirs(os.path.join(folder, ticker), exist_ok=True)
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            existing_df = existing_df.dropna(how='all')
            if not existing_df.empty:
                if existing_df.isnull().any().any():
                    print(f"Warning: Missing data found in existing file for {ticker}, re-downloading full range")
                    existing_df = None
                    last_date = None
                else:
                    last_date = existing_df.index.max()
            else:
                last_date = None
        except Exception as e:
            try:
                existing_df = pd.read_csv(file_path)
                date_col = None
                if 'Date' in existing_df.columns:
                    date_col = 'Date'
                elif len(existing_df.columns) > 0:
                    date_col = existing_df.columns[0]
                if date_col:
                    sample_values = existing_df[date_col].dropna().head(5).tolist()
                    valid_dates = 0
                    for val in sample_values:
                        try:
                            pd.to_datetime(str(val))
                            valid_dates += 1
                        except:
                            pass
                    if valid_dates >= 2:
                        try:
                            existing_df[date_col] = pd.to_datetime(existing_df[date_col])
                            existing_df.set_index(date_col, inplace=True)
                            existing_df = existing_df.dropna(how='all')
                            if not existing_df.empty and not existing_df.isnull().any().any():
                                last_date = existing_df.index.max()
                            else:
                                existing_df = None
                                last_date = None
                        except Exception as parse_e:
                            print(f"Warning: Failed to parse dates in existing file for {ticker}: {parse_e} - ignoring old data")
                            existing_df = None
                            last_date = None
                    else:
                        existing_df = None
                        last_date = None
                else:
                    existing_df = None
                    last_date = None
            except Exception as e2:
                print(f"Warning reading existing price file for {ticker}: {e2} - ignoring old data")
                existing_df = None
                last_date = None
    else:
        existing_df = None
        last_date = None
    start_download = start_date
    if last_date:
        start_date_dt = last_date + timedelta(days=1)
        if start_date_dt <= datetime.today():
            start_download = start_date_dt.strftime("%Y-%m-%d")
        else:
            print(f"No new price data for {ticker}.")
            return
    try:
        new_df = yf.download(ticker, start=start_download, progress=False, auto_adjust=True)
        if new_df.empty:
            print(f"No new data for {ticker}.")
            return
        new_df.index = pd.to_datetime(new_df.index)
        new_df.index.name = "Date"

        # Normalize column names - flatten MultiIndex if present
        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = new_df.columns.get_level_values(0)

        # Ensure we have the standard OHLCV columns
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in new_df.columns for col in expected_cols):
            print(f"Warning: Missing expected columns for {ticker}, skipping update")
            return

        # Keep only the expected columns
        new_df = new_df[expected_cols]

        if existing_df is not None:
            # Also normalize existing_df columns if needed
            if isinstance(existing_df.columns, pd.MultiIndex):
                existing_df.columns = existing_df.columns.get_level_values(0)
            if not all(col in existing_df.columns for col in expected_cols):
                print(f"Warning: Existing data missing expected columns for {ticker}, re-downloading full range")
                existing_df = None
                df = new_df
            else:
                existing_df = existing_df[expected_cols]
                df = pd.concat([existing_df, new_df])
                df = df[~df.index.duplicated(keep='last')]
        else:
            df = new_df

        df.to_csv(file_path, index=True)
        print(f"Saved price data for {ticker}")
    except Exception as e:
        print(f"Error downloading price data for {ticker}: {e}")
        raise

def get_html(url, max_retries=3, delay=0.8):
    import time
    import random
    for attempt in range(max_retries):
        try:
            time.sleep(delay + random.uniform(0, 1))
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            response = session.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.HTTPError as e:
            if e.response.status_code in [401, 403, 429]:
                print(f"Access denied for {url} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
                    continue
            else:
                print(f"HTTP error fetching {url}: {e}")
            return ""
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"Connection error for {url} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
                continue
            return ""
        except Exception as e:
            print(f"Unexpected error fetching {url}: {e}")
            return ""
    return ""

if 'original_scrape_yahoo_fundamentals' not in globals():
    def original_scrape_yahoo_fundamentals(ticker):
        return {}

def scrape_yahoo_fundamentals(ticker):
    advanced_file = os.path.join(DATA_FOLDER, ticker, f"{ticker}_advanced.json")
    existing_data = {}
    if os.path.exists(advanced_file):
        try:
            with open(advanced_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"Error reading existing advanced file for {ticker}: {e}")
    data = existing_data.copy()
    yahoo_data = original_scrape_yahoo_fundamentals(ticker)
    for key, value in yahoo_data.items():
        if key not in data or data[key] is None:
            data[key] = value
    if not data.get("Free Cash Flow") or not data.get("EV/EBITDA"):
        additional_data = scrape_additional_fundamentals(ticker)
        for key, value in additional_data.items():
            if key not in data or data[key] is None:
                data[key] = value
    os.makedirs(os.path.join(DATA_FOLDER, ticker), exist_ok=True)
    with open(advanced_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def scrape_finviz_data(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        html = get_html(url)
        if not html:
            raise RuntimeError(f"No HTML content returned for {ticker} from Finviz.")
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_="snapshot-table2")
        if not table:
            raise ValueError(f"No data table found on Finviz for {ticker}.")
        data = {}
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            for i in range(0, len(cols), 2):
                key = cols[i].text.strip()
                value = cols[i + 1].text.strip()
                data[key] = value
        return data
    except Exception as e:
        print(f"Error scraping Finviz data for {ticker}: {e}")
        return {}

def scrape_macrotrends_data(ticker):
    url = f"https://www.macrotrends.net/stocks/charts/{ticker}/financial-ratios"
    try:
        # Use small retries to avoid hanging in constrained SSL environments
        html = get_html(url, max_retries=1, delay=0.2)
        if not html:
            raise RuntimeError(f"No HTML content returned for {ticker} from Macrotrends.")
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            raise ValueError(f"No data tables found on Macrotrends for {ticker}.")
        data = {}
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    key = cols[0].text.strip()
                    value = cols[1].text.strip()
                    data[key] = value
        return data
    except requests.HTTPError as e:
        print(f"HTTP error fetching {url}: {e}")
        return {}
    except Exception as e:
        print(f"Error scraping Macrotrends data for {ticker}: {e}")
        return {}

def scrape_additional_fundamentals(ticker):
    data = {}
    finviz_data = scrape_finviz_data(ticker)
    if finviz_data:
        data.update(finviz_data)
    macrotrends_data = scrape_macrotrends_data(ticker)
    if macrotrends_data:
        data.update(macrotrends_data)
    return data

def scrape_insider_trading(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        html = get_html(url)
        if not html:
            raise RuntimeError(f"No HTML content returned for {ticker} from Finviz.")
        soup = BeautifulSoup(html, "html.parser")
        insider_table = soup.find("table", class_="body-table")
        if not insider_table:
            print(f"No insider trading data found for {ticker}.")
            return {}
        data = []
        rows = insider_table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 7:
                data.append({
                    "Owner": cols[0].text.strip(),
                    "Relationship": cols[1].text.strip(),
                    "Date": cols[2].text.strip(),
                    "Transaction": cols[3].text.strip(),
                    "Cost": cols[4].text.strip(),
                    "Shares": cols[5].text.strip(),
                    "Value": cols[6].text.strip(),
                })
        return {"Insider Trading": data}
    except Exception as e:
        print(f"Error scraping insider trading data for {ticker}: {e}")
        return {}

def scrape_esg_scores(ticker):
    print(f"ESG scores scraping for {ticker} is not implemented yet.")
    return {}

def scrape_options_data(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/options"
    try:
        html = get_html(url)
        if not html:
            raise RuntimeError(f"No HTML content returned for {ticker} from Yahoo Finance.")
        soup = BeautifulSoup(html, "html.parser")
        options_table = soup.find("table", class_="calls")
        if not options_table:
            print(f"No options data found for {ticker}.")
            return {}
        data = []
        rows = options_table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 7:
                data.append({
                    "Contract Name": cols[0].text.strip(),
                    "Last Trade Date": cols[1].text.strip(),
                    "Strike": cols[2].text.strip(),
                    "Last Price": cols[3].text.strip(),
                    "Bid": cols[4].text.strip(),
                    "Ask": cols[5].text.strip(),
                    "Change": cols[6].text.strip(),
                    "Volume": cols[7].text.strip(),
                    "Open Interest": cols[8].text.strip(),
                })
        return {"Options Data": data}
    except Exception as e:
        print(f"Error scraping options data for {ticker}: {e}")
        return {}

def scrape_short_interest(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        html = get_html(url)
        if not html:
            raise RuntimeError(f"No HTML content returned for {ticker} from Finviz.")
        soup = BeautifulSoup(html, "html.parser")
        short_interest = soup.find(string="Short Float")
        if not short_interest:
            print(f"No short interest data found for {ticker}.")
            return {}
        value = short_interest.find_next("td").text.strip()
        return {"Short Interest": value}
    except Exception as e:
        print(f"Error scraping short interest data for {ticker}: {e}")
        return {}

def scrape_dividends(ticker):
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        if not dividends.empty:
            dividend_data = []
            for date, amount in dividends.items():
                dividend_data.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Amount": float(amount)
                })
            return {"Dividends": dividend_data}
        else:
            return {"Dividends": []}
    except Exception as e:
        print(f"Error scraping dividends for {ticker}: {e}")
        return {"Dividends": []}

def scrape_sector_trends(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        html = get_html(url)
        if not html:
            raise RuntimeError(f"No HTML content returned for {ticker} from Finviz.")
        soup = BeautifulSoup(html, "html.parser")
        sector = soup.find(string="Sector")
        industry = soup.find(string="Industry")
        if not sector or not industry:
            print(f"No sector or industry data found for {ticker}.")
            return {}
        sector_value = sector.find_next("td").text.strip()
        industry_value = industry.find_next("td").text.strip()
        return {"Sector": sector_value, "Industry": industry_value}
    except Exception as e:
        print(f"Error scraping sector trends for {ticker}: {e}")
        return {}

def scrape_all_data(ticker, folder):
    os.makedirs(os.path.join(folder, ticker), exist_ok=True)
    json_path = os.path.join(folder, ticker, f"{ticker}_advanced.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
            if len(existing_data) >= 10:
                return True
        except:
            pass
    print(f"Scraping advanced data for {ticker}")
    data = {}
    try:
        data.update(scrape_yahoo_fundamentals(ticker))
    except Exception as e:
        print(f"Error scraping Yahoo fundamentals for {ticker}: {e}")
    try:
        data.update(scrape_finviz_data(ticker))
    except Exception as e:
        print(f"Error scraping Finviz data for {ticker}: {e}")
    try:
        data.update(scrape_macrotrends_data(ticker))
    except Exception as e:
        print(f"Error scraping Macrotrends data for {ticker}: {e}")
    try:
        data.update(scrape_additional_fundamentals(ticker))
    except Exception as e:
        print(f"Error scraping additional fundamentals for {ticker}: {e}")
    try:
        data.update(scrape_insider_trading(ticker))
    except Exception as e:
        print(f"Error scraping insider trading for {ticker}: {e}")
    try:
        data.update(scrape_options_data(ticker))
    except Exception as e:
        print(f"Error scraping options data for {ticker}: {e}")
    try:
        data.update(scrape_short_interest(ticker))
    except Exception as e:
        print(f"Error scraping short interest for {ticker}: {e}")
    try:
        data.update(scrape_dividends(ticker))
    except Exception as e:
        print(f"Error scraping dividends for {ticker}: {e}")
    try:
        data.update(scrape_sector_trends(ticker))
    except Exception as e:
        print(f"Error scraping sector trends for {ticker}: {e}")
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved advanced data for {ticker}")
        return True
    except Exception as e:
        print(f"Error saving data for {ticker}: {e}")
        return False

def process_tickers_daily(limit: int | None = None):
    print("Starting daily ticker processing...")
    all_tickers = get_all_tickers()
    print(f"Found {len(all_tickers)} total tickers")
    try:
        with open(TODO_FILE, 'r') as f:
            todo = json.load(f)
    except:
        todo = all_tickers.copy()
    try:
        with open(COMPLETED_FILE, 'r') as f:
            completed = json.load(f)
    except:
        completed = []
    print(f"TODO: {len(todo)}, Completed: {len(completed)}")
    tickers_to_check = list(set(todo + completed))
    if limit:
        tickers_to_check = tickers_to_check[:max(1, int(limit))]
    print(f"Checking {len(tickers_to_check)} tickers for updates")
    updated_prices = 0
    updated_advanced = 0
    for ticker in tickers_to_check:
        try:
            price_file = os.path.join(DATA_FOLDER, ticker, f"{ticker}_price.csv")
            needs_price_update = True
            if os.path.exists(price_file):
                try:
                    df = pd.read_csv(price_file, parse_dates=['Date'])
                    last_date = df['Date'].max()
                    days_since_update = (datetime.now() - last_date.tz_localize(None) if last_date.tz else datetime.now() - last_date).days
                    has_data = not df[['Open', 'High', 'Low', 'Close']].isnull().all().all()
                    # Allow data to be up to 2 days old (accounting for weekends and market delays)
                    if days_since_update <= 2 and has_data:
                        needs_price_update = False
                except:
                    pass
            if needs_price_update:
                print(f"Updating price data for {ticker}")
                update_price_data(ticker, START_DATE, DATA_FOLDER)
                updated_prices += 1
            advanced_file = os.path.join(DATA_FOLDER, ticker, f"{ticker}_advanced.json")
            needs_advanced_update = True
            if os.path.exists(advanced_file):
                try:
                    with open(advanced_file, 'r') as f:
                        data = json.load(f)
                    if len(data) >= 10:
                        needs_advanced_update = False
                except:
                    pass
            if needs_advanced_update:
                print(f"Updating advanced data for {ticker}")
                if scrape_all_data(ticker, DATA_FOLDER):
                    updated_advanced += 1
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    with open(TODO_FILE, 'w') as f:
        json.dump(todo, f)
    with open(COMPLETED_FILE, 'w') as f:
        json.dump(completed, f)
    print(f"Daily processing completed. Updated {updated_prices} price data, {updated_advanced} advanced data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily stock data updater")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers to process")
    args = parser.parse_args()
    process_tickers_daily(limit=args.limit)
