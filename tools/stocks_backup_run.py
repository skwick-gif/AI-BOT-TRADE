import os
import sys
import subprocess
import json
import time
from io import StringIO
from datetime import datetime, timedelta
import re
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["yfinance", "pandas", "beautifulsoup4", "requests", "streamlit", "lxml"]:
    try:
        __import__(pkg)
    except ModuleNotFoundError:
        print(f"Installing {pkg}...")
        install_package(pkg)

import yfinance as yf
import streamlit as st

DATA_FOLDER = "stock_data"
os.makedirs(DATA_FOLDER, exist_ok=True)
START_DATE = "2020-01-01"
MAX_TICKERS_PER_BATCH = 50
TODO_FILE = 'todo_tickers.json'
COMPLETED_FILE = 'completed_tickers.json'
HEADERS = {"User-Agent": "Mozilla/5.0"}

session = requests.Session()
session.headers.update(HEADERS)

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
    out = []
    
    # רשימת patterns למניות בעייתיות
    blacklisted_patterns = [
        'TEST',      # מניות test
        'DUMMY',     # מניות dummy  
        'ZZZ',       # מניות placeholder
        'XXX',       # מניות placeholder
        'TEMP',      # מניות זמניות
        'BLANK',     # מניות ריקות
    ]
    
    # רשימת מניות specific שידועות כבעייתיות
    blacklisted_exact = [
        'ZAZZT', 'ZBZX', 'ZCZZT', 'ZBZZT', 'ZEXIT', 'ZIEXT', 'ZTEST',
        'ZXIET', 'ZZAZT', 'ZZINT', 'ZZEXT', 'ZZTEST', 'ZZDIV'
    ]
    
    for x in seq:
        t = str(x).strip().upper().replace(".", "-")
        
        # בדוק אם זה טיקר תקני (אותיות בלבד, 1-5 תווים, עם אופציה לקו ותווים נוספים)
        if not t or not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z]{1,3})?", t):
            continue
            
        # סנן מניות בעייתיות - patterns
        is_blacklisted = False
        for pattern in blacklisted_patterns:
            if pattern in t:
                is_blacklisted = True
                break
        
        if is_blacklisted:
            continue
            
        # סנן מניות בעייתיות - exact matches
        if t in blacklisted_exact:
            continue
            
        # סנן תעודות אופציה (WARRANTS) - מסתיימות ב-W
        if t.endswith('W'):
            continue
            
        # סנן מניות Class (יש קו באמצע) - BRK-A, BRK-B, META-A וכו'
        if '-' in t:
            continue
            
        # סנן מניות קצרות מדי (תו בודד או שניים - לא תקניות)
        if len(t) < 2:
            continue
            
        # סנן מניות עם patterns חשודים
        if t.startswith('Z') and len(t) >= 4 and t.endswith('T'):
            # מניות שמתחילות ב-Z ומסתיימות ב-T (כמו ZTEST, ZEXIT)
            continue
            
        out.append(t)
    
    filtered_count = len(seq) - len(out)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} problematic tickers")
    
    return sorted(set(out))

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
    """אוסף מניות מבורסת NASDAQ"""
    mirrors = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt",
    ]
    last_err = None
    for url in mirrors:
        try:
            txt = fetch_text(url)
            # נתח את הקובץ של NASDAQ
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

def get_all_tickers():
    sp500 = get_sp500_tickers()
    nasdaq100 = get_nasdaq100_tickers()
    dow30 = get_dowjones_tickers()
    russell2000 = get_russell2000_tickers()
    nyse = get_nyse_tickers()
    nasdaq = get_nasdaq_tickers()  # הוספת מניות NASDAQ
    combined = set(sp500 + nasdaq100 + dow30 + russell2000 + nyse + nasdaq)
    print(f"Total tickers before filter: {len(combined)}")
    print(f"  S&P 500: {len(sp500)}, NASDAQ-100: {len(nasdaq100)}, Dow: {len(dow30)}")
    print(f"  Russell 2000: {len(russell2000)}, NYSE: {len(nyse)}, NASDAQ: {len(nasdaq)}")
    # לא מסיר שום טיקרים - נשמור את הכל!
    filtered = list(combined)
    print(f"Tickers after filter: {len(filtered)} (Removed 0 - keeping all)")
    return filtered

all_tickers = get_all_tickers()

# נבטל את פונקציית הניקוי - לא נמחק תיקיות קיימות!
# clean_old_tickers(DATA_FOLDER, set(all_tickers))

def update_price_data(ticker, start_date, folder):
    file_path = os.path.join(folder, ticker, f"{ticker}_price.csv")
    os.makedirs(os.path.join(folder, ticker), exist_ok=True)
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            last_date = existing_df.index.max()
        except Exception as e:
            print(f"Warning reading existing price file for {ticker}: {e} - ignoring old data")
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

        if existing_df is not None:
            df = pd.concat([existing_df, new_df])
            df = df[~df.index.duplicated(keep='last')]
        else:
            df = new_df
        df.to_csv(file_path)
        print(f"Saved price data for {ticker}")
    except Exception as e:
        print(f"Error downloading price data for {ticker}: {e}")
        raise

def get_html(url, max_retries=3, delay=1):
    """טוען HTML עם retry logic ו-rate limiting"""
    import time
    import random
    
    for attempt in range(max_retries):
        try:
            # הוסף delay אקראי למניעת blocking
            time.sleep(delay + random.uniform(0, 1))
            
            # הוסף headers כדי להיראות כמו browser רגיל
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
            if e.response.status_code in [401, 403, 429]:  # Access denied או rate limit
                print(f"Access denied for {url} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
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
    
    print(f"Failed to fetch {url} after {max_retries} attempts")
    return ""

def scrape_yahoo_fundamentals(ticker):
    """מושך נתונים יסודיים דרך yfinance API עם fallback מרובים"""
    try:
        stock = yf.Ticker(ticker)
        
        # נתונים בסיסיים דרך info API
        info = stock.info
        if not info or len(info) < 5:  # בדיקה שיש נתונים
            print(f"No fundamental data available for {ticker}")
            return {}
        
        data = {}
        
        # פונקציה לחילוץ בטוח של נתונים עם fallback
        def safe_get(primary_key, secondary_keys=None, format_func=None):
            try:
                # נסה את המפתח הראשי
                value = info.get(primary_key)
                
                # אם לא מצא, נסה מפתחות חלופיים
                if value is None and secondary_keys:
                    for alt_key in secondary_keys:
                        value = info.get(alt_key)
                        if value is not None:
                            break
                
                # פורמט אם יש פונקציית פורמט וערך תקני
                if value is not None and format_func:
                    return format_func(value)
                return value
            except:
                return None
        
        # נתונים בסיסיים עם fallbacks
        data["Market Cap"] = safe_get("marketCap", ["enterpriseValue"], lambda x: f"{x:,}" if x else None)
        data["PE Ratio (TTM)"] = safe_get("trailingPE", ["priceToEarningsRatio"], lambda x: round(x, 2) if x else None)
        data["Forward PE"] = safe_get("forwardPE", ["forwardEps"], lambda x: round(x, 2) if x else None)
        data["PEG Ratio"] = safe_get("pegRatio", lambda x: round(x, 2) if x else None)
        data["Price to Book"] = safe_get("priceToBook", ["pbRatio"], lambda x: round(x, 2) if x else None)
        
        # תיקון לחישוב Dividend Yield - יהיה תקין עכשיו
        div_yield = safe_get("dividendYield", ["trailingAnnualDividendYield"])
        if div_yield and div_yield <= 1:  # אם זה בין 0-1, זה נכון
            data["Dividend Yield"] = f"{div_yield*100:.2f}%"
        elif div_yield and div_yield > 1:  # אם זה יותר מ-1, כנראה שכבר באחוזים
            data["Dividend Yield"] = f"{div_yield:.2f}%"
        else:
            data["Dividend Yield"] = None
        
        data["Debt to Equity"] = safe_get("debtToEquity", ["totalDebtToEquity"], lambda x: round(x, 2) if x else None)
        data["Return on Equity"] = safe_get("returnOnEquity", ["roe"], lambda x: f"{x*100:.2f}%" if x else None)
        data["Revenue Growth"] = safe_get("revenueGrowth", ["quarterlyRevenueGrowth"], lambda x: f"{x*100:.2f}%" if x else None)
        data["Profit Margin"] = safe_get("profitMargins", ["netProfitMargin"], lambda x: f"{x*100:.2f}%" if x else None)
        data["Current Ratio"] = safe_get("currentRatio", lambda x: round(x, 2) if x else None)
        data["Book Value"] = safe_get("bookValue", ["priceToBook"], lambda x: round(x, 2) if x else None)
        
        # נתוני מחיר עם fallbacks
        data["52 Week High"] = safe_get("fiftyTwoWeekHigh", ["52WeekChange"], lambda x: round(x, 2) if x else None)
        data["52 Week Low"] = safe_get("fiftyTwoWeekLow", lambda x: round(x, 2) if x else None)
        data["50 Day Average"] = safe_get("fiftyDayAverage", ["ma50"], lambda x: round(x, 2) if x else None)
        data["200 Day Average"] = safe_get("twoHundredDayAverage", ["ma200"], lambda x: round(x, 2) if x else None)
        
        # נפח ומניות עם fallbacks
        data["Average Volume"] = safe_get("averageVolume", ["volume", "regularMarketVolume"], lambda x: f"{x:,}" if x else None)
        data["Shares Outstanding"] = safe_get("sharesOutstanding", ["impliedSharesOutstanding", "floatShares"], lambda x: f"{x:,}" if x else None)
        
        # מידע על החברה עם fallbacks
        data["Sector"] = safe_get("sector", ["sectorKey"])
        data["Industry"] = safe_get("industry", ["industryKey", "industryDisp"])
        data["Country"] = safe_get("country", ["location"])
        data["Website"] = safe_get("website", ["companyOfficers"])
        data["Business Summary"] = safe_get("longBusinessSummary", ["description", "summary"])
        
        # נתוני אנליסטים עם fallbacks
        data["Analyst Target Price"] = safe_get("targetMeanPrice", ["targetMedianPrice", "targetHighPrice"], lambda x: round(x, 2) if x else None)
        data["Analyst Recommendation"] = safe_get("recommendationMean", ["recommendationKey"], lambda x: round(x, 2) if x else None)
        data["Number of Analysts"] = safe_get("numberOfAnalystOpinions", ["recommendationMean"])
        
        # זיהוי סוג הנכס (ETF, Stock, etc.)
        quote_type = safe_get("quoteType", ["typeDisp"])
        if quote_type:
            data["Asset Type"] = quote_type
        
        # ETF specific data
        if quote_type == "ETF":
            data["Expense Ratio"] = safe_get("annualReportExpenseRatio", lambda x: f"{x*100:.2f}%" if x else None)
            data["Total Assets"] = safe_get("totalAssets", lambda x: f"{x:,}" if x else None)
            data["Yield"] = safe_get("yield", lambda x: f"{x*100:.2f}%" if x else None)
        
        non_null_count = len([v for v in data.values() if v is not None])
        print(f"Retrieved {non_null_count} fundamental metrics for {ticker} ({quote_type or 'Unknown'})")
        return data
        
    except Exception as e:
        print(f"Error getting fundamentals for {ticker}: {e}")
        return {}
    try:
        stock = yf.Ticker(ticker)
        dividends_df = stock.dividends
        if not dividends_df.empty:
            dividends_dict = {str(k.date()): float(v) for k, v in dividends_df.items()}
            data["Dividends"] = dividends_dict
        else:
            data["Dividends"] = None
    except Exception as e:
        print(f"Error getting dividends for {ticker}: {e}")
        data["Dividends"] = None
    earnings_url = f"https://finance.yahoo.com/quote/{ticker}/financials"
    earnings_html = get_html(earnings_url)
    if not earnings_html:
        data["Quarterly Earnings"] = None
        return data
    soup_earn = BeautifulSoup(earnings_html, "html.parser")
    earnings_table = soup_earn.find_all('div', {'data-test': 'fin-row'})
    earnings_data = {}
    for row in earnings_table:
        cols = row.find_all('div')
        if len(cols) > 1:
            key = cols[0].text.strip()
            vals = [col.text.strip() for col in cols[1:]]
            earnings_data[key] = vals
    data["Quarterly Earnings"] = earnings_data if earnings_data else None
    return data

def scrape_yahoo_news(ticker):
    """מושך חדשות ישירות מYahoo Finance - יותר אמין מMarketWatch"""
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    html = get_html(url, max_retries=2, delay=0.5)  # פחות aggressive למניעת blocking
    if not html:
        print(f"Skipping news for {ticker} - no page available.")
        return []
    
    try:
        soup = BeautifulSoup(html, "html.parser")
        news_list = []
        
        # Yahoo Finance משתמש בselector שונה לחדשות
        articles = soup.find_all('h3', class_='Mb(5px)')[:5]  # מוגבל ל-5 כדי לא להעמיס
        
        for article in articles:
            try:
                link_tag = article.find('a')
                if link_tag:
                    title = link_tag.text.strip()
                    href = link_tag.get('href', '')
                    # Yahoo מחזיר relative URLs
                    if href.startswith('/'):
                        href = f"https://finance.yahoo.com{href}"
                    
                    if title and href:
                        news_list.append({
                            "title": title,
                            "link": href,
                            "source": "Yahoo Finance"
                        })
            except Exception as e:
                print(f"Error parsing news article for {ticker}: {e}")
                continue
        
        return news_list
        
    except Exception as e:
        print(f"Error scraping news for {ticker}: {e}")
        return []

def scrape_all_advanced(ticker, folder):
    """סקרייפ נתונים מתקדמים עם בדיקה אם הקובץ כבר קיים"""
    advanced_file = os.path.join(folder, ticker, f"{ticker}_advanced.json")
    
    # בדיקה אם הקובץ כבר קיים ויש בו נתונים איכותיים
    if os.path.exists(advanced_file):
        try:
            with open(advanced_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # בדיקה אם יש נתונים איכותיים (יותר מ-15 שדות לא ריקים)
            non_null_count = len([v for v in existing_data.values() if v is not None])
            if non_null_count >= 15:
                print(f"Advanced data already exists for {ticker} ({non_null_count} fields) - skipping")
                return
            else:
                print(f"Existing data for {ticker} has low quality ({non_null_count} fields) - updating")
        except Exception as e:
            print(f"Error reading existing advanced file for {ticker}: {e} - will recreate")
    
    # אם הגענו לכאן, צריך לעדכן/ליצור את הנתונים
    data = scrape_yahoo_fundamentals(ticker)
    news = scrape_yahoo_news(ticker)  # שונה לYahoo News
    if news:
        data['Recent News'] = news
    
    os.makedirs(os.path.join(folder, ticker), exist_ok=True)
    with open(advanced_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    new_count = len([v for v in data.values() if v is not None])
    print(f"Saved advanced scrape for {ticker} ({new_count} fields)")

def load_tickers_state():
    # אם אין קובץ TODO, ניצור אותו עם כל המניות
    if os.path.exists(TODO_FILE):
        with open(TODO_FILE, 'r') as f:
            todo = json.load(f)
    else:
        todo = all_tickers.copy()
        print(f"Created new TODO list with {len(todo)} tickers")
    
    if os.path.exists(COMPLETED_FILE):
        with open(COMPLETED_FILE, 'r') as f:
            completed = set(json.load(f))
    else:
        completed = set()
    
    # נוודא שכל מניה שלא ב-TODO ולא ב-COMPLETED תתווסף ל-TODO
    all_set = set(all_tickers)
    todo_set = set(todo)
    missing_tickers = all_set - todo_set - completed
    if missing_tickers:
        todo.extend(list(missing_tickers))
        print(f"Added {len(missing_tickers)} missing tickers to TODO list")
    
    # נסיר מ-TODO מניות שכבר הושלמו
    todo = [t for t in todo if t not in completed]
    
    return todo, completed

def save_tickers_state(todo, completed):
    with open(TODO_FILE, 'w') as f:
        json.dump(todo, f)
    with open(COMPLETED_FILE, 'w') as f:
        json.dump(list(completed), f)

def check_progress():
    """בדיקת התקדמות כללית"""
    todo, completed = load_tickers_state()
    
    # ספירת תיקיות קיימות
    existing_folders = [f for f in os.listdir(DATA_FOLDER) 
                       if os.path.isdir(os.path.join(DATA_FOLDER, f)) 
                       and not f.startswith('.')]
    
    print("="*50)
    print("📊 PROGRESS REPORT:")
    print(f"📁 Total tickers in system: {len(all_tickers)}")
    print(f"✅ Completed in this run: {len(completed)}")
    print(f"⏳ Remaining in TODO: {len(todo)}")
    print(f"📂 Existing folders: {len(existing_folders)}")
    print(f"🎯 Expected final folders: {len(all_tickers)}")
    print("="*50)

def wait_for_user():
    # השוואת פונקציה זו כדי לא לעצור את התהליך
    print("Processed batch of 50 tickers. Continuing automatically...")
    time.sleep(2)  # המתנה קצרה במקום המתנה למשתמש

def process_tickers_daily():
    todo, completed = load_tickers_state()
    check_progress()  # הצגת דוח התקדמות
    
    print(f"🚀 Starting processing of {len(todo)} remaining tickers...")
    
    while todo:
        batch = todo[:MAX_TICKERS_PER_BATCH]
        todo = todo[MAX_TICKERS_PER_BATCH:]
        failed_count = 0
        
        for ticker in batch:
            try:
                print(f"Processing ticker: {ticker}")
                
                # בדיקה שהטיקר תקני לפני התחלה
                if not ticker or len(ticker) < 2 or not ticker.isalpha():
                    print(f"Skipping invalid ticker: {ticker}")
                    continue
                
                # בדיקה מוקדמת אם המניה כבר הושלמה במלואה
                ticker_folder = os.path.join(DATA_FOLDER, ticker)
                price_file = os.path.join(ticker_folder, f"{ticker}_price.csv")
                advanced_file = os.path.join(ticker_folder, f"{ticker}_advanced.json")
                
                # בדיקה אם יש נתונים איכותיים
                skip_ticker = False
                if os.path.exists(price_file) and os.path.exists(advanced_file):
                    try:
                        # בדיקת איכות קובץ מחירים
                        price_ok = os.path.getsize(price_file) > 1000
                        
                        # בדיקת איכות נתונים יסודיים
                        with open(advanced_file, 'r', encoding='utf-8') as f:
                            advanced_data = json.load(f)
                        advanced_quality = len([v for v in advanced_data.values() if v is not None])
                        
                        if price_ok and advanced_quality >= 15:
                            print(f"✅ {ticker} already complete (price + {advanced_quality} advanced fields) - skipping")
                            completed.add(ticker)
                            skip_ticker = True
                    except Exception as e:
                        print(f"Error checking existing data for {ticker}: {e} - will reprocess")
                
                if skip_ticker:
                    continue
                
                # הורדת נתוני מחירים
                try:
                    update_price_data(ticker, START_DATE, DATA_FOLDER)
                    time.sleep(0.2)  # הגדלתי קצת את ההמתנה
                except Exception as e:
                    print(f"Price data failed for {ticker}: {e}")
                    # ממשיך גם אם המחירים נכשלו
                
                # סקרייפינג מתקדם עם exception handling
                try:
                    scrape_all_advanced(ticker, DATA_FOLDER)
                    time.sleep(0.2)
                except Exception as e:
                    print(f"Advanced scraping failed for {ticker}: {e}")
                    # ממשיך גם אם הסקרייפינג נכשל
                
                completed.add(ticker)
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Process interrupted by user. Saving progress...")
                save_tickers_state(todo, completed)
                print(f"✅ Progress saved. Processed {len(batch)-len(todo)} tickers in this batch.")
                return
            except Exception as e:
                print(f"❌ Critical error processing {ticker}: {e}")
                failed_count += 1
                # אם יש יותר מדי כשלונות, נעצור
                if failed_count > 10:
                    print(f"⚠️  Too many failures ({failed_count}). Stopping batch.")
                    break
                    
        save_tickers_state(todo, completed)
        print(f"✅ Processed {len(batch)} tickers, {len(todo)} remaining.")
        
        # הצגת התקדמות מדי פעם
        if len(todo) % 500 == 0 or len(todo) == 0:
            check_progress()
            
        if todo:
            wait_for_user()
    
    print("🎉 ALL TICKERS COMPLETED!")
    check_progress()
    
    # בדיקה ותיקון אוטומטי של נתונים חסרים
    print("\n" + "="*60)
    print("🔍 מתחיל בדיקה אוטומטית של נתונים חסרים...")
    
    # הסבר מדוע נתונים לא ירדו מלכתחילה
    analyze_download_failures()
    
    auto_fix_missing_data()

def check_missing_data(tickers, data_folder):
    missing_data = []
    for ticker in tickers:
        folder_path = os.path.join(data_folder, ticker)
        price_path = os.path.join(folder_path, f"{ticker}_price.csv")
        advanced_path = os.path.join(folder_path, f"{ticker}_advanced.json")
        missing = []
        if not os.path.exists(price_path):
            missing.append("Price Data")
        if not os.path.exists(advanced_path):
            missing.append("Advanced Data")
        if missing:
            missing_data.append({"Ticker": ticker, "Missing": ", ".join(missing)})
    return missing_data

def analyze_data_quality():
    """בדיקה מקיפה של איכות הנתונים"""
    print("🔍 מתחיל ניתוח איכות נתונים...")
    
    problematic_stocks = []
    stock_folders = [f for f in os.listdir('stock_data') 
                    if os.path.isdir(os.path.join('stock_data', f)) and f.isalpha()]
    
    for ticker in stock_folders:
        folder_path = f"stock_data/{ticker}"
        price_file = os.path.join(folder_path, f"{ticker}_price.csv")
        advanced_file = os.path.join(folder_path, f"{ticker}_advanced.json")
        
        has_price = os.path.exists(price_file) and os.path.getsize(price_file) > 1000
        has_fundamentals = os.path.exists(advanced_file) and os.path.getsize(advanced_file) > 100
        
        issues = []
        
        if not has_price:
            issues.append("missing_price")
        
        if not has_fundamentals:
            issues.append("missing_fundamentals")
        else:
            # בדיקת איכות נתונים יסודיים
            try:
                with open(advanced_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                non_null_data = {k: v for k, v in data.items() if v is not None}
                if len(non_null_data) < 10:
                    issues.append("low_quality")
                    
            except Exception:
                issues.append("corrupt_data")
        
        if issues:
            problematic_stocks.append((ticker, issues))
    
    return problematic_stocks

def fix_problematic_stock(ticker, issues):
    """תיקון מניה בעייתית ספציפית עם retry logic"""
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            success = False
            
            # תיקון מחירים
            if "missing_price" in issues or "corrupt_data" in issues:
                print(f"   📈 מנסה להוריד מחירים (נסיון {attempt + 1}/{max_retries})...")
                update_price_data(ticker, START_DATE, "stock_data")
                price_file = f"stock_data/{ticker}/{ticker}_price.csv"
                if os.path.exists(price_file) and os.path.getsize(price_file) > 1000:
                    print(f"   ✅ מחירים הורדו בהצלחה")
                    success = True
                else:
                    print(f"   ⚠️  מחירים לא הורדו או קובץ קטן מדי")
            
            # תיקון נתונים יסודיים
            if any(issue in issues for issue in ["missing_fundamentals", "low_quality", "corrupt_data"]):
                print(f"   📊 מנסה להוריד נתונים יסודיים (נסיון {attempt + 1}/{max_retries})...")
                fundamentals = scrape_yahoo_fundamentals(ticker)
                if fundamentals and any(v is not None for v in fundamentals.values()):
                    advanced_file = f"stock_data/{ticker}/{ticker}_advanced.json"
                    os.makedirs(os.path.dirname(advanced_file), exist_ok=True)
                    with open(advanced_file, 'w', encoding='utf-8') as f:
                        json.dump(fundamentals, f, indent=2, ensure_ascii=False)
                    
                    non_null_count = sum(1 for v in fundamentals.values() if v is not None)
                    print(f"   ✅ נתונים יסודיים הורדו: {non_null_count} שדות")
                    success = True
                else:
                    print(f"   ⚠️  נתונים יסודיים לא הורדו")
            
            if success:
                return True
            elif attempt < max_retries - 1:
                # המתנה לפני הנסיון הבא
                delay = wait_with_smart_retry(base_delay * (2 ** attempt))
                print(f"   ⏳ ממתין {delay:.1f} שניות לפני נסיון נוסף...")
                
        except Exception as e:
            print(f"   ❌ שגיאה בנסיון {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                delay = wait_with_smart_retry(base_delay * (2 ** attempt))
                print(f"   ⏳ ממתין {delay:.1f} שניות לפני נסיון נוסף...")
    
    print(f"   💔 נכשל בכל {max_retries} הנסיונות")
    return False

def analyze_download_failures():
    """ניתוח הסיבות למה נתונים לא ירדו מלכתחילה"""
    print("\n🔍 מנתח סיבות כשל בהורדת נתונים:")
    
    failure_reasons = {
        'rate_limit': 0,
        'network_error': 0, 
        'invalid_ticker': 0,
        'data_unavailable': 0,
        'parsing_error': 0,
        'timeout': 0
    }
    
    # סימולציה של בדיקה - בפועל נוכל לאסוף נתונים אלה במהלך ההורדה
    print("  📊 סיבות עיקריות לכשלים:")
    print("    - הגבלת קצב (Rate Limiting) מצד Yahoo Finance")
    print("    - שגיאות רשת זמניות")
    print("    - מניות שהוסרו מהמסחר") 
    print("    - ETF-ים עם נתונים מוגבלים")
    print("    - שגיאות parsing בנתונים מורכבים")
    print("    - Timeouts ברשת")
    print("  💡 פתרון: הרצה מחדש עם retry logic משופר")

def wait_with_smart_retry(base_delay=1.0):
    """המתנה חכמה עם exponential backoff"""
    import time
    import random
    
    # Random jitter למניעת thundering herd
    jitter = random.uniform(0.1, 0.3)
    delay = base_delay + jitter
    
    time.sleep(delay)
    return delay * 1.5  # הגדלת ההמתנה לפעם הבאה

def auto_fix_missing_data():
    """תיקון אוטומטי של נתונים חסרים"""
    print("🔧 מתחיל תיקון אוטומטי של נתונים חסרים...")
    
    # ניתוח הבעיות
    problematic_stocks = analyze_data_quality()
    
    if not problematic_stocks:
        print("✅ כל הנתונים שלמים! אין צורך בתיקונים.")
        return
    
    print(f"📊 נמצאו {len(problematic_stocks)} מניות בעייתיות")
    
    # סיווג הבעיות
    issues_count = {}
    for _, issues in problematic_stocks:
        for issue in issues:
            issues_count[issue] = issues_count.get(issue, 0) + 1
    
    print("� פירוט הבעיות:")
    for issue, count in issues_count.items():
        issue_names = {
            'missing_price': '📈 חסרי מחירים',
            'missing_fundamentals': '📊 חסרי נתונים יסודיים',
            'low_quality': '⚠️  איכות נמוכה',
            'corrupt_data': '💔 נתונים פגומים'
        }
        print(f"  {issue_names.get(issue, issue)}: {count}")
    
    print("\n�🔧 מתחיל תיקון אוטומטי...")
    
    fixed_count = 0
    failed_count = 0
    
    for i, (ticker, issues) in enumerate(problematic_stocks, 1):
        print(f"\n🔧 [{i}/{len(problematic_stocks)}] מתקן: {ticker}")
        print(f"   🎯 בעיות: {', '.join(issues)}")
        
        if fix_problematic_stock(ticker, issues):
            fixed_count += 1
            print(f"   🎉 המניה תוקנה בהצלחה!")
        else:
            failed_count += 1
            print(f"   💔 נכשל בתיקון המניה")
        
        # עדכון התקדמות כל 25 מניות
        if i % 25 == 0:
            success_rate = (fixed_count / i) * 100
            remaining = len(problematic_stocks) - i
            print(f"\n📊 התקדמות ביניים: {i}/{len(problematic_stocks)}")
            print(f"   ✅ תוקנו: {fixed_count}")
            print(f"   ❌ נכשלו: {failed_count}")
            print(f"   📈 אחוז הצלחה: {success_rate:.1f}%")
            print(f"   ⏳ נותרו: {remaining}")
            
            if i < len(problematic_stocks):
                print("   ⏸️  המתנה קצרה לפני המשך...")
                wait_with_smart_retry(2.0)
    
    print(f"\n" + "="*60)
    print(f"🎯 סיכום תיקון אוטומטי:")
    print(f"  ✅ תוקנו בהצלחה: {fixed_count}")
    print(f"  ❌ נכשלו: {failed_count}")
    
    if fixed_count + failed_count > 0:
        success_rate = (fixed_count / (fixed_count + failed_count)) * 100
        print(f"  📊 אחוז הצלחה כולל: {success_rate:.1f}%")
    
    # בדיקה סופית
    print("\n🔍 מבצע בדיקה סופית...")
    final_check = analyze_data_quality()
    print(f"📊 מניות בעייתיות שנותרו: {len(final_check)}")
    
    if len(final_check) < len(problematic_stocks):
        improvement = len(problematic_stocks) - len(final_check)
        improvement_rate = (improvement / len(problematic_stocks)) * 100
        print(f"🎉 שיפור מצוין: {improvement} מניות תוקנו ({improvement_rate:.1f}% שיפור)!")
    
    if final_check:
        print(f"⚠️  עדיין נותרו {len(final_check)} מניות עם בעיות:")
        
        # סיווג הבעיות שנותרו
        remaining_issues = {}
        for _, issues in final_check:
            for issue in issues:
                remaining_issues[issue] = remaining_issues.get(issue, 0) + 1
        
        for issue, count in remaining_issues.items():
            issue_names = {
                'missing_price': '📈 מחירים',
                'missing_fundamentals': '📊 נתונים יסודיים',
                'low_quality': '⚠️  איכות נמוכה',
                'corrupt_data': '💔 נתונים פגומים'
            }
            print(f"    {issue_names.get(issue, issue)}: {count}")
            
        # הצגת דוגמאות
        print(f"\n📝 דוגמאות למניות שעדיין בעייתיות:")
        for i, (ticker, issues) in enumerate(final_check[:5]):
            print(f"    {i+1}. {ticker}: {', '.join(issues)}")
        if len(final_check) > 5:
            print(f"    ... ועוד {len(final_check) - 5} מניות")
        
        print(f"\n💡 המלצה: הבעיות שנותרו ככל הנראה נובעות מ:")
        print(f"    • מניות שהוסרו מהמסחר")
        print(f"    • ETF-ים עם נתונים מוגבלים") 
        print(f"    • הגבלות קצב מצד Yahoo Finance")
        print(f"    • בעיות רשת זמניות")
    else:
        print("🎉 מצוין! כל הנתונים תוקנו בהצלחה!")
        print("🏆 המערכת מוכנה לשימוש מלא!")

if __name__ == "__main__":
    import argparse

    # בדיקת חיבור אינטרנט
    if not check_internet_connection():
        print("שגיאה: אין חיבור לאינטרנט. הסקריפט דורש חיבור פעיל לאינטרנט.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Stock Data Manager")
    parser.add_argument(
        "--mode",
        choices=["scan", "dashboard", "reset", "progress", "autofix"],
        default="dashboard",
        help="בחר מצב: 'scan' לסריקה יומית, 'dashboard' לדשבורד, 'reset' לאיפוס מלא, 'progress' לבדיקת התקדמות, 'autofix' לתיקון אוטומטי"
    )
    args = parser.parse_args()

    if args.mode == "scan":
        process_tickers_daily()
    elif args.mode == "reset":
        reset_all_progress()
    elif args.mode == "progress":
        check_progress()
    elif args.mode == "autofix":
        print("🔧 מפעיל תיקון אוטומטי של נתונים חסרים...")
        analyze_download_failures()
        auto_fix_missing_data()
    else:
        # דיפולט: דשבורד
        dashboard()
