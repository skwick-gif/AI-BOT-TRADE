import os
import sys
import subprocess
import json
import time
import logging
import argparse
from io import StringIO
from datetime import datetime, timedelta
import re
import csv
import pandas as pd
import requests
from bs4 import BeautifulSoup
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_fetcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
warnings.filterwarnings("ignore", message="Unknown datetime string format", category=UserWarning)

import yfinance as yf

# Configuration
class Config:
    DATA_FOLDER = "stock_data"
    START_DATE = "2020-01-01"
    MAX_TICKERS_PER_BATCH = 50
    TODO_FILE = 'todo_tickers.json'
    COMPLETED_FILE = 'completed_tickers.json'
    RETRY_COUNT_FILE = 'retry_counts.json'
    MAX_RETRIES = 3
    REQUEST_DELAY = 2  # seconds between web requests
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# קבועים נוספים
MAX_RETRIES = 3
RETRY_DELAY = 1.0

# -------- קלאס ניהול מצב --------
class StateManager:
    """מנהל את מצב העיבוד של הטיקרים"""
    
    def __init__(self):
        self.todo_file = "todo_tickers.json"
        self.completed_file = "completed_tickers.json"
        self.retry_file = "retry_counts.json"
    
    def initialize_state(self):
        """אתחול מצב או טעינת מצב קיים"""
        # אם אין קבצי מצב, צור אותם
        if not os.path.exists(self.todo_file):
            all_tickers = get_all_tickers()
            self.save_todo_tickers(all_tickers)
            logger.info(f"Initialized TODO list with {len(all_tickers)} tickers")
        
        if not os.path.exists(self.completed_file):
            self.save_completed_tickers([])
            logger.info("Initialized empty completed list")
        
        if not os.path.exists(self.retry_file):
            self.save_retry_counts({})
            logger.info("Initialized empty retry counts")
    
    def get_todo_tickers(self):
        """טען רשימת טיקרים לעיבוד"""
        try:
            with open(self.todo_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading TODO list: {e}")
            return []
    
    def get_completed_tickers(self):
        """טען רשימת טיקרים שהושלמו"""
        try:
            with open(self.completed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading completed list: {e}")
            return []
    
    def get_retry_counts(self):
        """טען מספרי ניסיונות חוזרים"""
        try:
            with open(self.retry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading retry counts: {e}")
            return {}
    
    def save_todo_tickers(self, todo_list):
        """שמור רשימת TODO"""
        try:
            with open(self.todo_file, 'w', encoding='utf-8') as f:
                json.dump(todo_list, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving TODO list: {e}")
    
    def save_completed_tickers(self, completed_list):
        """שמור רשימת completed"""
        try:
            with open(self.completed_file, 'w', encoding='utf-8') as f:
                json.dump(completed_list, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving completed list: {e}")
    
    def save_retry_counts(self, retry_counts):
        """שמור מספרי ניסיונות"""
        try:
            with open(self.retry_file, 'w', encoding='utf-8') as f:
                json.dump(retry_counts, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving retry counts: {e}")
    
    def mark_ticker_completed(self, ticker):
        """העבר טיקר מ-TODO ל-completed"""
        todo = self.get_todo_tickers()
        completed = self.get_completed_tickers()
        retry_counts = self.get_retry_counts()
        
        if ticker in todo:
            todo.remove(ticker)
        
        if ticker not in completed:
            completed.append(ticker)
        
        # נקה מספר ניסיונות
        if ticker in retry_counts:
            del retry_counts[ticker]
        
        self.save_todo_tickers(todo)
        self.save_completed_tickers(completed)
        self.save_retry_counts(retry_counts)
    
    def increment_retry_count(self, ticker):
        """העלה מספר ניסיונות לטיקר"""
        retry_counts = self.get_retry_counts()
        retry_counts[ticker] = retry_counts.get(ticker, 0) + 1
        self.save_retry_counts(retry_counts)
        return retry_counts[ticker]
    
    def save_state(self):
        """שמור מצב נוכחי (נקרא מעת לעת)"""
        logger.debug("State saved successfully")

def setup_logging():
    """הגדרת מערכת הלוגים"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stock_processing.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# אתחול משתנים גלובליים
DATA_FOLDER = Config.DATA_FOLDER
os.makedirs(DATA_FOLDER, exist_ok=True)
START_DATE = Config.START_DATE
MAX_TICKERS_PER_BATCH = Config.MAX_TICKERS_PER_BATCH
TODO_FILE = Config.TODO_FILE
COMPLETED_FILE = Config.COMPLETED_FILE
HEADERS = Config.HEADERS

session = requests.Session()
session.headers.update(HEADERS)
session.verify = True

def load_retry_counts():
    """Load retry counts from file"""
    try:
        with open(Config.RETRY_COUNT_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_retry_counts(retry_counts):
    """Save retry counts to file"""
    with open(Config.RETRY_COUNT_FILE, 'w') as f:
        json.dump(retry_counts, f, indent=2)

def should_retry(ticker, retry_counts):
    """Check if we should retry a ticker based on failure count"""
    return retry_counts.get(ticker, 0) < Config.MAX_RETRIES

def increment_retry_count(ticker, retry_counts):
    """Increment retry count for a ticker"""
    retry_counts[ticker] = retry_counts.get(ticker, 0) + 1
    save_retry_counts(retry_counts)

def reset_retry_count(ticker, retry_counts):
    """Reset retry count for successful ticker"""
    if ticker in retry_counts:
        del retry_counts[ticker]
        save_retry_counts(retry_counts)

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
        'ZXIET', 'ZZAZT', 'ZZINT', 'ZZEXT', 'ZZTEST', 'ZZDIV', 'XTSLA'
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
        if t.endswith('W') and t not in ['SDOW', 'UDOW']:
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
        logger.info(f"Filtered out {filtered_count} problematic tickers")
    
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
                logger.error(f"Error parsing NASDAQ csv: {e}")
                continue
                
            if "Symbol" in df.columns:
                tickers = clean_tickers(df["Symbol"].astype(str).tolist())
            else:
                logger.error(f"Symbol column not found in NASDAQ data columns: {df.columns.tolist()}")
                continue
                
            if len(tickers) > 100:
                logger.info(f"NASDAQ tickers ({len(tickers)}) downloaded from {url}")
                return tickers
            logger.warning("NASDAQ resolver returned too few tickers")
            
        except Exception as e:
            last_err = e
            logger.error(f"Failed to download NASDAQ from {url}: {e}")
            continue
    
    logger.error(f"Failed all NASDAQ mirrors. Last error: {last_err}")
    return []

def get_nyse_tickers():
    """אוסף מניות מבורסת NYSE"""
    mirrors = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt",
    ]
    last_err = None
    for url in mirrors:
        try:
            txt = fetch_text(url)
            try:
                df = pd.read_csv(StringIO(txt), sep="|")
            except Exception as e:
                logger.error(f"Error parsing NYSE csv: {e}")
                continue
                
            if "NASDAQ Symbol" in df.columns:
                tickers = clean_tickers(df["NASDAQ Symbol"].astype(str).tolist())
            elif "Symbol" in df.columns:
                tickers = clean_tickers(df["Symbol"].astype(str).tolist())
            else:
                logger.error(f"Expected symbol column not found in NYSE data columns: {df.columns.tolist()}")
                continue
                
            if len(tickers) > 100:
                logger.info(f"NYSE tickers ({len(tickers)}) downloaded from {url}")
                return tickers
            logger.warning("NYSE resolver returned too few tickers")
            
        except Exception as e:
            last_err = e
            logger.error(f"Failed to download NYSE from {url}: {e}")
            continue
    
    logger.error(f"Failed all NYSE mirrors. Last error: {last_err}")
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
    """אוסף מניות ממדד ראסל 2000"""
    URL_RUSSELL2000 = ("https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
                       "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund")
    try:
        logger.info("Downloading Russell 2000 tickers via official iShares CSV...")
        csv_text = fetch_text(URL_RUSSELL2000)
        
        # שמור עותק גלם לבדיקה
        with open("IWM_raw.csv", "w", encoding="utf-8") as f:
            f.write(csv_text)
            
        header_idx = find_header_line(csv_text)
        if header_idx < 0:
            logger.error("Header line with 'Ticker' not found in IWM CSV")
            return []
            
        trimmed = "\n".join(csv_text.splitlines()[header_idx:])
        delim = sniff_delimiter("\n".join(csv_text.splitlines()[header_idx:header_idx+5]))
        
        df = pd.read_csv(StringIO(trimmed), sep=delim, engine="python")
        
        # חפש עמודת ticker
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
            logger.error(f"Could not find 'Ticker' column in IWM CSV. Columns: {list(df.columns)}")
            return []
            
        tickers = clean_tickers(df[ticker_col].astype(str).tolist())
        logger.info(f"Russell 2000 tickers downloaded: {len(tickers)}")
        return tickers
        
    except Exception as e:
        logger.error(f"Error downloading Russell 2000 tickers: {e}")
        return []

def find_header_line(csv_text):
    """מוצא את שורת הכותרת בקובץ CSV"""
    for i, line in enumerate(csv_text.splitlines()):
        if "Ticker" in line.split(",") or "Ticker" in line:
            return i
    return -1

def sniff_delimiter(sample):
    """מגלה את התו המפריד בקובץ CSV"""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        return dialect.delimiter
    except Exception:
        return ","

def get_all_tickers():
    # If there is an existing completed/todo state file, prefer returning that raw list
    completed_file = Config.COMPLETED_FILE
    todo_file = Config.TODO_FILE
    if os.path.exists(completed_file):
        try:
            with open(completed_file, 'r', encoding='utf-8') as f:
                completed_tickers = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read {completed_file}: {e}")
            completed_tickers = []
        todo_tickers = []
        if os.path.exists(todo_file):
            try:
                with open(todo_file, 'r', encoding='utf-8') as f:
                    todo_tickers = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read {todo_file}: {e}")
        combined_from_files = completed_tickers + todo_tickers
        logger.info(f"Using raw ticker list from state files: {len(combined_from_files)} tickers (completed: {len(completed_tickers)}, todo: {len(todo_tickers)})")
        return sorted(set(combined_from_files))

    sp500 = get_sp500_tickers()
    nasdaq100 = get_nasdaq100_tickers()
    dow30 = get_dowjones_tickers()
    russell2000 = get_russell2000_tickers()
    nyse = get_nyse_tickers()
    nasdaq = get_nasdaq_tickers()  # הוספת מניות NASDAQ
    combined = set(sp500 + nasdaq100 + dow30 + russell2000 + nyse + nasdaq)
    
    # Add additional ETFs and leveraged products
    additional_etfs = [
        'DIA', 'QQQ', 'IWM', 'SPY',  # Main ETFs
        'SDOW', 'SPXL', 'SPXU', 'SQQQ', 'TNA', 'TQQQ', 'TZA', 'UDOW', 'UPRO'  # Leveraged ETFs
    ]
    combined.update(additional_etfs)
    
    print(f"Total tickers before filter: {len(combined)}")
    print(f"  S&P 500: {len(sp500)}, NASDAQ-100: {len(nasdaq100)}, Dow: {len(dow30)}")
    print(f"  Russell 2000: {len(russell2000)}, NYSE: {len(nyse)}, NASDAQ: {len(nasdaq)}")
    print(f"  Additional ETFs: {len(additional_etfs)}")
    # סנן מניות בעייתיות
    filtered = clean_tickers(list(combined))
    removed = len(combined) - len(filtered)
    print(f"Tickers after filter: {len(filtered)} (Removed {removed} - filtered out problematic tickers)")
    
    # מחק מידע של מניות מסוננות
    removed_tickers = combined - set(filtered)
    for ticker in removed_tickers:
        folder = os.path.join(DATA_FOLDER, ticker)
        if os.path.exists(folder):
            import shutil
            shutil.rmtree(folder)
            print(f"Deleted data for {ticker}")
    
    return filtered

all_tickers = get_all_tickers()

def update_price_data(ticker, start_date, folder):
    """עדכון נתוני מחירים לטיקר מסוים עם טיפול משופר בשגיאות"""
    file_path = os.path.join(folder, ticker, f"{ticker}_price.csv")
    os.makedirs(os.path.join(folder, ticker), exist_ok=True)
    
    # בדוק אם קיים קובץ
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            existing_df = existing_df.dropna(how='all')
            
            if not existing_df.empty:
                # בדוק כותרות שגויות
                if existing_df.columns.tolist() == ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
                    logger.warning(f"Malformed header detected for {ticker}, fixing...")
                    fixed_df = existing_df.reset_index()
                    if len(fixed_df.columns) >= 6:
                        fixed_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        fixed_df['Date'] = pd.to_datetime(fixed_df['Date'])
                        fixed_df.set_index('Date', inplace=True)
                        existing_df = fixed_df
                        
                if existing_df.isnull().any().any():
                    logger.warning(f"Missing data found in existing file for {ticker}, re-downloading full range")
                    existing_df = None
                    last_date = None
                else:
                    last_date = existing_df.index.max()
            else:
                last_date = None
                
        except Exception as e:
            logger.warning(f"Error reading existing file for {ticker}: {e} - re-downloading")
            existing_df = None
            last_date = None
    else:
        existing_df = None
        last_date = None
    
    # קבע תאריך התחלה להורדה
    start_download = start_date
    if last_date:
        start_date_dt = last_date + timedelta(days=1)
        if start_date_dt <= datetime.today():
            start_download = start_date_dt.strftime("%Y-%m-%d")
        else:
            logger.info(f"No new price data needed for {ticker}")
            return
    
    try:
        new_df = yf.download(ticker, start=start_download, progress=False, auto_adjust=True)
        
        if new_df.empty:
            logger.warning(f"No new data available for {ticker}")
            return
            
        new_df.index = pd.to_datetime(new_df.index)
        new_df.index.name = "Date"
        
        # נרמל שמות עמודות
        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = new_df.columns.get_level_values(0)
        
        # ודא שיש לנו את העמודות הנדרשות
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in new_df.columns for col in expected_cols):
            logger.warning(f"Missing expected columns for {ticker}, skipping update")
            return
        
        new_df = new_df[expected_cols]
        
        # מזג עם נתונים קיימים
        if existing_df is not None:
            if isinstance(existing_df.columns, pd.MultiIndex):
                existing_df.columns = existing_df.columns.get_level_values(0)
                
            if not all(col in existing_df.columns for col in expected_cols):
                logger.warning(f"Existing data missing expected columns for {ticker}, re-downloading full range")
                df = new_df
            else:
                existing_df = existing_df[expected_cols]
                df = pd.concat([existing_df, new_df])
                df = df[~df.index.duplicated(keep='last')]
        else:
            df = new_df
        
        # שמור עם כותרות נכונות
        df.to_csv(file_path, index=True)
        logger.info(f"Successfully saved price data for {ticker} ({len(df)} records)")
        
    except Exception as e:
        logger.error(f"Error downloading price data for {ticker}: {e}")
        raise

def get_html(url, max_retries=3, delay=0.8):
    """מורד HTML מURL עם טיפול משופר בשגיאות וחזרות"""
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
            logger.debug(f"Successfully fetched HTML from {url}")
            return response.text
            
        except requests.HTTPError as e:
            if e.response.status_code in [401, 403, 429]:
                logger.warning(f"Access denied for {url} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
                    continue
            else:
                logger.error(f"HTTP error fetching {url}: {e}")
            return ""
            
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.warning(f"Connection error for {url} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
                continue
            return ""
            
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return ""
    
    logger.error(f"Failed to fetch HTML from {url} after {max_retries} attempts")
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
    """עיבוד יומי של טיקרים עם מנגנון ניהול מצב משופר"""
    logger.info("Starting daily ticker processing...")
    
    # אתחל או טען מצב
    state_manager = StateManager()
    state_manager.initialize_state()
    
    all_tickers = get_all_tickers()
    logger.info(f"Found {len(all_tickers)} total tickers")
    
    todo = state_manager.get_todo_tickers()
    completed = state_manager.get_completed_tickers()
    retry_counts = state_manager.get_retry_counts()
    
    logger.info(f"TODO: {len(todo)}, Completed: {len(completed)}")
    
    # אמת שטיקרים חדשים יתווספו לרשימת TODO
    new_tickers = set(all_tickers) - set(todo) - set(completed)
    if new_tickers:
        todo.extend(list(new_tickers))
        logger.info(f"Added {len(new_tickers)} new tickers to TODO list")
    
    # בחר טיקרים לעיבוד
    tickers_to_process = todo[:limit] if limit else todo.copy()
    
    if not tickers_to_process:
        logger.info("No tickers to process. All tickers completed.")
        return
    
    logger.info(f"Processing {len(tickers_to_process)} tickers...")
    
    successful_count = 0
    error_count = 0
    
    for i, ticker in enumerate(tickers_to_process, 1):
        logger.info(f"Processing {ticker} ({i}/{len(tickers_to_process)})")
        
        try:
            # בדוק אם נדרש עדכון מחירים
            price_file = os.path.join(DATA_FOLDER, ticker, f"{ticker}_price.csv")
            needs_price_update = True
            
            if os.path.exists(price_file):
                try:
                    df = pd.read_csv(price_file, parse_dates=['Date'])
                    if not df.empty:
                        last_date = df['Date'].max()
                        days_since = (datetime.now() - last_date.tz_localize(None) if last_date.tz else datetime.now() - last_date).days
                        has_data = not df[['Open', 'High', 'Low', 'Close']].isnull().all().all()
                        
                        if days_since <= 2 and has_data:
                            needs_price_update = False
                except Exception as e:
                    logger.warning(f"Error checking existing price data for {ticker}: {e}")
            
            # עדכן מחירים אם נדרש
            if needs_price_update:
                update_price_data(ticker, START_DATE, DATA_FOLDER)
                successful_count += 1
                logger.info(f"Successfully updated price data for {ticker}")
            else:
                logger.info(f"Price data for {ticker} is up to date")
            
            # בדוק אם נדרש עדכון נתוני advanced
            advanced_file = os.path.join(DATA_FOLDER, ticker, f"{ticker}_advanced.json")
            needs_advanced_update = True
            
            if os.path.exists(advanced_file):
                try:
                    with open(advanced_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # אם יש לפחות 10 שדות נתונים, נחשב שהמידע מלא
                    if len(data) >= 10:
                        needs_advanced_update = False
                        logger.info(f"Advanced data for {ticker} is complete ({len(data)} fields)")
                except Exception as e:
                    logger.warning(f"Error reading advanced file for {ticker}: {e}")
            
            # עדכן נתוני advanced אם נדרש
            if needs_advanced_update:
                logger.info(f"Updating advanced data for {ticker}")
                try:
                    if scrape_all_data(ticker, DATA_FOLDER):
                        logger.info(f"Successfully scraped advanced data for {ticker}")
                    else:
                        logger.warning(f"Failed to scrape advanced data for {ticker}")
                except Exception as e:
                    logger.error(f"Error scraping advanced data for {ticker}: {e}")
            
            # העבר ל-completed רק אחרי שעדכנו הכל
            state_manager.mark_ticker_completed(ticker)
            logger.info(f"Successfully processed {ticker} (price + advanced data)")
                
        except Exception as e:
            error_count += 1
            retry_count = state_manager.increment_retry_count(ticker)
            
            if retry_count <= MAX_RETRIES:
                logger.warning(f"Error processing {ticker} (attempt {retry_count}/{MAX_RETRIES}): {e}")
            else:
                logger.error(f"Max retries exceeded for {ticker}, removing from TODO: {e}")
                state_manager.mark_ticker_completed(ticker)
        
        # שמור מצב כל 10 טיקרים
        if i % 10 == 0:
            state_manager.save_state()
    
    # שמור מצב סופי
    state_manager.save_state()
    
    logger.info(f"Daily processing completed. Successfully processed: {successful_count}, Errors: {error_count}")
    
    # הדפס סטטיסטיקות
    remaining_todo = len(state_manager.get_todo_tickers())
    total_completed = len(state_manager.get_completed_tickers())
    
    logger.info(f"Status: {total_completed} completed, {remaining_todo} remaining")
    
    if remaining_todo == 0:
        logger.info("🎉 All tickers have been processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily stock data updater")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers to process")
    args = parser.parse_args()
    
    setup_logging()
    process_tickers_daily(limit=args.limit)
