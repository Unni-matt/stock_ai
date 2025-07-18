# ✅ STEP 0: Install dependencies
!pip install yfinance ta tqdm requests chardet --quiet

# ✅ STEP 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ✅ STEP 2: Imports
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import requests
from io import StringIO
from tqdm import tqdm
import os, csv, re, time
from urllib.parse import quote
import chardet

pd.set_option("display.max_columns", None)

# ✅ STEP 3: Index fetcher config
index_folder = '/content/drive/My Drive/mathew_index'
os.makedirs(index_folder, exist_ok=True)

all_indices = [
    "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
    "NIFTY BANK", "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY SMALLCAP 100",
    "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250", "NIFTY MICROCAP 250", "NIFTY TOTAL MARKET",
    "NIFTY LARGEMIDCAP 250", "NIFTY500 LARGEMIDSMALL EQUAL-CAP WEIGHTED",
    "NIFTY MIDSMALLCAP 400", "NIFTY MIDSMALL INDIA CONSUMPTION",
    "NIFTY SMALLCAP 50", "NIFTY MIDSMLCAP", "NIFTY IT", "NIFTY FMCG", "NIFTY PHARMA",
    "NIFTY AUTO", "NIFTY ENERGY", "NIFTY MEDIA", "NIFTY REALTY", "NIFTY METAL",
    "NIFTY PSU BANK", "NIFTY PRIVATE BANK", "NIFTY FINANCIAL SERVICES",
    "NIFTY CONSUMER DURABLES", "NIFTY OIL & GAS", "NIFTY HEALTHCARE INDEX",
    "NIFTY CAPITAL MARKETS", "NIFTY TRANSPORTATION & LOGISTICS",
    "NIFTY CPSE", "NIFTY PSE", "NIFTY SERVICES SECTOR",
    "NIFTY INDIA CONSUMPTION", "NIFTY INFRASTRUCTURE", "NIFTY MNC",
    "NIFTY GROWTH SECTORS 15", "NIFTY INDIA MANUFACTURING",
    "NIFTY INDIA DEFENCE", "NIFTY INDIA TOURISM", "NIFTY INDIA DIGITAL",
    "NIFTY EV & NEW AGE AUTOMOTIVE", "NIFTY MOBILITY", "NIFTY HOUSING",
    "NIFTY CORE HOUSING", "NIFTY IPO", "NIFTY SHARIAH 25", "NIFTY50 SHARIAH",
    "NIFTY500 SHARIAH", "NIFTY100 ESG", "NIFTY100 ENHANCED ESG",
    "NIFTY100 ALPHA 30", "NIFTY200 VALUE 30", "NIFTY200 ALPHA 30",
    "NIFTY200 MOMENTUM 30", "NIFTY200 QUALITY 30", "NIFTY100 QUALITY 30",
    "NIFTY100 LOW VOLATILITY 30", "NIFTY100 EQUAL WEIGHT", "NIFTY50 VALUE 20",
    "NIFTY50 EQUAL WEIGHT", "NIFTY50 LOW VOLATILITY 50",
    "NIFTY500 VALUE 50", "NIFTY500 QUALITY 50", "NIFTY500 LOW VOLATILITY 50",
    "NIFTY500 EQUAL WEIGHT", "NIFTY500 MOMENTUM 50",
    "NIFTY500 MULTICAP MOMENTUM QUALITY 50", "NIFTY500 MULTICAP INFRASTRUCTURE 50:30:20",
    "NIFTY500 MULTICAP INDIA MANUFACTURING 50:30:20",
    "NIFTY500 MULTICAP 50:25:25", "NIFTY TOP 10 EQUAL WEIGHT",
    "NIFTY TOP 15 EQUAL WEIGHT", "NIFTY TOP 20 EQUAL WEIGHT",
    "NIFTY ALPHA 50", "NIFTY ALPHA LOW-VOLATILITY 30",
    "NIFTY ALPHA QUALITY LOW-VOLATILITY 30",
    "NIFTY ALPHA QUALITY VALUE LOW-VOLATILITY 30",
    "NIFTY MIDCAP150 QUALITY 50", "NIFTY MIDCAP150 MOMENTUM 50",
    "NIFTY SMALLCAP250 QUALITY 50", "NIFTY SMALLCAP250 MOMENTUM QUALITY 100",
    "NIFTY MIDSMALLCAP400 MOMENTUM QUALITY 100",
    "NIFTY QUALITY LOW-VOLATILITY 30", "NIFTY DIVIDEND OPPORTUNITIES 50",
    "NIFTY FINANCIAL SERVICES 25/50", "NIFTY MIDCAP SELECT",
    "NIFTY LIQUID 15", "NIFTY MIDCAP LIQUID 15", "NIFTY SHARIAH 25",
    "NIFTY FINANCIAL SERVICES EX-BANK", "NIFTY MIDSMALL HEALTHCARE",
    "NIFTY MIDSMALL IT & TELECOM", "NIFTY MIDSMALL FINANCIAL SERVICES",
    "NIFTY RURAL", "NIFTY NON-CYCLICAL CONSUMER"
]

def get_output_filename(index_name):
    formatted_date = datetime.now().strftime("%d-%b-%Y")
    clean_name = index_name.lower().replace("&", "and")
    clean_name = re.sub(r"[^a-z0-9]+", "-", clean_name).strip("-")
    return f"MW-{clean_name.upper()}-{formatted_date}.csv"

def fetch_live_constituents(index_name):
    headers = {
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'connection': 'keep-alive',
        # --- PASTE YOUR NSE COOKIE BELOW ---
        'cookie': 'PUT_YOUR_NSE_COOKIE_HERE',
        'host': 'www.nseindia.com',
        'referer': 'https://www.nseindia.com/market-data/live-equity-market',
        'sec-ch-ua': '"Not)A;Brand";v="8", "Chromium";v="138", "Microsoft Edge";v="138"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0',
        'x-requested-with': 'XMLHttpRequest',
    }
    api_url = f"https://www.nseindia.com/api/equity-stockIndices?csv=true&index={quote(index_name)}&selectValFormat=crores"
    print(f"📥 Fetching {index_name} ...")
    try:
        resp = requests.get(api_url, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), skiprows=10)
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        if "SYMBOL" not in df.columns or df.shape[0] < 3:
            print(f"❌ Invalid data received for {index_name}. The response was not a valid CSV.")
            return None
        return df
    except Exception as e:
        print(f"❌ Error fetching {index_name}: {e}")
        return None

for index in all_indices:
    df = fetch_live_constituents(index)
    if df is not None and not df.empty:
        filename = get_output_filename(index)
        filepath = os.path.join(index_folder, filename)
        df.to_csv(filepath, index=False)
        print(f"✅ Saved: {filepath}")
    else:
        print(f"❌ Failed to process: {index}")
    time.sleep(2)

def get_nse_symbols(max_stocks=100):
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://nsearchives.nseindia.com"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = df.columns.str.strip()
        syms = df["SYMBOL"].dropna().astype(str).str.strip().unique()
        return [s + ".NS" for s in syms][:max_stocks]
    except Exception as e:
        print(f"❌ Failed to fetch NSE symbols: {e}")
        return []

def fetch_bse_corporate_actions(from_date="20150101"):
    to_date = datetime.now().strftime("%Y%m%d")
    base_url = "https://api.bseindia.com/BseIndiaAPI/api/CorpactCSVDownload/w"
    params = {"scripcode": "", "Fdate": from_date, "TDate": to_date, "Purposecode": "", "strSearch": "S",
              "ddlindustrys": "", "ddlcategorys": "E", "segment": "0"}
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    full_url = f"{base_url}?{query_string}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv",
               "Referer": "https://www.bseindia.com/corporates/corporate_act.aspx"}
    try:
        response = requests.get(full_url, headers=headers, timeout=30)
        response.raise_for_status()
        csv_data = response.text.lstrip('\ufeff')
        df = pd.read_csv(StringIO(csv_data))
        df.columns = df.columns.str.strip()
        df["Ex Date"] = pd.to_datetime(df["Ex Date"], errors="coerce")
        df["Security Code"] = df["Security Code"].astype(str)
        df["CorporateAction"] = df["Purpose"].str.strip().str.slice(0, 40)
        df.dropna(subset=['Ex Date'], inplace=True)
        return df
    except Exception as e:
        print(f"❌ Failed to fetch BSE corporate actions: {e}")
        return pd.DataFrame()

def compute_supertrend(df, atr, period=10, multiplier=3):
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()
    close = df['Close'].to_numpy()
    atr_np = atr.to_numpy()
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr_np)
    lowerband = hl2 - (multiplier * atr_np)
    direction = np.ones(len(df), dtype=int)
    supertrend = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        if close[i] > upperband[i-1]: direction[i] = 1
        elif close[i] < lowerband[i-1]: direction[i] = -1
        else: direction[i] = direction[i-1]
        if direction[i] == 1: lowerband[i] = max(lowerband[i], lowerband[i-1])
        else: upperband[i] = min(upperband[i], upperband[i-1])
        supertrend[i] = lowerband[i] if direction[i] == 1 else upperband[i]
    df["Supertrend"] = supertrend
    df["Supertrend_Direction"] = direction
    return df

def add_advanced_candle_patterns(df):
    body = abs(df['Close'] - df['Open'])
    candle_range = df['High'] - df['Low']
    upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']
    with np.errstate(divide='ignore', invalid='ignore'):
        df["Doji"] = ((body / candle_range) < 0.1).astype(int)
    df["Hammer"] = ((lower_wick > 2 * body) & (upper_wick < body)).astype(int)
    df["EngulfingBull"] = ((df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) &
                           (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))).astype(int)
    df["EngulfingBear"] = ((df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) &
                           (df['Close'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1))).astype(int)
    return df

def compute_streaks(df):
    change = np.sign(df["Close"].diff())
    up_mask = (change == 1)
    down_mask = (change == -1)
    df['UpStreak'] = up_mask.cumsum() - up_mask.cumsum().where(~up_mask).ffill().fillna(0)
    df['DownStreak'] = down_mask.cumsum() - down_mask.cumsum().where(~down_mask).ffill().fillna(0)
    return df

# ==== BUILD DATASET ====
START_DATE = (datetime.now() - timedelta(days=365*10)).strftime("%Y-%m-%d")
END_DATE = datetime.now().strftime("%Y-%m-%d")
TARGET_DAYS = 5
TARGET_PCT = 0.02
VOLUME_SPIKE_THRESHOLD = 2.0

symbols = get_nse_symbols(100)
bse_df = fetch_bse_corporate_actions()

raw = yf.download(
    tickers=symbols,
    start=START_DATE,
    end=END_DATE,
    progress=True,
    threads=True,
    group_by="ticker",
    auto_adjust=True
)

all_dfs = []

for sym in tqdm(symbols, desc="Processing symbols"):
    try:
        if sym not in raw or raw[sym].dropna().empty: continue
        df = raw[sym].copy().dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        if df.shape[0] < 250: continue
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        close = df["Close"]

        # --- INDICATORS ---
        df["RSI"] = RSIIndicator(close).rsi()
        df["MACD"] = MACD(close).macd_diff()
        df["EMA20"] = EMAIndicator(close, 20).ema_indicator()
        df["EMA50"] = EMAIndicator(close, 50).ema_indicator()
        df["SMA50"] = SMAIndicator(close, 50).sma_indicator()
        df["SMA200"] = SMAIndicator(close, 200).sma_indicator()
        df["ADX"] = ADXIndicator(df["High"], df["Low"], close).adx()
        bb = BollingerBands(close, 20)
        df["BB_Width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / close
        df["ATR"] = AverageTrueRange(df["High"], df["Low"], close).average_true_range()

        # --- CUSTOM FEATURES ---
        df = compute_supertrend(df, atr=df["ATR"])
        df = add_advanced_candle_patterns(df)
        df = compute_streaks(df)

        # --- EVENT-BASED FEATURES ---
        df["MACD_BullishCross"] = ((df["MACD"] > 0) & (df["MACD"].shift(1) <= 0)).astype(int)
        df["GoldenCross"] = ((df["SMA50"] > df["SMA200"]) & (df["SMA50"].shift(1) <= df["SMA200"].shift(1))).astype(int)
        df["VolatilityCompression"] = (df["BB_Width"] < df["BB_Width"].rolling(20).quantile(0.1)).astype(int)
        df["VolumeSpike"] = (df["Volume"] > df["Volume"].rolling(20).mean() * VOLUME_SPIKE_THRESHOLD).astype(int)

        # --- TIME-BASED FEATURES ---
        df["DayOfWeek"] = df.index.dayofweek
        df["Month"] = df.index.month

        # --- TARGET VARIABLE ---
        df["FutureClose"] = close.shift(-TARGET_DAYS)
        df["Target"] = ((df["FutureClose"] - close) / close > TARGET_PCT).astype(int)
        df["Symbol"] = sym

        # --- CORPORATE ACTIONS ---
        nse_base_symbol = sym.replace(".NS", "")
        bse_relevant = bse_df[bse_df['Security Name'].str.contains(nse_base_symbol, case=False, na=False)].copy()
        if not bse_relevant.empty:
            bse_agg = bse_relevant.groupby('Ex Date')['CorporateAction'].apply(lambda x: '; '.join(x.astype(str).unique())).reset_index()
            action_dates_df = bse_agg.set_index('Ex Date').sort_index()
            action_dates_df['action_date'] = action_dates_df.index
            merged_last = pd.merge_asof(df, action_dates_df, left_index=True, right_index=True, direction='backward')
            df['DaysSinceAction'] = (df.index - merged_last['action_date']).dt.days
            df['CorporateAction'] = merged_last['CorporateAction']
            merged_next = pd.merge_asof(df, action_dates_df[['action_date']], left_index=True, right_index=True, direction='forward')
            df['DaysUntilAction'] = (merged_next['action_date'] - df.index).dt.days
            #for col in ['CorporateAction', 'DaysSinceAction', 'DaysUntilAction']:
            for col in ['CorporateAction', 'DaysSinceAction', 'DaysUntilAction']:
                if col not in df.columns:
                    df[col] = np.nan if 'Days' in col else '-'
        df['CorporateAction'] = df['CorporateAction'].fillna('-')
        df["HasCorporateAction"] = (df["CorporateAction"] != "-").astype(int)
        df["IsDividend"] = df["CorporateAction"].str.contains("dividend", case=False, na=False).astype(int)
        df["IsBonus"] = df["CorporateAction"].str.contains("bonus", case=False, na=False).astype(int)
        df["IsSplit"] = df["CorporateAction"].str.contains("split", case=False, na=False).astype(int)
        df["DividendAmount"] = pd.to_numeric(df["CorporateAction"].str.extract(r'Rs.?\s?([\d.]+)', expand=False), errors='coerce')
        all_dfs.append(df)
    except Exception as e:
        print(f"❌ Error processing {sym}: {e}")

if all_dfs:
    dataset = pd.concat(all_dfs)
    dataset.dropna(subset=['SMA200', 'FutureClose'], inplace=True)
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = dataset.select_dtypes(include=np.number).columns
    dataset[numeric_cols] = dataset[numeric_cols].fillna(0)
    object_cols = dataset.select_dtypes(include='object').columns
    dataset[object_cols] = dataset[object_cols].fillna('-')

    # ----------- INDEX TAGGING (Optimized for Performance) ------------
    print("\n🔖 Adding index membership columns...")
    index_files = [f for f in os.listdir(index_folder) if f.endswith('.csv')]
    new_index_columns = []
    for file in index_files:
        index_path = os.path.join(index_folder, file)
        try:
            with open(index_path, 'rb') as f:
                rawdata = f.read(4096)
                encoding = chardet.detect(rawdata)['encoding'] or 'utf-8'
            df_index = pd.read_csv(index_path, encoding=encoding)
            symbol_col_candidates = [c for c in df_index.columns if "symbol" in c.lower()]
            if not symbol_col_candidates: continue
            symbols_in_index = df_index[symbol_col_candidates[0]].astype(str).str.replace('.NS','',regex=False).str.strip().str.upper().unique()
            match = re.search(r'MW-(.*?)-\d{2}-\w{3}-\d{4}\.csv', file, re.IGNORECASE)
            if match:
                tag = match.group(1).upper().replace('-', '_').replace("&", "AND")
                new_col = dataset['Symbol'].str.replace('.NS', '', regex=False).str.upper().isin(symbols_in_index).astype(int)
                new_col.name = tag
                new_index_columns.append(new_col)
                print(f"✅ {file} -> {tag} ({len(symbols_in_index)} symbols)")
        except Exception as e:
            print(f"❌ Could not tag {file}: {e}")
    if new_index_columns:
        dataset = pd.concat([dataset] + new_index_columns, axis=1)

    dataset = dataset.copy()

    # ----------- SECTOR TAGGING -----------
    print("\n🏷️ Adding PrimarySector column using index columns...")

    pattern_sector = [
        ("IT", "IT"),
        ("FMCG", "FMCG"),
        ("PHARMA", "Pharma"),
        ("BANK", "Banking"),
        ("AUTO", "Auto"),
        ("ENERGY", "Energy"),
        ("MEDIA", "Media"),
        ("REALTY", "Realty"),
        ("METAL", "Metal"),
        ("HEALTH", "Healthcare"),
        ("FINANCIAL", "Finance"),
        ("CAPITAL MARKETS", "Finance"),
        ("CONSUMER DURABLES", "Consumer"),
        ("CONSUMPTION", "Consumer"),
        ("SERVICES", "Services"),
        ("CPSE", "PSU"),
        ("PSE", "PSU"),
        ("INFRA", "Infrastructure"),
        ("MNC", "MNC"),
        ("DIGITAL", "Tech"),
        ("DEFENCE", "Defence"),
        ("RURAL", "Rural"),
        ("MANUFACTURING", "Manufacturing"),
        ("HOUSING", "Real Estate"),
        ("SHARIAH", "Shariah"),
        ("ESG", "ESG"),
        ("QUALITY", "Quality"),
        ("MOMENTUM", "Momentum"),
        ("LOW VOLATILITY", "Low Volatility"),
        ("DIVIDEND", "Dividend"),
        ("LIQUID", "High Liquidity"),
        ("MULTICAP", "Multicap"),
        ("EQUAL WEIGHT", "Equal Weight"),
        ("VALUE", "Value"),
        ("ALPHA", "Alpha"),
        ("GROWTH", "Growth"),
        ("TOTAL MARKET", "Total Market"),
        ("LARGEMIDCAP", "Large-Midcap"),
        ("MIDCAP", "Midcap"),
        ("SMALLCAP", "Smallcap"),
        ("MICROCAP", "Microcap"),
        ("IPO", "IPO"),
    ]
    dataset["PrimarySector"] = "Unclassified"
    for pattern, sector in pattern_sector:
        pattern_key = pattern.replace(" ", "_").upper()
        matching_cols = [col for col in dataset.columns if pattern_key in col.upper()]
        for col in matching_cols:
            mask = (dataset[col] == 1) & (dataset["PrimarySector"] == "Unclassified")
            dataset.loc[mask, "PrimarySector"] = sector

    # ✅ Save final dataset with sector
    output_path = f"/content/drive/My Drive/stock_dataset_with_sector_{datetime.now().strftime('%Y-%m-%d')}.csv"
    dataset.to_csv(output_path, index=False)
    print(f"✅ Dataset updated with PrimarySector and saved as:\n{output_path}")

    # ✅ Preview
    #print(dataset[["Symbol", "PrimarySector"]].drop_duplicates().head(30))
    # Option 1: Show the first 30 rows with all columns
    display(dataset.head(30))  # This shows a nice table in Colab
else:
    print("❌ No stock data was processed.")
