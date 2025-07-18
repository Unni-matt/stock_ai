# ✅ STEP 0: Install required packages
!pip install requests feedparser tqdm chardet --quiet

import requests
import feedparser
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from urllib.parse import quote
import socket

# --- Prevent feedparser from hanging ---
socket.setdefaulttimeout(20)

# ✅ STEP 1: Stock/company-specific news sources
NEWSDATA_API_KEY = "*********************"   # <-- YOUR KEY
symbols = [
    "TATA MOTORS", "RELIANCE", "INFOSYS", "HDFC BANK", "ICICI BANK",
    "ITC", "SBIN", "WIPRO", "BHARTI AIRTEL", "ADANI"
]

def fetch_newsdata_articles(stock_name, max_articles=10):
    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&q={quote(stock_name)}&country=in&language=en&category=business,top,world"
    all_articles = []
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            resp = r.json()
            for article in resp.get("results", [])[:max_articles]:
                all_articles.append({
                    "stock": stock_name,
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "pubDate": article.get("pubDate"),
                    "link": article.get("link"),
                    "source": article.get("source_id", "NewsData.io")
                })
    except Exception as e:
        print(f"❌ Error NewsData {stock_name}: {e}")
    return all_articles

newsdata_news = []
for stock in tqdm(symbols, desc="NewsData.io"):
    newsdata_news.extend(fetch_newsdata_articles(stock, max_articles=5))
newsdata_df = pd.DataFrame(newsdata_news)

def fetch_google_news(stock_name, max_articles=10):
    query = quote(f"{stock_name} stock OR {stock_name} NSE OR {stock_name} BSE")
    url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({
            "stock": stock_name,
            "title": entry.get("title", ""),
            "description": entry.get("description", ""),
            "pubDate": entry.get("published", ""),
            "link": entry.get("link", ""),
            "source": entry.get("source", {}).get("title", "Google News") if entry.get("source") else "Google News"
        })
    return articles

google_news = []
for stock in tqdm(symbols, desc="Google News"):
    google_news.extend(fetch_google_news(stock, max_articles=5))
google_news_df = pd.DataFrame(google_news)

def fetch_yahoo_news(stock_code, max_articles=10):
    possible_symbols = {
        "TATA MOTORS": "TATAMOTORS.NS",
        "RELIANCE": "RELIANCE.NS",
        "INFOSYS": "INFY.NS",
        "HDFC BANK": "HDFCBANK.NS",
        "ICICI BANK": "ICICIBANK.NS",
        "ITC": "ITC.NS",
        "SBIN": "SBIN.NS",
        "WIPRO": "WIPRO.NS",
        "BHARTI AIRTEL": "BHARTIARTL.NS",
        "ADANI": "ADANIENT.NS",
    }
    ticker = possible_symbols.get(stock_code.upper(), "")
    if not ticker: return []
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_articles]:
        articles.append({
            "stock": ticker,
            "title": entry.get("title", ""),
            "description": entry.get("description", ""),
            "pubDate": entry.get("published", ""),
            "link": entry.get("link", ""),
            "source": "Yahoo Finance"
        })
    return articles

yahoo_news = []
for stock in tqdm(symbols, desc="Yahoo Finance"):
    yahoo_news.extend(fetch_yahoo_news(stock, max_articles=5))
yahoo_news_df = pd.DataFrame(yahoo_news)

# ✅ STEP 2: Add ALL possible RSS sources (the giant list you gave!)
rss_sources = [
    # --- INDIAN NATIONAL BUSINESS/FINANCE ---
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380681.cms",
    "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
    "https://www.moneycontrol.com/rss/markets.xml",
    "https://www.moneycontrol.com/rss/buzzingstocks.xml",
    "https://www.moneycontrol.com/rss/banking.xml",
    "https://www.moneycontrol.com/rss/ipo.xml",
    "https://www.moneycontrol.com/rss/commodities.xml",
    "https://www.moneycontrol.com/rss/business.xml",
    "https://www.moneycontrol.com/rss/economy.xml",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://www.business-standard.com/rss/companies-101.rss",
    "https://www.business-standard.com/rss/economy-policy-102.rss",
    "https://www.business-standard.com/rss/international-103.rss",
    "https://www.livemint.com/rss/markets",
    "https://www.livemint.com/rss/companies",
    "https://www.livemint.com/rss/economy",
    "https://www.livemint.com/rss/opinion",
    "https://www.bqprime.com/feed",
    "https://www.financialexpress.com/feed/",
    "https://www.thehindubusinessline.com/markets/stock-markets/?service=rss",
    "https://www.thehindubusinessline.com/economy/?service=rss",
    "https://www.hindustantimes.com/business/rss/topnews/rssfeed.xml",
    "https://www.indiatimes.com/business/rss",
    "https://www.freepressjournal.in/business/feed",
    "https://www.newindianexpress.com/Business/rssfeed/?id=196&getXmlFeed=true",
    "https://zeenews.india.com/rss/business-news.xml",
    "https://www.onmanorama.com/business/news.html/rssfeed.xml",
    "https://www.deccanherald.com/business/rss-feeds",
    "https://theprint.in/feed/",
    # --- SECTORAL / THEMATIC ---
    "https://energy.economictimes.indiatimes.com/rss/topstories.cms",
    "https://realty.economictimes.indiatimes.com/rss/topstories.cms",
    "https://auto.economictimes.indiatimes.com/rss/topstories.cms",
    "https://retail.economictimes.indiatimes.com/rss/topstories.cms",
    "https://cio.economictimes.indiatimes.com/rss/topstories.cms",
    "https://bfsi.economictimes.indiatimes.com/rss/topstories.cms",
    "https://health.economictimes.indiatimes.com/rss/topstories.cms",
    "https://travel.economictimes.indiatimes.com/rss/topstories.cms",
    "https://www.autocarindia.com/rss/news",
    "https://www.cio.in/rss/news",
    "https://www.pharmabiz.com/RSS/RssFeeds.aspx?cat=GENERAL",
    "https://www.foodbusinessnews.net/rss/topic/97-fmcg",
    "https://www.moneycontrol.com/rss/commodities.xml",
    "https://economictimes.indiatimes.com/mf/rssfeeds/4693514.cms",
    # --- IPOs, MUTUAL FUNDS ---
    "https://www.moneycontrol.com/rss/ipo.xml",
    "https://economictimes.indiatimes.com/markets/ipo/news/rssfeeds/52211617.cms",
    "https://www.moneycontrol.com/rss/mf.xml",
    "https://www.valueresearchonline.com/rss/mutual-funds/latest-news/",
    # --- REGIONAL / INDIAN LANGUAGES ---
    "https://www.bhaskar.com/rss-feed/2318/",
    "https://www.jagran.com/rss/business.xml",
    "https://navbharattimes.indiatimes.com/rssfeedstopstories.cms",
    "https://www.divyabhaskar.co.in/rss-feed/25/",
    "https://www.sandesh.com/rss/business.xml",
    "https://www.dinamalar.com/rss_section.asp?sec=business",
    "https://www.manoramaonline.com/business/business-news.xml",
    "https://www.mathrubhumi.com/business-news/rssfeed.xml",
    "https://vijaykarnataka.com/rssfeeds/2280.cms",
    "https://www.andhrajyothy.com/business/feed",
    "https://www.loksatta.com/rss/section/business/",
    "https://www.esakal.com/rss/section/business",
    "https://www.anandabazar.com/business?service=rss",
    "https://www.etemaaddaily.com/rss/biz.xml",
    # --- GLOBAL BUSINESS & FINANCE ---
    "http://feeds.reuters.com/reuters/businessNews",
    "http://feeds.reuters.com/reuters/INbusinessNews",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.bloomberg.com/feed/podcast/bloomberg_markets.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.nasdaq.com/feed/rssoutbound?category=Business",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.ft.com/?format=rss",
    "https://www.wsj.com/xml/rss/3_7031.xml",
    "https://www.forbes.com/business/feed/",
    "https://www.seekingalpha.com/market_currents.xml",
    "https://www.investing.com/rss/news_25.rss",
    "https://asia.nikkei.com/rss/feed/nar",
    "https://www.channelnewsasia.com/business/rss",
    "https://www.businesstimes.com.sg/companies-markets/rss.xml",
    "https://www.smh.com.au/rss/business.xml",
    "https://www.theaustralian.com.au/business/latest-news/rss",
    "https://www.afr.com/rss/feed/breaking-news-seo",
    "https://english.etnews.com/rss/news.xml",
    # --- MISC. (STARTUPS, TECH, POLICY, ETC.) ---
    "https://inc42.com/feed/",
    "https://yourstory.com/feed",
    "https://www.vccircle.com/rssfeeds/allnews",
    "https://www.entrackr.com/feed/",
    "https://trak.in/tags/business/feed/",
    "https://www.startupindia.gov.in/rss/news-feed.xml",
    # --- CRYPTO/WEB3 ---
    "https://cointelegraph.com/rss",
    "https://news.bitcoin.com/feed/",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://www.newsbtc.com/feed/",
    # --- US/EUROPE/ASIA/AFRICA REGIONAL BUSINESS ---
    "https://feeds.skynews.com/feeds/rss/business.xml",
    "https://www.africanews.com/feed/rss/business/",
    "https://www.japantimes.co.jp/feed/",
    "https://www.lemonde.fr/economie/rss_full.xml",
    "https://elpais.com/rss/america_economia.xml",
    # --- OTHER MAJOR NEWS SITES WITH BUSINESS SECTIONS ---
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.scmp.com/rss/91/feed",
    "https://www.theguardian.com/uk/business/rss",
    "https://rss.dw.com/rdf/rss-en-bus",
    "https://news.google.com/rss/search?q=indian+stock+market+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
]

def fetch_rss_articles(rss_sources, max_articles_per_feed=30):
    all_articles = []
    for url in tqdm(rss_sources, desc="Fetching RSS feeds"):
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles_per_feed]:
                all_articles.append({
                    "stock": "",
                    "title": entry.get("title", "").strip(),
                    "description": entry.get("summary", entry.get("description", "")),
                    "pubDate": entry.get("published", entry.get("pubDate", "")),
                    "link": entry.get("link", ""),
                    "source": feed.feed.get("title", url)
                })
        except Exception as e:
            print(f"❌ Error parsing {url}: {e}")
    return all_articles

all_rss_articles = fetch_rss_articles(rss_sources, max_articles_per_feed=30)
all_rss_df = pd.DataFrame(all_rss_articles)

# ✅ STEP 3: Merge ALL sources & Save CSV
all_news = pd.concat([
    newsdata_df, google_news_df, yahoo_news_df, all_rss_df
], ignore_index=True)
all_news.drop_duplicates(subset=['title', 'link'], inplace=True)

csv_path = f"all_news_combined_{datetime.now().strftime('%Y-%m-%d')}.csv"
all_news.to_csv(csv_path, index=False)
print(f"\n✅ All news sources merged and saved as: {csv_path}")

from IPython.display import display
display(all_news.head(30))
