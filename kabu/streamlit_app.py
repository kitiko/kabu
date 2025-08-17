import streamlit as st
import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import ImpersonateError, HTTPError
from bs4 import BeautifulSoup
import logging
import time
import re
from datetime import datetime, date
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import unicodedata
import random
import json
import os
import google.generativeai as genai

# ==============================================================================
# 1. ログ設定
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 1.1. Google Gemini APIの設定
# ==============================================================================
# APIキーを直接コードに設定します
try:
    api_key = "AIzaSyCfRAXzND5SX5gECeq8HGX0_5mSIcFgJMY"
    genai.configure(api_key=api_key)
    logger.info("Gemini APIキーをハードコードで設定しました。")
except Exception as e:
    st.error(f"APIキーの設定中にエラーが発生しました。キーが有効か確認してください。エラー: {e}")
    st.stop()


def generate_prompt(ticker_code, candidate_list_str=None):
    """AI類似銘柄検索用のプロンプトを生成する関数（改善版）"""

    task_instruction = ""
    if candidate_list_str:
        task_instruction = f"""
# 最重要タスク
以下の【候補企業リスト】の中から、前提条件で指定された企業に最も事業内容が類似する企業を最大5社選定してください。リストにない企業は絶対に出力に含めないでください。

【候補企業リスト】
{candidate_list_str}
"""
    else:
        task_instruction = """
# タスク
前提条件で指定された企業に対し、日本市場全体から最も事業内容が類似する企業を最大5社選定してください。
"""

    return f"""
あなたは、豊富な経験を持つプロの株式アナリストです。以下の要件に従い、指定された企業に最も適したピアグループ（競合企業群）を選定してください。

# 目的
指定された企業について、投資価値評価（バリュエーション）や戦略的な相対比較分析を行う上で、最も比較可能性の高いピアグループを客観的かつ論理的な根拠に基づいて特定する。

# 出力形式
銘柄コードのみをカンマ区切りで5つまで出力してください。説明や他のテキストは一切含めないでください。
例: 9984,4755,9432,9433,6758

# 禁止事項
- 単に「大手日本企業」「有名ブランド」「多国籍企業」といった曖昧で高レベルな共通点だけで類似企業を選定しないでください。
- 必ず、企業の主力事業（最も収益を上げているセグメント）が類似していることを最優先の判断基準としてください。

# 良い例と悪い例
- 良い例：ヤクルト（2267）を分析する場合、同じ飲料・食品メーカーである森永乳業（2264）やキリンHD（2503）は適切な類似企業です。
- 悪い例：ヤクルト（2267）に対して、事業内容が全く異なるソニー（6758）や任天堂（7974）は不適切な類似企業です。

# 前提条件
分析対象企業: 証券コード {ticker_code}

{task_instruction}
"""

# ==============================================================================
# 1.5. アクセスコントロール
# ==============================================================================
@st.cache_data
def get_supported_browsers():
    """
    実行環境で利用可能なブラウザ偽装プロファイルを動的にテストし、
    サポートされているプロファイルのリストを返す。
    """
    potential_browsers = [
        "chrome124", "chrome123", "chrome120", "chrome119", "chrome117", "chrome116",
        "chrome120_android", "safari17_0"
    ]
    supported = []
    logger.info("利用可能なブラウザプロファイルの確認を開始します...")
    for browser in potential_browsers:
        try:
            s = curl_requests.Session()
            s.impersonate = browser
            s.get("https://www.google.com", timeout=10)
            supported.append(browser)
            logger.info(f"  ✅ プロファイル '{browser}' はサポートされています。")
        except ImpersonateError:
            logger.warning(f"  ❌ プロファイル '{browser}' はサポートされていません。")
        except Exception as e:
            logger.warning(f"  ⚠️ プロファイル '{browser}' のテスト中にエラーが発生しました: {e}")
    if not supported:
        logger.error("重大: サポートされているブラウザプロファイルが見つかりませんでした。")
        st.error("利用可能なブラウザプロファイルが見つかりませんでした。アプリを続行できません。")
    return supported

class BrowserRotator:
    def __init__(self):
        supported_list = get_supported_browsers()
        if not supported_list:
            raise RuntimeError("サポートされているブラウザがないため、処理を続行できません。")
        chrome_weights = {
            "chrome116": 5, "chrome117": 8, "chrome119": 20,
            "chrome120": 25, "chrome123": 15, "chrome124": 12
        }
        self.chrome_versions = []
        self.mobile_versions = []
        self.safari_versions = []
        for browser in supported_list:
            if "android" in browser:
                self.mobile_versions.append(browser)
            elif browser.startswith("safari"):
                self.safari_versions.append(browser)
            elif browser.startswith("chrome"):
                weight = chrome_weights.get(browser, 1)
                self.chrome_versions.append((browser, weight))
        logger.info(f"初期化完了。Chrome: {[v[0] for v in self.chrome_versions]}, Mobile: {self.mobile_versions}, Safari: {self.safari_versions}")

    def get_random_browser(self):
        browser_types = []
        if self.chrome_versions: browser_types.append(("chrome", 65))
        if self.mobile_versions: browser_types.append(("mobile", 25))
        if self.safari_versions: browser_types.append(("safari", 10))
        if not browser_types:
            if self.chrome_versions:
                versions, weights = zip(*self.chrome_versions)
                return random.choices(versions, weights=weights, k=1)[0]
            raise RuntimeError("回転に使用できるブラウザがありません。")
        population, weights = zip(*browser_types)
        chosen_type = random.choices(population, weights=weights, k=1)[0]
        if chosen_type == "chrome":
            versions, weights = zip(*self.chrome_versions)
            return random.choices(versions, weights=weights, k=1)[0]
        elif chosen_type == "mobile":
            return random.choice(self.mobile_versions)
        else:
            return random.choice(self.safari_versions)

# ==============================================================================
# 2. ヘルパー関数
# ==============================================================================
def create_copy_button(text_to_copy: str, button_text: str, key: str):
    js_escaped_text = json.dumps(text_to_copy)
    button_id = f"copy-btn-{key}"
    html_code = f"""
    <style>
        #{button_id} {{
            display: inline-flex; align-items: center; justify-content: center;
            font-weight: 400; padding: 0.25rem 0.75rem; border-radius: 0.5rem;
            min-height: 38.4px; margin: 0px; line-height: 1.6; color: #31333F;
            background-color: #FFFFFF; border: 1px solid rgba(49, 51, 63, 0.2);
            cursor: pointer; transition: all .2s ease-in-out;
        }}
        #{button_id}:hover {{ border-color: #FF4B4B; color: #FF4B4B; }}
        #{button_id}.copied {{ border-color: #008000; color: #008000; }}
    </style>
    <button id="{button_id}">{button_text}</button>
    <script>
        document.getElementById('{button_id}').addEventListener('click', function() {{
            navigator.clipboard.writeText({js_escaped_text}).then(() => {{
                let btn = document.getElementById('{button_id}');
                const originalText = btn.innerHTML;
                btn.innerHTML = '✅ Copied!';
                btn.classList.add('copied');
                setTimeout(() => {{
                    btn.innerHTML = originalText;
                    btn.classList.remove('copied');
                }}, 2000);
            }}).catch(err => {{ console.error('Failed to copy: ', err); }});
        }});
    </script>
    """
    st.components.v1.html(html_code, height=50)

# ==============================================================================
# 3. 銘柄検索用のヘルパー関数とデータロード
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JPX_STOCK_LIST_PATH = os.path.join(BASE_DIR, "jpx_list.xls")

@st.cache_data
def load_jpx_stock_list():
    try:
        df = pd.read_excel(JPX_STOCK_LIST_PATH, header=None, engine='xlrd')
        # ★修正: 業種コードのカラム(4)も読み込む
        if df.shape[1] < 6:
            st.error(f"銘柄リストファイル({JPX_STOCK_LIST_PATH})の形式が想定と異なります。")
            return pd.DataFrame(columns=['code', 'name', 'market', 'sector_code', 'sector', 'normalized_name'])
        df = df.iloc[:, [1, 2, 3, 4, 5]]
        df.columns = ['code', 'name', 'market', 'sector_code', 'sector']
        df.dropna(subset=['code', 'name', 'sector_code'], inplace=True)

        # 銘柄コードを文字列として確実に処理する
        def clean_code(x):
            if pd.isna(x):
                return ""
            if isinstance(x, float):
                return str(int(x))
            return str(x).strip().upper()

        df['code'] = df['code'].apply(clean_code)
        # 業種コードを数値型に変換
        df['sector_code'] = pd.to_numeric(df['sector_code'], errors='coerce').astype('Int64')
        df = df[df['code'].str.fullmatch(r'(\d{4}|\d{3}[A-Z])', na=False)]
        df['normalized_name'] = df['name'].apply(normalize_text)
        logger.info(f"銘柄リストをロードしました: {len(df)}件")
        return df
    except FileNotFoundError:
        st.error(f"銘柄リストファイル ({JPX_STOCK_LIST_PATH}) が見つかりません。")
        return pd.DataFrame(columns=['code', 'name', 'market', 'sector_code', 'sector', 'normalized_name'])
    except Exception as e:
        if "xlrd" in str(e).lower():
            st.error("Excelファイル(.xls)を読み込むためのライブラリ 'xlrd' がインストールされていません。")
        else:
            st.error(f"銘柄リストの読み込み中に予期せぬエラーが発生しました: {e}")
        return pd.DataFrame(columns=['code', 'name', 'market', 'sector_code', 'sector', 'normalized_name'])

def normalize_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text)
    text = "".join([chr(ord(c) + 96) if "ぁ" <= c <= "ん" else c for c in text])
    text = text.upper()
    remove_words = ['ホールディングス', 'グループ', '株式会社', '合同会社', '有限会社', '(株)', '(同)', '(有)', ' ', '　', '・', '-']
    for word in remove_words:
        text = text.replace(word, '')
    return text.strip()

# ==============================================================================
# 戦略と業種の定義
# ==============================================================================
STRATEGY_WEIGHTS = {
    "⚖️ バランス型（バランス）": {"safety": 0.25, "value": 0.25, "quality": 0.25, "growth": 0.25},
    "💎 バリュー重視（価値重視）": {"safety": 0.35, "value": 0.40, "quality": 0.20, "growth": 0.05},
    "🚀 グロース重視（成長重視）": {"safety": 0.10, "value": 0.20, "quality": 0.35, "growth": 0.35},
    "🛡️ 健全性重視（安全第一）": {"safety": 0.50, "value": 0.25, "quality": 0.15, "growth": 0.10}
}
# ★追加: シクリカル銘柄の業種コードを定義
CYCLICAL_SECTOR_CODES = {
    1050, 3100, 3150, 3200, 3300, 3350, 3400, 3450, 3500, 3550, 3600, 3650, 3700, 3750, 5050, 5100, 5150, 5200, 6050
}

# ==============================================================================
# データ処理クラス
# ==============================================================================
class IntegratedDataHandler:
    def __init__(self):
        self.stock_list_df = load_jpx_stock_list()
        self.browser_rotator = BrowserRotator()
        self.session = None

    def _reset_session(self):
        logger.info("新しいセッションを初期化します...")
        self.session = curl_requests.Session()
        try:
            selected_version = self.browser_rotator.get_random_browser()
            self.session.impersonate = selected_version
            logger.info(f"セッションの偽装バージョンとして '{selected_version}' を使用します。")
            self.session.get("https://www.buffett-code.com/", timeout=20)
            logger.info("セッションのウォームアップに成功しました。")
        except Exception as e:
            logger.error(f"セッションの初期化（ウォームアップ）に失敗しました: {e}", exc_info=True)
            st.error(f"バフェットコードへの初期アクセスに失敗しました。詳細: {e}")
            self.session = None

    def get_ticker_info_from_query(self, query: str) -> dict | None:
        query_original = query.strip()
        query_upper = query_original.upper()
        if re.fullmatch(r'(\d{4}|\d{3}[A-Z])', query_upper):
            code_to_search = query_upper
            if not self.stock_list_df.empty:
                stock_data = self.stock_list_df[self.stock_list_df['code'] == code_to_search]
                if not stock_data.empty:
                    return stock_data.iloc[0].to_dict()
                else:
                    logger.warning(f"銘柄コード '{code_to_search}' はリストに存在しませんが、分析を試みます。")
                    return {'code': code_to_search, 'name': f'銘柄 {code_to_search}', 'sector': '業種不明', 'sector_code': None}
            return {'code': code_to_search, 'name': f'銘柄 {code_to_search}', 'sector': '業種不明', 'sector_code': None}
        if self.stock_list_df.empty: return None
        normalized_query = normalize_text(query_original)
        if not normalized_query: return None
        matches = self.stock_list_df[self.stock_list_df['normalized_name'].str.contains(normalized_query, na=False)].copy()
        if not matches.empty:
            prime_matches = matches[matches['market'].str.contains('プライム', na=False)]
            target_df = prime_matches if not prime_matches.empty else matches
            target_df.loc[:, 'diff'] = target_df['normalized_name'].apply(lambda x: abs(len(x) - len(normalized_query)))
            stock_data = target_df.sort_values(by='diff').iloc[0]
            logger.info(f"検索クエリ '{query_original}' から銘柄 '{stock_data['name']} ({stock_data['code']})' を見つけました。")
            return stock_data.to_dict()
        logger.warning(f"検索クエリ '{query_original}' に一致する銘柄が見つかりませんでした。")
        return None

    YFINANCE_TRANSLATION_MAP = {
        'Total Revenue': '売上高', 'Revenue': '売上高', 'Operating Income': '営業利益', 'Operating Expense': '営業費用',
        'Cost Of Revenue': '売上原価', 'Gross Profit': '売上総利益', 'Selling General And Administration': '販売費及び一般管理費',
        'Research And Development': '研究開発費', 'Pretax Income': '税引前利益', 'Tax Provision': '法人税',
        'Net Income': '当期純利益', 'Net Income Common Stockholders': '親会社株主に帰属する当期純利益', 'Basic EPS': '1株当たり利益 (EPS)',
        'Diluted EPS': '希薄化後EPS', 'Total Assets': '総資産', 'Current Assets': '流動資産',
        'Cash And Cash Equivalents': '現金及び現金同等物', 'Cash': '現金', 'Receivables': '売上債権', 'Inventory': '棚卸資産',
        'Total Non Current Assets': '固定資産', 'Net PPE': '有形固定資産', 'Goodwill And Other Intangible Assets': 'のれん及びその他無形固定資産',
        'Total Liabilities Net Minority Interest': '負債合計', 'Current Liabilities': '流動負債', 'Payables And Accrued Expenses': '支払手形及び買掛金',
        'Current Debt': '短期有利子負債', 'Total Non Current Liabilities Net Minority Interest': '固定負債', 'Long Term Debt': '長期有利子負債',
        'Total Equity Gross Minority Interest': '純資産合計', 'Stockholders Equity': '株主資本', 'Retained Earnings': '利益剰余金',
        'Cash Flow From Continuing Operating Activities': '営業キャッシュフロー', 'Cash Flow From Continuing Investing Activities': '投資キャッシュフロー',
        'Cash Flow From Continuing Financing Activities': '財務キャッシュフロー', 'Net Change In Cash': '現金の増減額', 'Free Cash Flow': 'フリーキャッシュフロー',
    }

    def get_html_soup(self, url: str, retries: int = 3) -> BeautifulSoup | None:
        for attempt in range(retries):
            if self.session is None:
                logger.warning("セッションが無効です。新しいセッションを初期化します。")
                self._reset_session()
                if self.session is None:
                    st.error("セッションの再初期化に失敗しました。処理を中断します。")
                    return None
            logger.info(f"URLにアクセス試行 ({attempt + 1}/{retries}): {url}")
            try:
                headers = {
                    'Referer': 'https://www.buffett-code.com/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Sec-Ch-Ua': f'"Chromium";v="{self.session.impersonate.split("chrome")[1]}", "Not/A)Brand";v="99"',
                    'Sec-Ch-Ua-Mobile': '?0',
                    'Sec-Ch-Ua-Platform': '"Windows"',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'same-origin',
                    'Sec-Fetch-User': '?1',
                    'Upgrade-Insecure-Requests': '1',
                }
                wait_time = random.uniform(4.0, 7.0) * (attempt + 1)
                logger.info(f"{wait_time:.2f}秒待機します...")
                time.sleep(wait_time)
                response = self.session.get(url, timeout=30, headers=headers)
                response.raise_for_status()
                logger.info(f"URLへのアクセス成功 (ステータスコード: {response.status_code}): {url}")
                return BeautifulSoup(response.content, 'html.parser')
            except HTTPError as e:
                logger.error(f"HTTPエラー発生 (試行 {attempt + 1}/{retries}): {url}, ステータス: {e.response.status_code}, エラー: {e}", exc_info=False)
                if e.response.status_code in [403, 405, 429]:
                    logger.warning("アクセスがブロックされた可能性があるため、セッションをリセットして再試行します。")
                    self._reset_session()
                elif e.response.status_code >= 500:
                    time.sleep(10)
            except Exception as e:
                logger.error(f"予期せぬエラー発生 (試行 {attempt + 1}/{retries}): {url}, エラー: {e}", exc_info=True)
                self._reset_session()
        st.error(f"バフェットコードへのアクセスに失敗しました ({retries}回試行後)。サイトがメンテナンス中か、IPがブロックされた可能性があります。")
        return None

    def get_risk_free_rate(self) -> float | None:
        url = "https://jp.investing.com/rates-bonds/japan-10-year-bond-yield"
        logger.info(f"リスクフリーレート取得試行 (新規セッション使用): {url}")
        try:
            with curl_requests.Session() as temp_session:
                impersonate_version = self.browser_rotator.get_random_browser()
                temp_session.impersonate = impersonate_version
                logger.info(f"Investing.comへのアクセスに '{impersonate_version}' を使用します。")
                time.sleep(random.uniform(1.0, 2.0))
                response = temp_session.get(url, timeout=25)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                yield_element = soup.find('div', attrs={'data-test': 'instrument-price-last'})
                if not yield_element: yield_element = soup.find('div', class_=re.compile(r'instrument-price_last__'))
                if not yield_element: yield_element = soup.select_one('[data-test="instrument-price-last"], .instrument-price_last__2wE7v')
                if yield_element:
                    rate = float(yield_element.text.strip()) / 100
                    logger.info(f"リスクフリーレートの取得に成功しました: {rate:.4f}")
                    return rate
                else:
                    logger.error("金利データが見つかりませんでした。")
                    st.toast("⚠️ 金利データが見つかりませんでした。", icon="⚠️")
                    return None
        except Exception as e:
            logger.error(f"リスクフリーレートの取得に失敗しました: {e}", exc_info=True)
            st.toast("⚠️ リスクフリーレートの取得に失敗しました。", icon="⚠️")
            return None

    def get_listing_date(self, ticker_code: str) -> str | None:
        url = f"https://finance.yahoo.co.jp/quote/{ticker_code}.T/profile"
        logger.info(f"上場年月日取得試行 (新規セッション使用): {url}")
        try:
            with curl_requests.Session() as temp_session:
                impersonate_version = self.browser_rotator.get_random_browser()
                temp_session.impersonate = impersonate_version
                logger.info(f"Yahoo Financeへのアクセスに '{impersonate_version}' を使用します。")
                time.sleep(random.uniform(1.0, 2.0))
                response = temp_session.get(url, timeout=25)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                th_tag = soup.find('th', string='上場年月日')
                if th_tag:
                    td_tag = th_tag.find_next_sibling('td')
                    if td_tag:
                        listing_date_str = td_tag.get_text(strip=True)
                        logger.info(f"銘柄 {ticker_code} の上場年月日 '{listing_date_str}' を取得しました。")
                        return listing_date_str
                logger.warning(f"銘柄 {ticker_code} の上場年月日が見つかりませんでした。")
                return None
        except Exception as e:
            logger.error(f"上場年月日の取得中にエラーが発生しました ({ticker_code}): {e}", exc_info=True)
            st.toast(f"⚠️ {ticker_code}の上場日取得に失敗しました。", icon="⚠️")
            return None

    def parse_financial_value(self, s: str) -> int | float | None:
        s = str(s).replace(',', '').strip()
        if s in ['-', '---', '']: return None
        is_negative = s.startswith(('△', '-'))
        s = s.lstrip('△-')
        try:
            total = 0
            if '兆' in s: total += float(re.findall(r'(\d+\.?\d*)', s)[0]) * 1000000
            elif '億' in s: total += float(re.findall(r'(\d+\.?\d*)', s)[0]) * 100
            elif '百万円' in s: total += float(re.findall(r'(\d+\.?\d*)', s)[0])
            elif '万円' in s: total += float(re.findall(r'(\d+\.?\d*)', s)[0]) * 0.01
            elif re.match(r'^\d+\.?\d*$', s): total = float(s)
            else: return s
            return -int(total) if is_negative else int(total)
        except (ValueError, TypeError, IndexError): return s

    def extract_all_financial_data(self, soup: BeautifulSoup) -> dict | None:
        financial_table = soup.find('table', class_='financial-table')
        if not financial_table: return None
        thead, tbody = financial_table.find('thead'), financial_table.find('tbody')
        if not thead or not tbody: return None
        period_headers = thead.find('tr').find_all('th')
        if len(period_headers) <= 1: return None
        valid_periods = []
        for i, th in enumerate(period_headers[1:]):
            header_text = th.text.strip()
            if header_text and "E" not in header_text.upper() and "C" not in header_text.upper():
                valid_periods.append({'name': header_text, 'index': i + 1})
        if not valid_periods: return None
        all_periods_data = OrderedDict()
        for row in tbody.find_all('tr'):
            cells = row.find_all(['th', 'td'])
            item_name = cells[0].text.strip()
            if not item_name or not re.search(r'[a-zA-Z\u3040-\u30FF\u4E00-\u9FFF]', item_name): continue
            for period in valid_periods:
                period_name = period['name']
                if period_name not in all_periods_data: all_periods_data[period_name] = {}
                if len(cells) > period['index']:
                    display_value = cells[period['index']].get_text(strip=True)
                    if display_value not in ['-', '---', '']:
                        all_periods_data[period_name][item_name] = {'display': display_value, 'raw': self.parse_financial_value(display_value)}
        return all_periods_data

    def get_latest_financial_data(self, financial_data_dict: dict) -> dict:
        latest_year, latest_month, latest_data = -1, -1, {}
        if not financial_data_dict: return {}
        for period_name, data in financial_data_dict.items():
            match = re.search(r'(\d{2,4})[./](\d{1,2})', period_name)
            if match:
                year_str, month_str = match.groups()
                year = int(year_str)
                month = int(month_str)
                year_full = 2000 + year if year < 50 else 1900 + year if year < 100 else year
                if year_full > latest_year or (year_full == latest_year and month > latest_month):
                    latest_year, latest_month, latest_data = year_full, month, data
        return latest_data

    def get_value(self, data_dict: dict, keys: list[str], log_name: str) -> any:
        for key in keys:
            if key in data_dict and data_dict[key].get('raw') is not None:
                value = data_dict[key]['raw']
                logger.info(f"✅ {log_name}: 項目 '{key}' から値 ({value}) を取得しました。")
                return value
        logger.warning(f"⚠️ {log_name}: 項目が見つかりませんでした (試行キー: {keys})")
        return None

    def format_yfinance_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df_copy = df.copy()
        df_copy = df_copy.rename(index=self.YFINANCE_TRANSLATION_MAP)
        df_copy = df_copy.loc[df_copy.index.isin(self.YFINANCE_TRANSLATION_MAP.values())]
        df_copy.columns = [f"{col.year}.{col.month}" for col in df_copy.columns]
        exclude_rows = [name for name in df_copy.index if 'EPS' in name or '比率' in name or 'Rate' in name]
        for idx in df_copy.index:
            if idx not in exclude_rows:
                df_copy.loc[idx] = df_copy.loc[idx].apply(lambda x: x / 1e6 if pd.notna(x) else np.nan)
        return df_copy

    def _linear_interpolate(self, value, x1, y1, x2, y2):
        if value <= x1: return y1
        if value >= x2: return y2
        return y1 + (y2 - y1) * (value - x1) / (x2 - x1)

    def _score_net_cash_ratio(self, ratio):
        if ratio is None: return {'score': 0, 'evaluation': '---'}
        if ratio >= 1.0: return {'score': 100, 'evaluation': '【超安全圏・鉄壁】'}
        if ratio >= 0.8: return {'score': self._linear_interpolate(ratio, 0.8, 90, 1.0, 100), 'evaluation': '【極めて安全】'}
        if ratio >= 0.5: return {'score': self._linear_interpolate(ratio, 0.5, 80, 0.8, 90), 'evaluation': '【非常に安全・割安】'}
        if ratio >= 0.2: return {'score': self._linear_interpolate(ratio, 0.2, 60, 0.5, 80), 'evaluation': '【安全圏】'}
        if ratio >= 0.1: return {'score': self._linear_interpolate(ratio, 0.1, 40, 0.2, 60), 'evaluation': '【やや注意】'}
        if ratio > 0.01: return {'score': self._linear_interpolate(ratio, 0.01, 20, 0.1, 40), 'evaluation': '【要注意】'}
        if ratio > 0: return {'score': self._linear_interpolate(ratio, 0, 0, 0.01, 20), 'evaluation': '【要注意】'}
        return {'score': 0, 'evaluation': '【要警戒】'}

    def _score_cn_per(self, cn_per, keijo_rieki, pe, trailing_eps):
        if pe is None and trailing_eps is not None and trailing_eps < 0: return {'score': 10, 'evaluation': '【赤字企業 (EPS基準)】'}
        if not (keijo_rieki is not None and keijo_rieki > 0): return {'score': 10, 'evaluation': '【赤字・要注意】'}
        if cn_per is None: return {'score': 0, 'evaluation': '---'}
        if cn_per < 0: return {'score': 100, 'evaluation': '【究極の割安株】'}
        if cn_per < 2: return {'score': self._linear_interpolate(cn_per, 0, 100, 2, 95), 'evaluation': '【現金より安い会社】'}
        if cn_per < 4: return {'score': self._linear_interpolate(cn_per, 2, 95, 4, 90), 'evaluation': '【投資のど真ん中】'}
        if cn_per < 7: return {'score': self._linear_interpolate(cn_per, 4, 90, 7, 80), 'evaluation': '【まあ、悪くない】'}
        if cn_per < 10: return {'score': self._linear_interpolate(cn_per, 7, 80, 10, 70), 'evaluation': '【普通の会社】'}
        if cn_per < 15: return {'score': self._linear_interpolate(cn_per, 10, 70, 15, 50), 'evaluation': '【割高に思える】'}
        return {'score': 20, 'evaluation': '【論外・バブル】'}

    def _score_roic(self, roic, wacc):
        if roic is None: return {'score': 0, 'evaluation': '---'}
        roic_percent = roic * 100
        if roic_percent >= 20: return {'score': 100, 'evaluation': '【ワールドクラス】'}
        if roic_percent >= 15: return {'score': self._linear_interpolate(roic_percent, 15, 90, 20, 100), 'evaluation': '【業界の支配者】'}
        if roic_percent >= 10: return {'score': self._linear_interpolate(roic_percent, 10, 80, 15, 90), 'evaluation': '【優れた資本効率】'}
        if roic_percent >= 7: return {'score': self._linear_interpolate(roic_percent, 7, 70, 10, 80), 'evaluation': '【優良の入り口】'}
        if wacc is not None:
            if roic >= wacc: return {'score': self._linear_interpolate(roic_percent, wacc * 100, 60, 7, 70), 'evaluation': '【合格ライン】'}
            if roic < wacc: return {'score': self._linear_interpolate(roic_percent, 0, 40, wacc * 100, 60), 'evaluation': '【価値破壊】'}
        if roic < 0: return {'score': 20, 'evaluation': '【深刻な問題】'}
        return {'score': 40, 'evaluation': '【価値破壊】'}

    def _calculate_peg_score(self, peg_ratio: float | None) -> dict:
        if peg_ratio is None or peg_ratio < 0:
            score, evaluation = 0, "【成長鈍化・赤字】" if peg_ratio is not None else "---"
        elif peg_ratio <= 0.5: score, evaluation = 100, "【超割安な成長株】"
        elif peg_ratio <= 1.0: score, evaluation = self._linear_interpolate(peg_ratio, 0.5, 100, 1.0, 70), "【割安な成長株】"
        elif peg_ratio <= 1.5: score, evaluation = self._linear_interpolate(peg_ratio, 1.0, 70, 1.5, 40), "【適正価格】"
        elif peg_ratio < 2.0: score, evaluation = self._linear_interpolate(peg_ratio, 1.5, 40, 2.0, 0), "【やや割高】"
        else: score, evaluation = 0, "【割高】"
        return {'score': int(score), 'evaluation': evaluation}

    def _get_alternative_per(self, ticker_obj, info: dict) -> dict:
        trailing_pe = info.get('trailingPE')
        if trailing_pe is not None and trailing_pe > 0:
            logger.info(f"PER取得方法1: yfinance.infoから取得しました (trailingPE: {trailing_pe:.2f})")
            return {'value': trailing_pe, 'note': None}
        current_price = info.get('regularMarketPrice')
        trailing_eps = info.get('trailingEps')
        if current_price is not None and trailing_eps is not None and trailing_eps > 0:
            calculated_per = current_price / trailing_eps
            logger.info(f"PER取得方法2: 最新株価({current_price}) / 最新EPS({trailing_eps}) で計算しました (PER: {calculated_per:.2f})")
            return {'value': calculated_per, 'note': None}
        try:
            financials = ticker_obj.financials
            history = ticker_obj.history(period="3y")
            if not financials.empty and 'Basic EPS' in financials.index and not history.empty:
                if hasattr(history.index.dtype, 'tz') and history.index.dtype.tz is not None:
                    history.index = history.index.tz_localize(None)
                eps_series = financials.loc['Basic EPS'].dropna()
                if len(eps_series) >= 1:
                    latest_settlement_date = eps_series.index[0]
                    latest_eps = eps_series.iloc[0]
                    price_on_settlement = history.asof(latest_settlement_date)['Close']
                    if pd.notna(price_on_settlement) and pd.notna(latest_eps) and latest_eps > 0:
                        calculated_per = price_on_settlement / latest_eps
                        date_str = latest_settlement_date.strftime('%Y-%m-%d')
                        note = f"注記: 手計算PER ({date_str}時点)をPERとして代用"
                        logger.info(f"PER取得方法3: 直近決算データで計算 ({price_on_settlement:.2f} / {latest_eps:.2f} = {calculated_per:.2f})")
                        return {'value': calculated_per, 'note': note}
                if len(eps_series) >= 2:
                    prev_settlement_date = eps_series.index[1]
                    prev_eps = eps_series.iloc[1]
                    price_on_settlement = history.asof(prev_settlement_date)['Close']
                    if pd.notna(price_on_settlement) and pd.notna(prev_eps) and prev_eps > 0:
                        calculated_per = price_on_settlement / prev_eps
                        date_str = prev_settlement_date.strftime('%Y-%m-%d')
                        note = f"注記: 手計算PER ({date_str}時点)をPERとして代用"
                        logger.info(f"PER取得方法4: 2期前決算データで計算 ({price_on_settlement:.2f} / {prev_eps:.2f} = {calculated_per:.2f})")
                        return {'value': calculated_per, 'note': note}
        except Exception as e:
            logger.warning(f"代替PERの計算中にエラーが発生しました: {e}", exc_info=True)
        logger.warning("全てのPER取得/計算方法が失敗しました。")
        return {'value': None, 'note': None}

    def _calculate_scoring_indicators(self, all_fin_data: dict, yf_data: dict) -> dict:
        indicators = {'calc_warnings': [], 'formulas': {}, 'variables': {}}
        latest_bs_data = self.get_latest_financial_data(all_fin_data.get('貸借対照表', {}))
        latest_pl_data = self.get_latest_financial_data(all_fin_data.get('損益計算書', {}))
        market_cap, pe, rf_rate, mrp, trailing_eps = (yf_data.get(k) for k in ['marketCap', 'trailingPE', 'risk_free_rate', 'mkt_risk_premium', 'trailingEps'])
        beta = yf_data.get('beta')
        indicators['variables']['時価総額'] = market_cap
        indicators['variables']['PER (実績)'] = pe
        indicators['variables']['ベータ値'] = beta
        if beta is None:
            beta = 1.0
            indicators['calc_warnings'].append("注記: β値の代わりに1.0で代用")
        securities_keys = ['有価証券', '投資有価証券', 'その他の金融資産']
        securities = self.get_value(latest_bs_data, securities_keys, '有価証券')
        if securities is not None and securities < 0:
            indicators['calc_warnings'].append("注記: 有価証券がマイナスだったため0として計算")
            securities = 0
        indicators['variables']['有価証券'] = securities
        if securities is None:
            securities = 0
            indicators['calc_warnings'].append("注記: 有価証券が見つからないため0として計算")
        op_income = self.get_value(latest_pl_data, ['営業利益'], '営業利益')
        op_income_source = '営業利益'
        if op_income is None:
            op_income = self.get_value(latest_pl_data, ['税引前利益', '税金等調整前当期純利益'], '税引前利益(代替)')
            op_income_source = '税引前利益'
        if op_income is None:
            op_income = self.get_value(latest_pl_data, ['当期純利益', '親会社株主に帰属する当期純利益'], '当期純利益(代替)')
            op_income_source = '当期純利益'
        indicators['roic_source_key'] = op_income_source
        if op_income_source != '営業利益' and op_income is not None:
            indicators['calc_warnings'].append(f"信頼性警告: 営業利益の代わりに「{op_income_source}」を使用")
        indicators['variables'][f'NOPAT計算用利益 ({op_income_source})'] = op_income
        net_assets = self.get_value(latest_bs_data, ['純資産合計', '純資産'], '純資産')
        pretax_income = self.get_value(latest_pl_data, ['税引前利益', '税金等調整前当期純利益'], '税引前利益')
        corp_tax = self.get_value(latest_pl_data, ['法人税等', '法人税、住民税及び事業税'], '法人税等')
        keijo_rieki = self.get_value(latest_pl_data, ['経常利益'], '経常利益')
        net_income = self.get_value(latest_pl_data, ['当期純利益', '親会社株主に帰属する当期純利益'], '当期純利益')
        indicators['variables']['純資産'] = net_assets
        indicators['variables']['経常利益'] = keijo_rieki
        indicators['variables']['当期純利益'] = net_income
        def check_reqs(reqs, names):
            missing = [name for req, name in zip(reqs, names) if req is None]
            return None if not missing else f"不足: {', '.join(missing)}"
        current_assets = self.get_value(latest_bs_data, ['流動資産合計', '流動資産'], '流動資産')
        total_liabilities = self.get_value(latest_bs_data, ['負債合計'], '負債')
        if total_liabilities is None:
            total_liabilities = self.get_value(latest_bs_data, ['負債'], '負債')
            if total_liabilities is not None:
                indicators['calc_warnings'].append("注記: NC比率計算で「負債合計」の代わりに「負債」で代用")
        indicators['variables']['流動資産'] = current_assets
        indicators['variables']['負債合計'] = total_liabilities
        nc_ratio, nc_error = None, None
        nc_reqs, nc_names = [market_cap, current_assets, securities, total_liabilities], ["時価総額", "流動資産", "有価証券", "負債合計"]
        nc_error = check_reqs(nc_reqs, nc_names)
        if not nc_error:
            if market_cap > 0:
                nc_ratio = (current_assets + (securities * 0.7) - total_liabilities) / (market_cap / 1_000_000)
                indicators['formulas']['ネットキャッシュ比率'] = f"({current_assets:,.0f} + {securities:,.0f}*0.7 - {total_liabilities:,.0f}) / {market_cap/1e6:,.0f}"
            else:
                nc_error = "時価総額がゼロです"
        cnper_reqs, cnper_names = [pe, nc_ratio], ["PER", "ネットキャッシュ比率"]
        cn_per, cnper_error = None, check_reqs(cnper_reqs, cnper_names)
        if not cnper_error:
            cn_per = pe * (1 - nc_ratio)
            indicators['formulas']['キャッシュニュートラルPER'] = f"{pe:.2f} * (1 - {nc_ratio:.2f})"
        tax_rate = corp_tax / pretax_income if all(v is not None for v in [corp_tax, pretax_income]) and pretax_income > 0 else 0.3062
        indicators['variables']['税率'] = tax_rate
        debt = self.get_value(latest_bs_data, ['有利子負債合計', '有利子負債'], '有利子負債')
        net_debt = self.get_value(latest_bs_data, ['純有利子負債'], '純有利子負債')
        cash = self.get_value(latest_bs_data, ['現金', '現金及び預金'], '現金同等物')
        indicators['variables']['有利子負債'] = debt
        indicators['variables']['純有利子負債'] = net_debt
        indicators['variables']['現金同等物'] = cash
        interest_expense = self.get_value(latest_pl_data, ['支払利息', '金融費用'], '支払利息')
        cost_of_equity = rf_rate + beta * mrp if all(v is not None for v in [beta, rf_rate, mrp]) else None
        indicators['variables']['株主資本コスト'] = cost_of_equity
        effective_debt_for_wacc = debt
        if debt is None and net_debt is not None and cash is not None:
            effective_debt_for_wacc = net_debt + cash
            if effective_debt_for_wacc < 0: effective_debt_for_wacc = 0
        cost_of_debt = interest_expense / effective_debt_for_wacc if all(v is not None for v in [interest_expense, effective_debt_for_wacc]) and effective_debt_for_wacc > 0 else 0.0
        indicators['variables']['負債コスト'] = cost_of_debt
        wacc_reqs, wacc_names = [cost_of_equity, market_cap, effective_debt_for_wacc], ["株主資本コスト", "時価総額", "有利子負債(または代用値)"]
        wacc_error = check_reqs(wacc_reqs, wacc_names)
        wacc = None
        if not wacc_error:
            e, d_yen = market_cap, effective_debt_for_wacc * 1_000_000
            v = e + d_yen
            if v > 0:
                wacc = cost_of_equity * (e / v) + cost_of_debt * (1 - tax_rate) * (d_yen / v)
                indicators['formulas']['WACC'] = f"Ke {cost_of_equity:.2%} * (E/V {(e/v):.2%}) + Kd {cost_of_debt:.2%} * (1-T {tax_rate:.2%}) * (D/V {(d_yen/v):.2%})"
        roic, roic_error = None, None
        invested_capital_debt = debt
        if debt is None and net_debt is not None and cash is not None:
            invested_capital_debt = net_debt + cash
            indicators['calc_warnings'].append("注記: ROIC計算で純有利子負債を代用")
        roic_reqs, roic_names = [op_income, net_assets, invested_capital_debt], [op_income_source, "純資産", "有利子負債(または代用値)"]
        roic_error = check_reqs(roic_reqs, roic_names)
        if not roic_error:
            invested_capital = net_assets + invested_capital_debt
            nopat = op_income * (1 - tax_rate)
            if invested_capital > 0:
                roic = nopat / invested_capital
                indicators['formulas']['ROIC'] = f"{nopat:,.0f} / {invested_capital:,.0f}"
        nc_score_dict = self._score_net_cash_ratio(nc_ratio)
        cn_per_score_dict = self._score_cn_per(cn_per, keijo_rieki, pe, trailing_eps)
        roic_score_dict = self._score_roic(roic, wacc)
        indicators['net_cash_ratio'] = {'value': nc_ratio, 'reason': nc_error, **nc_score_dict}
        indicators['cn_per'] = {'value': cn_per, 'reason': cnper_error, **cn_per_score_dict}
        indicators['roic'] = {'value': roic, 'reason': roic_error, **roic_score_dict}
        indicators['wacc'] = {'value': wacc, 'reason': wacc_error}
        return indicators

    def get_yfinance_statements(self, ticker_obj):
        statements = {
            "年次損益計算書": self.format_yfinance_df(ticker_obj.financials),
            "四半期損益計算書": self.format_yfinance_df(ticker_obj.quarterly_financials),
            "年次貸借対照表": self.format_yfinance_df(ticker_obj.balance_sheet),
            "四半期貸借対照表": self.format_yfinance_df(ticker_obj.quarterly_balance_sheet),
            "年次CF計算書": self.format_yfinance_df(ticker_obj.cashflow),
            "四半期CF計算書": self.format_yfinance_df(ticker_obj.quarterly_cashflow),
        }
        return statements

    def get_timeseries_financial_metrics(self, ticker_obj, info) -> pd.DataFrame:
        financials = ticker_obj.financials
        balance_sheet = ticker_obj.balance_sheet
        hist = ticker_obj.history(period="5y")
        dividends = ticker_obj.dividends
        if hasattr(hist.index.dtype, 'tz') and hist.index.dtype.tz is not None: hist.index = hist.index.tz_localize(None)
        if hasattr(dividends.index.dtype, 'tz') and dividends.index.dtype.tz is not None: dividends.index = dividends.index.tz_localize(None)
        if financials.empty:
            logger.warning(f"銘柄 {info.get('shortName', '')}: yfinanceの年次財務データ(financials)が見つかりません。")
            financials = pd.DataFrame(columns=[pd.Timestamp.now() - pd.DateOffset(years=i) for i in range(4)])
        if balance_sheet.empty:
            logger.warning(f"銘柄 {info.get('shortName', '')}: yfinanceの年次貸借対照表データ(balance_sheet)が見つかりません。")
            balance_sheet = pd.DataFrame(columns=financials.columns)
        equity_keys = ['Total Stockholder Equity', 'Stockholders Equity', 'Total Equity']
        assets_keys = ['Total Assets']
        shares_keys = ['Share Issued', 'Ordinary Shares Number', 'Basic Average Shares']
        revenue_keys = ['Total Revenue', 'Revenues', 'Total Sales']
        net_income_keys = ['Net Income', 'Net Income From Continuing Operations']
        eps_keys = ['Basic EPS']
        def find_yf_value(df, keys, col):
            if df.empty or col not in df.columns: return None
            for key in keys:
                if key in df.index: return df.loc[key, col]
            return None
        metrics = []
        annual_columns = financials.columns[:min(4, financials.shape[1])]
        logger.info(f"{info.get('shortName', '')}: {len(annual_columns)}期分の年次データを処理します。")
        for date_col in annual_columns:
            stockholder_equity = find_yf_value(balance_sheet, equity_keys, date_col)
            total_assets = find_yf_value(balance_sheet, assets_keys, date_col)
            net_income = find_yf_value(financials, net_income_keys, date_col)
            shares_outstanding = find_yf_value(balance_sheet, shares_keys, date_col)
            total_revenue = find_yf_value(financials, revenue_keys, date_col)
            price = hist.asof(date_col)['Close'] if not hist.empty else None
            eps = find_yf_value(financials, eps_keys, date_col)
            equity_ratio = (stockholder_equity / total_assets) * 100 if pd.notna(stockholder_equity) and pd.notna(total_assets) and total_assets > 0 else None
            annual_dividends = 0
            if not dividends.empty:
                dividends_in_year = dividends[dividends.index.year == date_col.year]
                if not dividends_in_year.empty: annual_dividends = dividends_in_year.sum()
            roe = (net_income / stockholder_equity) * 100 if pd.notna(net_income) and pd.notna(stockholder_equity) and stockholder_equity != 0 else None
            sps = total_revenue / shares_outstanding if pd.notna(total_revenue) and pd.notna(shares_outstanding) and shares_outstanding != 0 else None
            psr = price / sps if pd.notna(price) and pd.notna(sps) and sps != 0 else None
            per = price / eps if pd.notna(price) and pd.notna(eps) and eps != 0 else None
            bps = stockholder_equity / shares_outstanding if pd.notna(stockholder_equity) and pd.notna(shares_outstanding) and shares_outstanding != 0 else None
            pbr = price / bps if pd.notna(price) and pd.notna(bps) and bps != 0 else None
            div_yield = (annual_dividends / price) * 100 if pd.notna(price) and price > 0 else None
            metrics.append({
                '決算日': date_col.strftime('%Y-%m-%d'), '年度': f"{date_col.year}年度", 'EPS (円)': eps, 'PER (倍)': per, 'PBR (倍)': pbr,
                'PSR (倍)': psr, 'ROE (%)': roe, '自己資本比率 (%)': equity_ratio, '年間1株配当 (円)': annual_dividends, '配当利回り (%)': div_yield
            })
        latest_equity_ratio = None
        if not balance_sheet.empty and not balance_sheet.columns.empty:
            latest_bs_col_name = balance_sheet.columns[0]
            latest_equity = find_yf_value(balance_sheet, equity_keys, latest_bs_col_name)
            latest_assets = find_yf_value(balance_sheet, assets_keys, latest_bs_col_name)
            if pd.notna(latest_equity) and pd.notna(latest_assets) and latest_assets > 0:
                latest_equity_ratio = (latest_equity / latest_assets) * 100
        roe_info = info.get('returnOnEquity')
        latest_metrics = {
            '決算日': date.today().strftime('%Y-%m-%d'), '年度': '最新', 'EPS (円)': info.get('trailingEps'), 'PER (倍)': info.get('trailingPE'),
            'PBR (倍)': info.get('priceToBook'), 'PSR (倍)': info.get('priceToSalesTrailing12Months'), 'ROE (%)': roe_info * 100 if roe_info else None,
            '自己資本比率 (%)': latest_equity_ratio, '年間1株配当 (円)': info.get('trailingAnnualDividendRate'), '配当利回り (%)': info.get('trailingAnnualDividendYield') * 100 if info.get('trailingAnnualDividendYield') else None
        }
        metrics.append(latest_metrics)
        df = pd.DataFrame(metrics).set_index('決算日').sort_index(ascending=True)
        df['EPS成長率 (対前年比) (%)'] = df['EPS (円)'].pct_change(fill_method=None) * 100
        return df.sort_index(ascending=False)

    def calculate_peg_ratios(self, ticker_obj, info: dict) -> dict:
        results = {
            'cagr_growth': {'value': None, 'growth': None, 'reason': 'データ不足', 'eps_points': [], 'start_eps': None, 'end_eps': None, 'years': 0},
            'single_year': {'value': None, 'growth': None, 'reason': 'データ不足'},
            'historical_pegs': {},
            'warnings': []
        }
        try:
            current_per = info.get('trailingPE')
            if not current_per:
                for key in results:
                    if key not in ['historical_pegs', 'warnings']:
                        results[key]['reason'] = '現在のPERが取得できません'
                return results
            financials = ticker_obj.financials
            if financials.empty or 'Basic EPS' not in financials.index:
                for key in results:
                    if key not in ['historical_pegs', 'warnings']:
                        results[key]['reason'] = 'EPSデータが見つかりません'
                return results
            annual_eps_data = financials.loc['Basic EPS'].dropna().sort_index(ascending=False)
            if len(annual_eps_data) >= 2:
                latest_annual_eps, prev_annual_eps = annual_eps_data.iloc[0], annual_eps_data.iloc[1]
                if pd.notna(latest_annual_eps) and pd.notna(prev_annual_eps) and prev_annual_eps > 0:
                    growth = (latest_annual_eps - prev_annual_eps) / prev_annual_eps
                    results['single_year']['growth'] = growth
                    if growth > 0:
                        results['single_year']['value'] = current_per / (growth * 100)
                        results['single_year']['reason'] = None
                    else:
                        results['single_year']['reason'] = '単年成長率がマイナス'
                else:
                    results['single_year']['reason'] = 'EPSデータ欠損または前期がマイナス'
            trailing_eps = info.get('trailingEps')
            if trailing_eps is not None:
                points = [trailing_eps] + annual_eps_data.tolist()
                valid_points = [p for p in points if pd.notna(p)]
                results['cagr_growth']['eps_points'] = valid_points
                if len(valid_points) >= 2:
                    start_eps = valid_points[-1]
                    end_eps = valid_points[0]
                    years = len(valid_points) - 1
                    results['cagr_growth']['start_eps'] = start_eps
                    results['cagr_growth']['end_eps'] = end_eps
                    results['cagr_growth']['years'] = years
                    if start_eps < 0 and end_eps > 0:
                        eps_improvement = end_eps - start_eps
                        results['cagr_growth']['growth'] = float('inf')
                        results['cagr_growth']['reason'] = f"{years}年でEPSが{eps_improvement:+.2f}改善"
                        results['cagr_growth']['value'] = None
                        results['warnings'].append('注記: 赤字から黒字に転換したためPEGは計算できませんが、EPSの絶対額は改善しています。')
                    elif start_eps > 0 and end_eps > 0:
                        cagr = (end_eps / start_eps)**(1/years) - 1
                        results['cagr_growth']['growth'] = cagr
                        if cagr > 0:
                            results['cagr_growth']['value'] = current_per / (cagr * 100)
                            results['cagr_growth']['reason'] = f'{years}年間のCAGR'
                        else:
                            results['cagr_growth']['reason'] = f'{years}年CAGRがマイナス'
                    else:
                        results['cagr_growth']['reason'] = '開始/終了EPSがマイナスまたはゼロのため計算不能'
                        if start_eps <= 0:
                            results['warnings'].append('注記: 開始EPSがマイナスまたはゼロのため、CAGRベースのPEGは計算できません。')
                else:
                    results['cagr_growth']['reason'] = '有効なEPSが2地点未満'
            history = ticker_obj.history(period="6y")
            if not history.empty and len(annual_eps_data) >= 2:
                if hasattr(history.index.dtype, 'tz') and history.index.dtype.tz is not None:
                    history.index = history.index.tz_localize(None)
                for i in range(len(annual_eps_data) - 1):
                    eps_curr = annual_eps_data.iloc[i]
                    eps_prev = annual_eps_data.iloc[i+1]
                    year_date = annual_eps_data.index[i]
                    if pd.notna(eps_curr) and pd.notna(eps_prev) and eps_prev > 0:
                        yoy_growth = (eps_curr - eps_prev) / eps_prev
                        if yoy_growth > 0:
                            price_at_fis_year = history.asof(year_date)['Close'] if not history.empty and not history.asof(year_date).empty else None
                            if price_at_fis_year:
                                historical_per = price_at_fis_year / eps_curr
                                peg = historical_per / (yoy_growth * 100)
                                results['historical_pegs'][f"{year_date.year}年度"] = peg
        except Exception as e:
            logger.error(f"PEGレシオ計算中にエラー: {e}", exc_info=True)
        return results

    def _format_period(self, period_original: str) -> str:
        text = period_original.replace('/', '.')
        try:
            year, month = text.split('.')
            return f"{year}年{int(month)}月"
        except (ValueError, IndexError):
            return period_original

    def get_shareholder_data(self, soup: BeautifulSoup) -> pd.DataFrame:
        shareholders_data = []
        yearly_div = soup.find('div', id='holder-yearly')
        if not yearly_div or not (table := yearly_div.find('table', class_='history-table')):
            return pd.DataFrame()
        headers = [th.get_text(strip=True) for th in table.find('thead').find_all('th')][1:]
        if tbody := table.find('tbody'):
            for row in tbody.find_all('tr'):
                cells = row.find_all('td')
                if not cells: continue
                shareholder_name = cells[0].get_text(strip=True)
                for i, cell in enumerate(cells[1:]):
                    period = self._format_period(headers[i])
                    cell_text = cell.get_text("\n", strip=True)
                    if cell_text == "-": continue
                    shares_match = re.search(r'([\d,]+)千株', cell_text)
                    percent_match = re.search(r'([\d\.]+)%', cell_text)
                    shares = int(shares_match.group(1).replace(',', '')) * 1000 if shares_match else 0
                    percentage = float(percent_match.group(1)) if percent_match else 0.0
                    if shares > 0:
                        shareholders_data.append({'会計期': period, '株主名': shareholder_name, '保有株式数 (株)': shares, '保有割合 (%)': percentage})
        df = pd.DataFrame(shareholders_data)
        if not df.empty:
            df['順位'] = df.groupby('会計期')['保有割合 (%)'].rank(method='first', ascending=False).astype(int)
        return df

    def get_governance_data(self, soup: BeautifulSoup) -> pd.DataFrame:
        governance_data = []
        header = soup.find('h2', string='役員の状況')
        if not header: return pd.DataFrame()
        tab_container = header.find_next_sibling('div')
        if not tab_container: return pd.DataFrame()
        if tab_ul := tab_container.find('ul', class_='nav-tabs'):
            tabs = tab_ul.find_all('a')
            panes = tab_container.find('div', class_='tab-content').find_all('div', class_='tab-pane')
            for tab, pane in zip(tabs, panes):
                period = self._format_period(tab.get_text(strip=True))
                self._parse_officer_table(pane, period, governance_data)
        else:
            period = "最新"
            self._parse_officer_table(tab_container, period, governance_data)
        df = pd.DataFrame(governance_data)
        if not df.empty and '会計期' in df.columns and df['会計期'].str.contains('年').any():
            df['会計期_dt'] = pd.to_datetime(df['会計期'].str.replace('年', '-').str.replace('月', ''), format='%Y-%m', errors='coerce')
            df = df.sort_values(by='会計期_dt', ascending=False).drop(columns='会計期_dt')
        return df

    def _parse_officer_table(self, container, period, data_list):
        officer_table = container.find('table', class_='officer__history-table')
        if not officer_table or not (tbody := officer_table.find('tbody')): return
        for row in tbody.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) < 5: continue
            position = cols[0].get_text(strip=True)
            name_parts = cols[1].get_text(separator='|', strip=True).split('|')
            name = name_parts[0] if name_parts else ''
            birth_date = name_parts[1] if len(name_parts) > 1 else ''
            age = name_parts[2] if len(name_parts) > 2 else ''
            shares_text = cols[4].get_text(strip=True).replace(',', '')
            shares = int(shares_text) if shares_text.isdigit() else 0
            data_list.append({'会計期': period, '役職': position, '氏名': name, '生年月日': birth_date, '年齢': age, '役員としての所有株式数': shares})

    def get_shareholder_and_governance_data(self, ticker_code: str) -> dict:
        s_soup = self.get_html_soup(f"https://www.buffett-code.com/company/{ticker_code}/mainshareholder")
        g_soup = self.get_html_soup(f"https://www.buffett-code.com/company/{ticker_code}/governance")
        df_shareholders = self.get_shareholder_data(s_soup) if s_soup else pd.DataFrame()
        df_governance = self.get_governance_data(g_soup) if g_soup else pd.DataFrame()
        is_owner_executive = False
        if not df_governance.empty:
            df_governance['大株主としての保有株式数'] = 0
            df_governance['大株主としての保有割合 (%)'] = 0.0
        if not df_shareholders.empty and not df_governance.empty:
            df_shareholders['照合名'] = df_shareholders['株主名'].str.replace(' ', '').str.replace('　', '')
            if '会計期' in df_shareholders.columns and df_shareholders['会計期'].str.contains('年').any():
                df_shareholders['会計期_dt'] = pd.to_datetime(df_shareholders['会計期'].str.replace('年', '-').str.replace('月', ''), format='%Y-%m', errors='coerce')
                latest_shares = df_shareholders.sort_values('会計期_dt').drop_duplicates('照合名', keep='last')
            else:
                latest_shares = df_shareholders.drop_duplicates('照合名', keep='last')
            shareholder_map = latest_shares.set_index('照合名')[['保有株式数 (株)', '保有割合 (%)']].apply(tuple, axis=1).to_dict()
            for index, row in df_governance.iterrows():
                governance_name_normalized = row['氏名'].replace(' ', '').replace('　', '')
                if governance_name_normalized in shareholder_map:
                    share_count, percentage = shareholder_map[governance_name_normalized]
                    df_governance.loc[index, '大株主としての保有株式数'] = share_count
                    df_governance.loc[index, '大株主としての保有割合 (%)'] = percentage
                    is_owner_executive = True
        return {"shareholders_df": df_shareholders, "governance_df": df_governance, "is_owner_executive": is_owner_executive}

    def perform_full_analysis(self, ticker_code: str, options: dict) -> dict:
        result = {'ticker_code': ticker_code, 'warnings': [], 'buffett_code_data': {}, 'timeseries_df': pd.DataFrame()}
        try:
            logger.info(f"--- 銘柄 {ticker_code} の分析を開始 ---")
            if self.session is None:
                raise ValueError("セッションが正常に初期化されませんでした。")
            info = None
            ticker_obj = None
            for attempt in range(3):
                try:
                    ticker_obj = yf.Ticker(f"{ticker_code}.T")
                    info = ticker_obj.info
                    if info and info.get('quoteType') is not None:
                        logger.info(f"銘柄 {ticker_code} の情報取得に成功しました。 ({attempt + 1}回目)")
                        break
                except Exception as e:
                    logger.warning(f"銘柄 {ticker_code} の情報取得に失敗 ({attempt + 1}/3回目): {e}")
                    if attempt < 2: time.sleep(5)
            if not info or info.get('quoteType') is None:
                raise ValueError("yfinanceから有効な情報を取得できませんでした。(3回試行後)")
            company_name = info.get('shortName') or info.get('longName') or f"銘柄 {ticker_code}"
            result['company_name'] = company_name
            result['yf_info'] = info
            result['is_ipo_within_5_years'] = False
            listing_date_str = self.get_listing_date(ticker_code)
            if listing_date_str:
                try:
                    listing_date = datetime.strptime(listing_date_str, '%Y年%m月%d日')
                    if (datetime.now() - listing_date) < pd.Timedelta(days=365.25 * 5):
                        result['is_ipo_within_5_years'] = True
                        logger.info(f"銘柄 {ticker_code} は上場5年以内の銘柄です。")
                except ValueError as e:
                    logger.warning(f"上場年月日 '{listing_date_str}' の日付形式の解析に失敗しました: {e}")
            if info.get('trailingPE') is None or info.get('trailingPE') <= 0:
                logger.info(f"銘柄 {ticker_code}: yfinanceのtrailingPEが不適切なため、代替PERの計算を試みます。")
                per_result = self._get_alternative_per(ticker_obj, info)
                if per_result['value'] is not None:
                    info['trailingPE'] = per_result['value']
                    logger.info(f"代替PER ({per_result['value']:.2f}) を採用しました。")
                    if per_result['note']:
                        result['warnings'].append(per_result['note'])
                else:
                    logger.warning(f"銘柄 {ticker_code}: 代替PERの計算に失敗しました。")
            for statement, path in {"貸借対照表": "bs", "損益計算書": "pl"}.items():
                url = f"https://www.buffett-code.com/company/{ticker_code}/financial/{path}"
                soup = self.get_html_soup(url)
                if soup:
                    all_data = self.extract_all_financial_data(soup)
                    if all_data:
                        result['buffett_code_data'][statement] = all_data
                    else:
                        logger.warning(f"Buffett-Codeから{statement}のデータ解析に失敗。")
                        result['buffett_code_data'][statement] = {}
                        raise ValueError(f"バフェットコードからの{statement}データ取得・解析に失敗しました。")
                else:
                    raise ValueError(f"バフェットコード({url})へのアクセスに失敗しました。")
            yf_data_for_calc = {**info, **options}
            result['scoring_indicators'] = self._calculate_scoring_indicators(result['buffett_code_data'], yf_data_for_calc)
            result['warnings'].extend(result['scoring_indicators'].pop('calc_warnings', []))
            peg_results = self.calculate_peg_ratios(ticker_obj, info)
            result['peg_analysis'] = peg_results
            if peg_results.get('warnings'):
                result['warnings'].extend(peg_results['warnings'])
            cagr_peg_value = peg_results['cagr_growth']['value']
            peg_score_dict = self._calculate_peg_score(cagr_peg_value)
            result['scoring_indicators']['peg'] = {'value': cagr_peg_value, 'reason': peg_results['cagr_growth']['reason'], **peg_score_dict}
            s_safety = result['scoring_indicators']['net_cash_ratio']['score']
            s_value = result['scoring_indicators']['cn_per']['score']
            s_quality = result['scoring_indicators']['roic']['score']
            s_growth = result['scoring_indicators']['peg']['score']
            result['strategy_scores'] = {}
            for name, weights in STRATEGY_WEIGHTS.items():
                weighted_score = (
                    s_safety * weights['safety'] +
                    s_value * weights['value'] +
                    s_quality * weights['quality'] +
                    s_growth * weights['growth']
                )
                result['strategy_scores'][name] = weighted_score
            result['final_average_score'] = result['strategy_scores']['⚖️ バランス型（バランス）']
            ts_df = self.get_timeseries_financial_metrics(ticker_obj, info)
            if not ts_df.empty:
                peg_col_name = 'PEG (実績)'
                peg_df = pd.DataFrame(peg_results['historical_pegs'].items(), columns=['年度', peg_col_name])
                ts_df = ts_df.reset_index().merge(peg_df, on='年度', how='left').set_index('決算日')
                latest_index = ts_df[ts_df['年度'] == '最新'].index
                if not latest_index.empty:
                    ts_df.loc[latest_index, peg_col_name] = peg_results['single_year']['value']
            result['timeseries_df'] = ts_df
            result['yfinance_statements'] = self.get_yfinance_statements(ticker_obj)
            try:
                shareholder_data = self.get_shareholder_and_governance_data(ticker_code)
                result.update(shareholder_data)
                logger.info(f"銘柄 {ticker_code} の大株主・役員情報の取得に成功。")
            except Exception as e:
                logger.error(f"銘柄 {ticker_code} の大株主・役員情報の取得中にエラー: {e}", exc_info=True)
                result['warnings'].append("大株主・役員情報の取得に失敗しました。")
                result['shareholders_df'] = pd.DataFrame()
                result['governance_df'] = pd.DataFrame()
                result['is_owner_executive'] = False
        except Exception as e:
            logger.error(f"銘柄 {ticker_code} の分析中にエラーが発生しました: {e}", exc_info=True)
            result['error'] = f"分析中にエラーが発生しました: {e}"
            if 'company_name' not in result:
                result['company_name'] = f"銘柄 {ticker_code} (エラー)"
        return result

# ==============================================================================
# 分析実行関数
# ==============================================================================
def run_stock_analysis(ticker_input_str: str, options: dict):
    """
    指定された銘柄リスト文字列に基づいて分析を実行し、結果を返す。
    """
    input_queries = [q.strip() for q in ticker_input_str.split(',') if q.strip()]
    if not input_queries:
        st.error("銘柄コードまたは会社名を入力してください。")
        return None

    search_handler = st.session_state.data_handler
    target_stocks = []
    not_found_queries = []
    with st.spinner("銘柄を検索しています..."):
        for query in input_queries:
            stock_info = search_handler.get_ticker_info_from_query(query)
            if stock_info:
                target_stocks.append(stock_info)
            else:
                not_found_queries.append(query)

    unique_target_stocks = list({stock['code']: stock for stock in target_stocks}.values())

    if not_found_queries:
        st.warning(f"以下の銘柄は見つかりませんでした: {', '.join(not_found_queries)}")

    if not unique_target_stocks:
        st.error("分析対象の銘柄が見つかりませんでした。")
        return None

    st.success(f"分析対象: {', '.join([s['code'] for s in unique_target_stocks])}")
    progress_bar = st.progress(0)
    progress_text = st.empty()
    all_results = {}
    data_handler = st.session_state.data_handler
    total_stocks = len(unique_target_stocks)

    data_handler._reset_session()
    for i, stock_info in enumerate(unique_target_stocks):
        if i > 0 and (i % 4 == 0 or data_handler.session is None):
            logger.info(f"定期的なセッションのリセットを実行 ({i}銘柄目)")
            data_handler._reset_session()

        progress_text.text(f"分析中... ({i+1}/{total_stocks}件完了): {stock_info.get('name', '')} ({stock_info['code']})")

        if data_handler.session is None:
            logger.error(f"銘柄 {stock_info['code']} のセッション初期化に失敗。スキップします。")
            display_key = f"{stock_info.get('name', stock_info['code'])} ({stock_info['code']})"
            all_results[display_key] = {
                'error': 'データ取得セッションの初期化に失敗しました。サイトがメンテナンス中か、ネットワークに問題がある可能性があります。',
                'company_name': stock_info.get('name', stock_info['code']),
                'ticker_code': stock_info['code']
            }
            progress_bar.progress((i + 1) / total_stocks)
            continue

        code = stock_info['code']
        result = data_handler.perform_full_analysis(code, options)
        result['sector'] = stock_info.get('sector', '業種不明')
        # ★追加: 業種コードも結果に含める
        result['sector_code'] = stock_info.get('sector_code')
        display_key = f"{result.get('company_name', code)} ({code})"
        all_results[display_key] = result
        progress_bar.progress((i + 1) / total_stocks)

    progress_text.empty()
    progress_bar.empty()
    return all_results

# ==============================================================================
# GUI表示用ヘルパー関数
# ==============================================================================
def get_recommendation(score):
    if score is None: return "---", "評価不能"
    if score >= 90: return "★★★★★", "神レベル"
    if score >= 80: return "★★★★☆", "非常に推奨"
    if score >= 70: return "★★★☆☆", "良い投資候補"
    if score >= 50: return "★★☆☆☆", "検討の価値あり"
    if score >= 30: return "★☆☆☆☆", "注意深い分析が必要"
    return "☆☆☆☆☆", "推奨しない"

def get_peg_investor_commentary(peg_value: float | None) -> str:
    if peg_value is None or peg_value < 0: return "評価不能：PEGレシオが計算できないか、成長率がマイナスです。"
    if peg_value < 0.5: return "💎 **超割安**<br><br>ピーター・リンチ： 「PERが成長率の半分であれば、それは非常に有望な掘り出し物だ。」まさに彼の理想であり、大きな利益をもたらす可能性を秘めた「お宝銘柄」と言えるでしょう。"
    if 0.5 <= peg_value < 1: return "✅ **割安**<br><br>ジム・クレイマー： 「我々が探しているのはこれだ！」と叫ぶ水準です。<br>ピーター・リンチ： この領域にある株を「バーゲン価格である可能性を秘めている」と評価します。両氏が最も好む、魅力的な投資領域です。"
    if peg_value == 1: return "⚖️ **適正**<br><br>ピーター・リンチ： 「公正な価格がついている企業のPERは、その成長率に等しい。」これが彼の定義した「適正価格」の基準点。ここから割安か割高かを判断します。"
    if 1 < peg_value < 2: return "🤔 **割高傾向**<br><br>ジム・クレイマー： 「2未満であれば許容できる」と語る、彼の柔軟性が表れる領域。リンチ氏なら慎重になりますが、クレイマー氏は素晴らしい企業であれば、この程度のプレミアムは許容範囲だと考えます。"
    if peg_value >= 2: return "❌ **割高**<br><br>ピーター・リンチ： 「PERが成長率の2倍であれば非常に危険だ」と警告します。<br>ジム・クレイマー： 「どんなにその会社が好きでも高すぎる（too rich for our blood）」と一蹴する水準。両氏が「手を出すべきではない」と口を揃える危険水域です。"
    return "評価不能"

def get_kiyohara_commentary(net_cash_ratio, cn_per, net_income):
    nc_comment = "### 清原式ネットキャッシュ比率の評価\n\n"
    if net_cash_ratio is None: nc_comment += "評価不能 (データ不足)"
    elif net_cash_ratio >= 1.0: nc_comment += "【超安全圏・鉄壁】🔵 企業の時価総額を上回る実質的な現金を保有する最高レベル。 理論上は、会社を丸ごと買収してもお釣りがくる計算になります。倒産リスクは極めて低く、下値不安が非常に小さい「鉄壁の財務」と言えます。清原氏がスクリーニングの対象としたのも、この100%を超えるような超割安銘柄でした。"
    elif net_cash_ratio >= 0.8: nc_comment += "【極めて安全】🟢 時価総額の大部分がネットキャッシュで裏付けられており、財務基盤は盤石です。株価が企業の実質的な価値に対して大幅に割安である可能性が非常に高い水準です。M&Aや大規模な株主還元（自社株買い、増配）のポテンシャルも秘めています。"
    elif net_cash_ratio >= 0.5: nc_comment += "【非常に安全・割安】🟢 時価総額の半分以上をネットキャッシュが占める、非常に安全な水準。 不況や予期せぬ経営環境の変化に対する耐性が極めて高く、安心して長期保有を検討できる財務内容です。多くの優良なバリュー株がこの領域に含まれる可能性があります。"
    elif net_cash_ratio >= 0.2: nc_comment += "【安全圏】🟡 十分に厚いネットキャッシュを保有しており、財務的な安定感があります。 一般的な基準で見れば、十分に財務健全性が高いと言えるレベルです。この水準でも、割安と判断できる銘柄は多く存在します。"
    elif net_cash_ratio >= 0.1: nc_comment += "【やや注意】🟠 一定のネットキャッシュはありますが、上記の水準と比較すると財務的な余裕は少なくなってきます。有利子負債の額や、本業でのキャッシュフロー創出力など、他の財務指標と合わせて慎重に評価する必要があります。"
    elif net_cash_ratio >= 0.01: nc_comment += "【要注意】🔴 ネットキャッシュがほとんどない状態です。すぐに危険というわけではありませんが、財務的なバッファーは小さいと言えます。特に、有利子負債の多い企業は、金利の上昇局面に注意が必要です。成長のための先行投資で一時的にこの水準になっている可能性もあります。"
    else: nc_comment += "【要警戒】🚨 実質的な現金よりも有利子負債が多い「ネットデット（純負債）」の状態。 清原氏のようなバリュー投資家が好む財務状況とは言えません。ただし、金融機関や、成長のために財務レバレッジを積極的に活用する企業（不動産業、IT関連など）では一般的です。事業内容や成長性を精査し、負債をコントロールできているかを厳しく見極める必要があります。"
    cn_per_comment = "\n\n<br><br>\n\n### キャッシュニュートラルPERの評価\n\n"
    if net_income is not None and net_income <= 0:
        if cn_per is not None and cn_per < 0:
            cn_per_comment += "【要注意株】🧐 「価値の罠」の可能性あり。事業が利益を生み出せていない赤字状態。どれだけ資産を持っていても、事業活動でそれを食いつぶしているかもしれません。赤字が一時的なものか、構造的なものか、その原因を詳しく調べる必要があります。"
        else:
            cn_per_comment += "【赤字企業・分析注意】事業が利益を生み出せていない赤字状態です。財務健全性（ネットキャッシュ比率）は重要ですが、事業そのものの将来性を慎重に評価する必要があります。"
    elif cn_per is None:
        cn_per_comment += "評価不能 (PER等のデータ不足のため計算不可)"
    elif cn_per < 0:
        cn_per_comment += "【究極の割安株】🤑 お宝株の可能性大。事業価値がマイナス（時価総額 < ネットキャッシュ）なのに利益は出ている状態。なぜ市場がこれほどまでに評価していないのか、隠れたリスクがないかを精査する価値が非常に高いです。"
    elif 0 <= cn_per < 2:
        cn_per_comment += "【現金より安い会社】🤯 💎\n\n> 「時価総額からネットキャッシュを引いた事業価値が、純利益の1～2年分しかないということ。これはもう、『ほぼタダ』で会社が手に入るのに等しい。なぜ市場がここまで見捨てているのか、何か特別な悪材料がないか徹底的に調べる必要があるが、そうでなければ『ありえない安値』だ。こういう会社は、誰かがその価値に気づけば、株価は簡単に2倍、3倍になる可能性を秘めている」\n\n**評価:** 最大限の買い評価。ただし、異常な安さの裏に隠れたリスク（訴訟、偶発債務など）がないかは慎重に確認する、というスタンスでしょう 🤔。"
    elif 2 <= cn_per < 4:
        cn_per_comment += "【私の投資のど真ん中】🎯 💪\n\n> 「実質PERが4倍以下。これが私の投資のど真ん中だ。 この水準であれば、多少の成長性の鈍化や業績のブレなど意に介さない。事業価値がこれだけ安ければ、下値リスクは限定的。市場参加者の多くがその価値に気づいていないだけで、放っておけばいずれ評価される。こういう銘柄こそ、安心して大きな金額を投じられる」\n\n**評価:** 最も信頼を置き、積極的に投資対象とする「コア・ゾーン」です。彼の投資術の神髄がこの価格帯にあると言えます ✅。"
    elif 4 <= cn_per < 7:
        cn_per_comment += "【まあ、悪くない水準】👍 🙂\n\n> 「実質PERが5倍、6倍ね…。まあ、悪くない水準だ。普通のバリュー投資家なら喜んで買うだろう。ただ、私に言わせれば、ここから先は『普通に安い会社』であって、驚くほどの安さではない。他に買うべきものがなければ検討するが、胸を張って『これは買いだ』と断言するには少し物足りなさを感じる」\n\n**評価:** 許容範囲ではあるものの、最高の投資対象とは見なしません。より割安な銘柄があれば、そちらを優先するでしょう。"
    elif 7 <= cn_per < 10:
        cn_per_comment += "【普通の会社】😐 📈\n\n> 「実質PERが10倍近くになってくると、もはや割安とは言えない。『普通の会社』の値段だ。 この水準の株を買うのであれば、ネットキャッシュの価値だけでは不十分で、将来の成長性がどれだけあるかという議論が不可欠になる。しかし、私にはその未来を正確に予測する能力はない」\n\n**評価:** 彼の得意とする「資産価値」を拠り所とした投資スタイルからは外れ始めます。成長性の評価という不確実な領域に入るため、投資対象としての魅力は大きく薄れます 🤷‍♂️。"
    elif 10 <= cn_per < 15:
        cn_per_comment += "【私には割高に思える】🤨 👎\n\n> 「多くの市場参加者が『適正水準だ』と言うかもしれないが、私にはもう割高に思える。ネットキャッシュを差し引いた事業価値ですら、利益の10年以上分を払うということ。それだけの価値があるというなら、よほど素晴らしい成長ストーリーと、それを実現できる経営陣が必要になる。私には博打にしか見えない 🎲」\n\n**評価:** 明確に「割高」と判断し、通常は投資対象としません。"
    else:
        cn_per_comment += "【論外。バブル以外の何物でもない】❌ 🤮\n\n> 「実質PERが20倍だの30倍だのというのは、はっきり言って論外だ。 どれだけ輝かしい未来を語られようと、それは単なる夢物語。株価は期待だけで形成されている。こういう会社がその後どうなるか、私は何度も見てきた。これは投資ではなく投機であり、バブル以外の何物でもない 💥。アナリストが全員で強気な推薦をしていたら、むしろ空売りを検討するくらいだ」\n\n**評価:** 投資対象として全く考えない水準です。むしろ市場の過熱を示すサインと捉え、警戒を強めるでしょう。"
    return nc_comment + cn_per_comment

# ==============================================================================
# --- Streamlit App Main ---
# ==============================================================================
st.set_page_config(page_title="統合型 企業価値分析ツール", layout="wide")

# --- セッションステートの初期化 ---
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = IntegratedDataHandler()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'rf_rate' not in st.session_state:
    st.session_state.rf_rate = 0.01
if 'rf_rate_manual' not in st.session_state:
    st.session_state.rf_rate_manual = st.session_state.rf_rate
if 'rf_rate_fetched' not in st.session_state:
    st.session_state.rf_rate_fetched = False
# ★修正1: テキストエリアのデフォルト値を空にする
if 'ticker_input_value' not in st.session_state:
    st.session_state.ticker_input_value = ""


# --- サイドバーUI ---
st.sidebar.title("分析設定")

# --- シンプル検索セクション ---
st.sidebar.subheader("銘柄検索（シンプル検索）")
ticker_input = st.sidebar.text_area(
    "銘柄コード or 会社名 (カンマ区切り)",
    value=st.session_state.ticker_input_value,
    key="ticker_input_widget"
)
analyze_button = st.sidebar.button("分析実行")

st.sidebar.markdown("---")

# --- AI類似銘柄検索セクション ---
st.sidebar.subheader("AI類似銘柄検索")
ai_search_query = st.sidebar.text_input(
    "対象企業 (コード or 会社名):",
    placeholder="例: 7203 or トヨタ自動車",
    key="ai_search_input"
)
ai_search_button = st.sidebar.button("類似銘柄検索")

st.sidebar.markdown("---")

# --- 詳細設定セクション ---
st.sidebar.subheader("詳細設定")
if not st.session_state.rf_rate_fetched:
    with st.spinner("最新のリスクフリーレートを取得中..."):
        rate = st.session_state.data_handler.get_risk_free_rate()
        if rate is not None:
            st.session_state.rf_rate = rate
            st.session_state.rf_rate_manual = rate
            st.success(f"リスクフリーレートを自動取得しました: {rate:.4f}")
    st.session_state.rf_rate_fetched = True

st.session_state.rf_rate_manual = st.sidebar.number_input(
    "リスクフリーレート(Rf)", value=st.session_state.rf_rate_manual, format="%.4f",
    help="日本の10年国債利回りを基準とします。自動取得に失敗した場合は手動で入力してください。"
)
st.session_state.rf_rate = st.session_state.rf_rate_manual
mrp = st.sidebar.number_input("マーケットリスクプレミアム(MRP)", value=0.06, format="%.2f")

# --- メイン画面 ---
st.title("統合型 企業価値分析ツール")
st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- メイン処理 ---
options = {'risk_free_rate': st.session_state.rf_rate, 'mkt_risk_premium': mrp}

# AI類似銘柄検索ボタンが押された時の処理
if ai_search_button:
    if not ai_search_query:
        st.sidebar.error("対象企業を入力してください。")
    else:
        search_handler = st.session_state.data_handler
        stock_info = search_handler.get_ticker_info_from_query(ai_search_query)

        if stock_info is None:
            st.sidebar.error(f"「{ai_search_query}」に一致する銘柄が見つかりませんでした。")
        else:
            target_code = stock_info['code']
            target_name = stock_info['name']
            target_sector = stock_info.get('sector')
            
            candidate_list_str = None
            status_message = f"企業「{target_name} ({target_code})」の類似銘柄をAIが検索中..."

            # 事前フィルタリング
            if target_sector and pd.notna(target_sector) and not search_handler.stock_list_df.empty:
                status_message = f"「{target_sector}」業種内で類似銘柄をAIが検索中..."
                candidate_df = search_handler.stock_list_df[
                    (search_handler.stock_list_df['sector'] == target_sector) &
                    (search_handler.stock_list_df['code'] != target_code)
                ]
                if not candidate_df.empty:
                    # プロンプト用に候補リストを作成（最大100件）
                    candidate_list = [f"- {row['name']} ({row['code']})" for index, row in candidate_df.head(100).iterrows()]
                    candidate_list_str = "\n".join(candidate_list)
            
            similar_tickers = ""
            with st.status(status_message, expanded=True) as status:
                try:
                    st.write("🧠 AIモデルを準備しています...")
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    time.sleep(1)

                    st.write(f"📝 {target_name} ({target_code})用の高度なプロンプトを生成しています...")
                    prompt = generate_prompt(target_code, candidate_list_str)
                    time.sleep(1)

                    st.write("⏳ AIが類似銘柄を分析中です... (これには数十秒かかる場合があります)")
                    response = model.generate_content(prompt)

                    st.write("⚙️ 応答データを整形しています...")
                    # ★修正: 数字、カンマ、大文字アルファベット以外を削除
                    cleaned_text = re.sub(r'[^0-9,A-Z]', '', response.text.upper())
                    similar_tickers = ",".join(filter(None, cleaned_text.split(',')))
                    time.sleep(1)

                    if similar_tickers:
                        status.update(label="✅ AI検索完了！", state="complete", expanded=False)
                    else:
                        status.update(label="⚠️ AIが類似銘柄を見つけられませんでした。", state="error", expanded=True)
                        st.warning("AIから類似銘柄を取得できませんでした。")

                except Exception as e:
                    status.update(label="❌ エラー発生", state="error", expanded=True)
                    st.error(f"AI検索中にエラーが発生しました: {e}")

            if similar_tickers:
                # ★修正2: 検索元の銘柄をリストの先頭に追加
                final_ticker_list = f"{target_code},{similar_tickers}"
                st.session_state.ticker_input_value = final_ticker_list
                st.success(f"AIが抽出した銘柄リストで分析を開始します: {final_ticker_list}")
                time.sleep(1)
                results = run_stock_analysis(final_ticker_list, options)
                if results:
                    st.session_state.results = results
                    st.rerun()

# 分析実行ボタンが押された時の処理
if analyze_button:
    st.session_state.ticker_input_value = ticker_input
    results = run_stock_analysis(ticker_input, options)
    if results:
        st.session_state.results = results

# --- 結果表示 ---
if st.session_state.results:
    all_results = st.session_state.results
    st.header("個別銘柄サマリー")
    strategy_options = list(STRATEGY_WEIGHTS.keys())
    selected_strategy = st.radio("表示戦略の切り替え:", strategy_options, horizontal=True, key='result_view_strategy')
    sorted_results = sorted(all_results.items(), key=lambda item: item[1].get('strategy_scores', {}).get(selected_strategy, -1), reverse=True)

    for display_key, result in sorted_results:
        ticker_code = result.get('ticker_code')
        if 'error' in result:
            with st.expander(f"▼ {display_key} - 分析エラー", expanded=True):
                st.error(f"分析中にエラーが発生しました。\n\n詳細: {result['error']}")
            continue

        score = result.get('strategy_scores', {}).get(selected_strategy)
        stars_text, _ = get_recommendation(score)
        score_color = "#28a745" if score is not None and score >= 70 else "#ffc107" if score is not None and score >= 40 else "#dc3545"
        score_text = f"{score:.1f}" if score is not None else "N/A"

        st.markdown(f"<hr style='border: 2px solid {score_color};'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.55, 0.3, 0.15])
        with col1:
            market_cap = result.get('yf_info', {}).get('marketCap')
            is_ipo_within_5_years = result.get('is_ipo_within_5_years', False)
            ipo_badge = f"<span style='display:inline-block; vertical-align:middle; padding:3px 8px; font-size:13px; font-weight:bold; color:white; background-color:#dc3545; border-radius:12px; margin-left:10px;'>上場5年以内</span>" if is_ipo_within_5_years else ""
            small_cap_badge = ""
            if market_cap and market_cap <= 10_000_000_000:
                small_cap_badge = f"<span style='display:inline-block; vertical-align:middle; padding:3px 8px; font-size:13px; font-weight:bold; color:white; background-color:#007bff; border-radius:12px; margin-left:10px;'>小型株</span>"
            
            # ★追加: シクリカル銘柄バッジの生成
            cyclical_badge = ""
            sector_code = result.get('sector_code')
            if sector_code and sector_code in CYCLICAL_SECTOR_CODES:
                 cyclical_badge = f"<span style='display:inline-block; vertical-align:middle; padding:3px 8px; font-size:13px; font-weight:bold; color:white; background-color:#6f42c1; border-radius:12px; margin-left:10px;'>シクリカル銘柄</span>"

            kabutan_link = ""
            if ticker_code:
                kabutan_url = f"https://kabutan.jp/stock/?code={ticker_code}"
                kabutan_link = f"<a href='{kabutan_url}' target='_blank' title='株探で株価を確認' style='text-decoration:none; margin-left:10px; font-size:20px; vertical-align:middle;'>🔗</a>"
            is_owner_exec = result.get('is_owner_executive', False)
            owner_badge = f"<span style='display:inline-block; vertical-align:middle; padding:3px 8px; font-size:13px; font-weight:bold; color:white; background-color:#28a745; border-radius:12px; margin-left:10px;'>大株主役員</span>" if is_owner_exec else ""
            sector = result.get('sector', '')
            sector_span = f"<span style='font-size:16px; color:grey; font-weight:normal; margin-left:10px;'>({sector})</span>" if sector and pd.notna(sector) else ""
            
            # ★修正: バッジ表示を追加
            st.markdown(f"### {display_key} {kabutan_link} {ipo_badge} {small_cap_badge} {owner_badge} {cyclical_badge} {sector_span}", unsafe_allow_html=True)
        with col2:
            info = result.get('yf_info', {})
            price, change, prev_close = info.get('regularMarketPrice'), info.get('regularMarketChange'), info.get('regularMarketPreviousClose')
            change_pct = (price - prev_close) / prev_close if all(isinstance(x, (int, float)) for x in [price, prev_close]) and prev_close > 0 else info.get('regularMarketChangePercent')
            if all(isinstance(x, (int, float)) for x in [price, change, change_pct]):
                st.metric(label="現在株価", value=f"{price:,.0f} 円", delta=f"前日比 {change:+.2f}円 ({change_pct:+.2%})")
        with col3:
            st.write("")
            st.write("")
            indicators = result.get('scoring_indicators', {})
            def format_for_copy(data):
                val = data.get('value')
                return f"{val:.2f} ({data.get('evaluation', '')})" if val is not None else "N/A"
            change_pct_text = f"({change_pct:+.2%})" if isinstance(change_pct, (int, float)) else ""
            price_text = f"株価: {price:,.0f}円 (前日比 {change:+.2f}円, {change_pct_text})" if all(isinstance(x, (int, float)) for x in [price, change]) else ""
            market_cap_val = result.get('yf_info', {}).get('marketCap')
            market_cap_text = ""
            if market_cap_val:
                if market_cap_val >= 1_000_000_000_000:
                    market_cap_text = f"時価総額: {market_cap_val / 1_000_000_000_000:,.2f} 兆円"
                else:
                    market_cap_text = f"時価総額: {market_cap_val / 100_000_000:,.2f} 億円"
            features = []
            if market_cap_val and market_cap_val <= 10_000_000_000:
                features.append("小型株")
            if result.get('is_owner_executive', False):
                features.append("大株主役員")
            if result.get('is_ipo_within_5_years', False):
                features.append("上場5年以内")
            if sector_code and sector_code in CYCLICAL_SECTOR_CODES:
                features.append("シクリカル銘柄")

            features_text = f"特徴: {', '.join(features)}" if features else ""
            owner_info_text = ""
            df_g = result.get('governance_df')
            if result.get('is_owner_executive', False) and df_g is not None and not df_g.empty and '大株主としての保有割合 (%)' in df_g.columns:
                owners = df_g[df_g['大株主としての保有割合 (%)'] > 0]
                if not owners.empty:
                    top_owner = owners.loc[owners['大株主としての保有割合 (%)'].idxmax()]
                    owner_name = top_owner.get('氏名', '不明')
                    owner_ratio = top_owner.get('大株主としての保有割合 (%)', 0)
                    owner_info_text = f"筆頭オーナー経営者: {owner_name} ({owner_ratio:.2f}%)"
            copy_text = f"■ {display_key}\n{price_text}"
            if market_cap_text: copy_text += f"\n{market_cap_text}"
            if features_text: copy_text += f"\n{features_text}"
            if owner_info_text: copy_text += f"\n{owner_info_text}"
            copy_text += (f"\n\n総合スコア ({selected_strategy}): {score_text}点 {stars_text}\n"
                          f"--------------------\nPEGレシオ (CAGR): {format_for_copy(indicators.get('peg',{}))}\n"
                          f"ネットキャッシュ比率: {format_for_copy(indicators.get('net_cash_ratio',{}))}\n"
                          f"キャッシュニュートラルPER: {format_for_copy(indicators.get('cn_per',{}))}\n"
                          f"ROIC: {format_for_copy(indicators.get('roic',{}))}")
            create_copy_button(copy_text, "📋 結果をコピー", key=f"copy_{display_key.replace(' ','_')}")
        st.markdown(f"#### 総合スコア ({selected_strategy}): <span style='font-size:28px; font-weight:bold; color:{score_color};'>{score_text}点</span> <span style='font-size:32px;'>{stars_text}</span>", unsafe_allow_html=True)
        if result.get('warnings'): st.info(f"{'; '.join(list(set(result.get('warnings',[]))))}。")
        with st.container():
            cols = st.columns(4)
            def show_metric(col, title, subtitle, data, warnings):
                with col:
                    note = ""
                    if title == "PEGレシオ (CAGR)" and any("PEG" in w for w in warnings): note = " *"
                    if title in ["ネットキャッシュ比率", "キャッシュニュートラルPER"] and any(k in w for w in warnings for k in ["NC比率", "負債", "有価券"]): note = " *"
                    if title == "ROIC" and any("ROIC" in w for w in warnings): note = " *"
                    val, score = data.get('value'), data.get('score', 0)
                    val_str = f"{val:.2f}" if val is not None else "N/A"
                    color = "#28a745" if score >= 70 else "#ffc107" if score >= 40 else "#dc3545"
                    st.markdown(f"<div style='text-align:center;'><p style='font-size:14px; color:#555; font-weight:bold; margin-bottom:0;'>{title}{note}</p><p style='font-size:11px; color:#777; margin-bottom:5px; margin-top:-2px;'>{subtitle}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:28px; color:{color}; font-weight:bold; text-align:center; margin:0;'>{val_str}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align:center; font-weight:bold; font-size:14px;'>スコア: <span style='color:{color};'>{score:.1f}点</span></p>", unsafe_allow_html=True)
                    if val is None: st.markdown(f"<p style='text-align:center; font-size:12px; color:red;'>({data.get('reason', '計算不能')})</p>", unsafe_allow_html=True)
                    else: st.markdown(f"<p style='text-align:center; font-size:12px; color:#777;'>{data.get('evaluation', '---')}</p>", unsafe_allow_html=True)
            show_metric(cols[0], "PEGレシオ (CAGR)", "成長性を考慮した株価の割安性", indicators.get('peg', {}), result.get('warnings', []))
            show_metric(cols[1], "ネットキャッシュ比率", "現金を考慮した割安度", indicators.get('net_cash_ratio', {}), result.get('warnings', []))
            show_metric(cols[2], "キャッシュニュートラルPER", "事業価値の割安度", indicators.get('cn_per', {}), result.get('warnings', []))
            show_metric(cols[3], "ROIC", "収益性・資本効率", indicators.get('roic', {}), result.get('warnings', []))
            with st.expander("詳細データを見る"):
                tab_titles = [
                    "時系列指標", "大株主・役員", "PEGレシオ (CAGR) 計算", "ネットキャッシュ比率計算", "キャッシュニュートラルPER計算", "ROIC計算", "WACC計算",
                    "PEGレシオコメント", "専門家コメント", "財務諸表(バフェットコード)", "ヤフーファイナンス財務"
                ]
                tabs = st.tabs(tab_titles)
                with tabs[0]:
                    ts_df = result.get('timeseries_df')
                    if ts_df is not None and not ts_df.empty:
                        display_columns = ['年度', 'EPS (円)', 'EPS成長率 (対前年比) (%)', 'PER (倍)', 'PBR (倍)', 'PEG (実績)', 'PSR (倍)', 'ROE (%)', '自己資本比率 (%)', '年間1株配当 (円)', '配当利回り (%)']
                        df_to_display = ts_df.copy().reset_index()
                        existing_cols = [col for col in display_columns if col in df_to_display.columns]
                        df_to_display = df_to_display[['決算日'] + existing_cols]
                        numeric_cols = {col: "{:.2f}" for col in df_to_display.select_dtypes(include=np.number).columns}
                        st.dataframe(df_to_display.style.format(numeric_cols, na_rep="-"))
                    else:
                        st.warning("時系列データを取得できませんでした。")
                with tabs[1]:
                    st.subheader(f"大株主・役員情報 ({ticker_code})")
                    df_s = result.get('shareholders_df')
                    df_g = result.get('governance_df')
                    is_owner_executive = result.get('is_owner_executive', False)
                    if df_s is None and df_g is None:
                        st.warning(f"{ticker_code} の大株主・役員データ取得に失敗したか、情報がありませんでした。")
                    else:
                        if is_owner_executive:
                            st.success("✅ **注目:** 役員に大株主が含まれています！（役員リスト内で緑色の太字で表示）", icon="⭐")
                        tab1_sh, tab2_gov = st.tabs(["大株主リスト", "役員リスト"])
                        with tab1_sh:
                            st.subheader(f"大株主リスト")
                            if df_s is not None and not df_s.empty:
                                s_periods = df_s['会計期'].unique()
                                s_selected_period = st.selectbox('会計期を選択:', options=s_periods, key=f"s_period_{ticker_code}")
                                s_display_df = df_s.loc[df_s['会計期'] == s_selected_period, ['順位', '株主名', '保有割合 (%)', '保有株式数 (株)']]
                                st.dataframe(s_display_df.style.format({'保有株式数 (株)': '{:,.0f}','保有割合 (%)': '{:.2f}%'}), use_container_width=True, hide_index=True)
                                st.download_button("📋 全期間の[大株主]データをCSVダウンロード", df_s.to_csv(index=False).encode('utf-8-sig'), f"shareholders_{ticker_code}.csv", 'text/csv', use_container_width=True, key=f"dl_s_{ticker_code}")
                            else:
                                st.warning("大株主情報が見つかりませんでした。", icon="⚠️")
                        with tab2_gov:
                            st.subheader(f"役員リスト")
                            if df_g is not None and not df_g.empty:
                                def highlight_owner_executive(row):
                                    is_owner = row.get('大株主としての保有株式数', 0) > 0
                                    return ['color: #008000; font-weight: bold;'] * len(row) if is_owner else [''] * len(row)
                                g_display_df = df_g.copy()
                                if '会計期_dt' in g_display_df.columns:
                                    latest_period_row = df_g.loc[df_g['会計期_dt'].idxmax()]
                                    latest_period = latest_period_row['会計期']
                                    st.info(f"最新の役員情報（{latest_period}時点）を表示しています。")
                                    g_display_df = g_display_df.loc[g_display_df['会計期'] == latest_period]
                                display_columns = ['役職', '氏名', '生年月日', '年齢', '役員としての所有株式数', '大株主としての保有株式数', '大株主としての保有割合 (%)']
                                display_columns = [col for col in display_columns if col in g_display_df.columns]
                                g_display_df = g_display_df[display_columns]
                                st.dataframe(
                                    g_display_df.style.format({
                                        '役員としての所有株式数': '{:,.0f}', '大株主としての保有株式数': '{:,.0f}', '大株主としての保有割合 (%)': '{:.2f}%'
                                    }).apply(highlight_owner_executive, axis=1),
                                    use_container_width=True, hide_index=True)
                                st.download_button("📋 全期間の[役員]データをCSVダウンロード", df_g.to_csv(index=False).encode('utf-8-sig'), f"governance_{ticker_code}.csv", 'text/csv', use_container_width=True, key=f"dl_g_{ticker_code}")
                            else:
                                st.warning("役員情報が見つかりませんでした。", icon="⚠️")
                with tabs[2]:
                    st.subheader("PEGレシオ (CAGR) の計算過程")
                    peg_analysis = result.get('peg_analysis', {})
                    peg_data = peg_analysis.get('cagr_growth', {})
                    peg_warnings = peg_analysis.get('warnings')
                    if peg_warnings: st.info(" ".join(list(set(peg_warnings))))
                    st.markdown(f"**計算式:** `PER / (EPSのCAGR * 100)`")
                    per_val = indicators.get('variables', {}).get('PER (実績)')
                    if peg_data.get('value') is not None and isinstance(per_val, (int, float)):
                        st.text(f"PER {per_val:.2f} / (CAGR {peg_data.get('growth', 0)*100:.2f} %) = {peg_data.get('value'):.2f}")
                        st.markdown(f"**CAGR ({peg_data.get('years', 'N/A')}年) 計算:** `(最終EPS / 初期EPS) ** (1 / 年数) - 1`")
                        if all(isinstance(x, (int, float)) for x in [peg_data.get('end_eps'), peg_data.get('start_eps'), peg_data.get('years')]) and peg_data.get('years', 0) > 0:
                            st.text(f"({peg_data['end_eps']:.2f} / {peg_data['start_eps']:.2f}) ** (1 / {peg_data['years']}) - 1 = {peg_data.get('growth', 0):.4f}")
                    else: st.error(f"計算不能。理由: {peg_data.get('reason', '不明')}")
                    st.markdown("**計算に使用したEPSデータ (新しい順):**")
                    eps_points = peg_data.get('eps_points', [])
                    if eps_points: st.text(str([f"{p:.2f}" if isinstance(p, (int, float)) else "N/A" for p in eps_points]))
                    else: st.warning('EPSデータが不足しています。')
                with tabs[3]:
                    st.subheader("清原式ネットキャッシュ比率の計算過程")
                    nc_warnings = [w for w in result.get('warnings', []) if any(k in w for k in ["NC比率", "純有利子負債", "有価証券", "負債"])]
                    if nc_warnings: st.info(" ".join(list(set(nc_warnings))))
                    st.markdown(f"**計算式:** `(流動資産 + 有価証券*0.7 - 負債合計) / 時価総額`")
                    formula = indicators.get('formulas', {}).get('ネットキャッシュ比率', indicators.get('net_cash_ratio', {}).get('reason'))
                    st.text(formula)
                    st.json({k: f"{v:,.0f} 百万円" if isinstance(v, (int, float)) else "N/A" for k, v in {
                        "流動資産": indicators.get('variables', {}).get('流動資産'), "有価証券": indicators.get('variables', {}).get('有価証券'),
                        "負債合計": indicators.get('variables', {}).get('負債合計'),
                        "時価総額": indicators.get('variables', {}).get('時価総額', 0)/1e6 if isinstance(indicators.get('variables', {}).get('時価総額'), (int, float)) else None
                    }.items()})
                with tabs[4]:
                    st.subheader("キャッシュニュートラルPERの計算過程")
                    st.markdown(f"**計算式:** `実績PER * (1 - ネットキャッシュ比率)`")
                    formula = indicators.get('formulas', {}).get('キャッシュニュートラルPER', indicators.get('cn_per', {}).get('reason'))
                    st.text(formula)
                    per_val = indicators.get('variables', {}).get('PER (実績)')
                    nc_ratio_val = indicators.get('net_cash_ratio', {}).get('value')
                    st.json({
                        "実績PER": f"{per_val:.2f} 倍" if isinstance(per_val, (int, float)) else "N/A",
                        "ネットキャッシュ比率": f"{nc_ratio_val:.2f}" if isinstance(nc_ratio_val, (int, float)) else f"N/A ({indicators.get('net_cash_ratio', {}).get('reason')})"
                    })
                with tabs[5]:
                    st.subheader("ROICの計算過程")
                    if roic_warnings := [w for w in result.get('warnings', []) if "ROIC" in w]: st.info(" ".join(list(set(roic_warnings))))
                    st.markdown(f"**計算式:** `NOPAT (税引後営業利益) / 投下資本 (純資産 + 有利子負債)`")
                    formula = indicators.get('formulas', {}).get('ROIC', indicators.get('roic', {}).get('reason'))
                    st.text(formula)
                    def format_value(v, is_curr=True, is_pct=False):
                        if isinstance(v, (int, float)): return f"{v:.2%}" if is_pct else f"{v:,.0f} 百万円" if is_curr else f"{v}"
                        return "N/A"
                    roic_vars = {
                        "NOPAT計算用利益": format_value(indicators.get('variables', {}).get(f"NOPAT計算用利益 ({indicators.get('roic_source_key', '')})")),
                        "税率": format_value(indicators.get('variables', {}).get('税率'), is_pct=True),
                        "純資産": format_value(indicators.get('variables', {}).get('純資産')),
                    }
                    debt_val = indicators.get('variables', {}).get('有利子負債')
                    if debt_val is not None: roic_vars["有利子負債"] = format_value(debt_val)
                    else: roic_vars["純有利子負債(代用)"] = format_value(indicators.get('variables', {}).get('純有利子負債'))
                    st.json(roic_vars)
                with tabs[6]:
                    st.subheader("WACC (加重平均資本コスト) の計算過程")
                    if wacc_warnings := [w for w in result.get('warnings', []) if "β値" in w]: st.info(" ".join(list(set(wacc_warnings))))
                    st.markdown(f"**計算式:** `株主資本コスト * 自己資本比率 + 負債コスト * (1 - 税率) * 負債比率`")
                    formula = indicators.get('formulas', {}).get('WACC', indicators.get('wacc', {}).get('reason'))
                    st.text(formula)
                    st.json({
                        "WACC計算結果": format_value(indicators.get('wacc', {}).get('value'), is_pct=True),
                        "株主資本コスト (Ke)": format_value(indicators.get('variables', {}).get('株主資本コスト'), is_pct=True),
                        "負債コスト (Kd)": format_value(indicators.get('variables', {}).get('負債コスト'), is_pct=True),
                        "税率": format_value(indicators.get('variables', {}).get('税率'), is_pct=True),
                        "ベータ値": f"{indicators.get('variables', {}).get('ベータ値'):.2f}" if isinstance(indicators.get('variables', {}).get('ベータ値'), (int,float)) else "N/A",
                        "リスクフリーレート": f"{st.session_state.rf_rate:.2%}",
                        "マーケットリスクプレミアム": f"{mrp:.2%}"
                    })
                with tabs[7]:
                    st.subheader("PEGレシオに基づく投資家コメント (リンチ / クレイマー)")
                    commentary = get_peg_investor_commentary(indicators.get('peg', {}).get('value'))
                    st.markdown(commentary, unsafe_allow_html=True)
                with tabs[8]:
                    st.subheader("専門家コメント")
                    commentary = get_kiyohara_commentary(indicators.get('net_cash_ratio', {}).get('value'), indicators.get('cn_per', {}).get('value'), indicators.get('variables', {}).get('当期純利益'))
                    st.markdown(commentary, unsafe_allow_html=True)
                with tabs[9]:
                    st.subheader("財務諸表 (バフェットコード)")
                    bc_data = result.get('buffett_code_data', {})
                    pl_data, bs_data = bc_data.get('損益計算書', {}), bc_data.get('貸借対照表', {})
                    if pl_data or bs_data:
                        all_items = set()
                        if pl_data: all_items.update(list(pl_data.values())[0].keys())
                        if bs_data: all_items.update(list(bs_data.values())[0].keys())
                        periods = sorted(list(set(pl_data.keys()) | set(bs_data.keys())), reverse=True)
                        display_df = pd.DataFrame(index=sorted(list(all_items)), columns=periods)
                        for period in periods:
                            for item in display_df.index:
                                val = (pl_data.get(period, {}).get(item, {}) or bs_data.get(period, {}).get(item, {})).get('display')
                                display_df.loc[item, period] = val or "-"
                        st.dataframe(display_df)
                    else: st.warning("財務諸表データを取得できませんでした。")
                with tabs[10]:
                    st.subheader("ヤフーファイナンス財務データ")
                    yf_statements = result.get('yfinance_statements', {})
                    if yf_statements:
                        for title, df in yf_statements.items():
                            if not df.empty:
                                st.markdown(f"**{title}** (単位: 百万円)")
                                st.dataframe(df.style.format("{:,.0f}", na_rep="-"))
                            else: st.markdown(f"**{title}**: データなし")
                    else: st.warning("Yahoo Financeから財務データを取得できませんでした。")

    st.markdown("---")
    st.header("👑 時価総額ランキング")
    ranking_data = []
    for display_key, result in all_results.items():
        if 'error' not in result and 'yf_info' in result:
            market_cap = result['yf_info'].get('marketCap')
            sector = result.get('sector', '業種不明')
            if market_cap is not None:
                ranking_data.append({ "銘柄": display_key, "業種": sector, "時価総額": market_cap })
    if ranking_data:
        df_ranking = pd.DataFrame(ranking_data).sort_values(by="時価総額", ascending=False)
        df_ranking.index = range(1, len(df_ranking) + 1)
        df_ranking.index.name = "順位"
        def format_market_cap_display(cap):
            if cap >= 1_000_000_000_000: return f"{cap / 1_000_000_000_000:,.2f} 兆円"
            return f"{cap / 100_000_000:,.2f} 億円"
        df_display = df_ranking.copy()
        df_display['時価総額'] = df_display['時価総額'].apply(format_market_cap_display)
        st.dataframe(df_display, use_container_width=True)
    else: st.info("ランキングを表示するためのデータがありません。")

    st.markdown("---")
    st.header("👑 オーナー経営者 保有割合ランキング")
    owner_executives = []
    for display_key, result in all_results.items():
        if 'error' in result: continue
        df_g = result.get('governance_df')
        if df_g is not None and not df_g.empty and '大株主としての保有割合 (%)' in df_g.columns:
            owners = df_g[df_g['大株主としての保有割合 (%)'] > 0]
            if not owners.empty:
                top_owner = owners.loc[owners['大株主としての保有割合 (%)'].idxmax()]
                owner_executives.append({
                    '銘柄コード': result.get('ticker_code'), '銘柄名': result.get('company_name'),
                    '役職': top_owner['役職'], '氏名': top_owner['氏名'], '保有割合 (%)': top_owner['大株主としての保有割合 (%)']
                })
    if owner_executives:
        ranking_df = pd.DataFrame(owner_executives).sort_values('保有割合 (%)', ascending=False).reset_index(drop=True)
        ranking_df.index += 1
        ranking_df.index.name = "順位"
        st.dataframe(ranking_df.style.format({'保有割合 (%)': '{:.2f}%'}), use_container_width=True)
    else: st.info("分析した銘柄に、大株主を兼ねる役員は見つかりませんでした。")

    st.markdown("---")
    st.header("時系列グラフ比較")
    metrics_to_plot = ['EPS (円)', 'EPS成長率 (対前年比) (%)', 'PER (倍)', 'PBR (倍)', 'ROE (%)', '自己資本比率 (%)', '年間1株配当 (円)', 'PEG (実績)']
    selected_metric = st.selectbox("比較する指標を選択してください", metrics_to_plot, key="metric_selector")
    visible_stocks = []
    cols_check = st.columns(min(len(all_results.keys()), 4))
    for i, key in enumerate(all_results.keys()):
        if cols_check[i % len(cols_check)].checkbox(key, value=True, key=f"check_{key}"):
            visible_stocks.append(key)
    fig, ax = plt.subplots(figsize=(10, 6))
    color_list = plt.get_cmap('tab10').colors
    all_x_labels = set()
    for key in visible_stocks:
        if 'error' not in all_results.get(key, {}):
            if (df := all_results.get(key, {}).get('timeseries_df')) is not None and not df.empty and '年度' in df.columns:
                all_x_labels.update(df['年度'].dropna().tolist())
    if all_x_labels and visible_stocks:
        sorted_x_labels = sorted(list(all_x_labels), key=lambda x: (x == '最新', x))
        for i, key in enumerate(visible_stocks):
            if 'error' in all_results.get(key, {}): continue
            df = all_results[key].get('timeseries_df')
            if df is not None and not df.empty and '年度' in df.columns:
                temp_df = df.set_index('年度')
                if selected_metric not in temp_df.columns: temp_df[selected_metric] = np.nan
                y_values = [temp_df.loc[label, selected_metric] if label in temp_df.index and pd.notna(temp_df.loc[label, selected_metric]) else np.nan for label in sorted_x_labels]
                y_series = pd.Series(y_values, index=range(len(sorted_x_labels))).interpolate(method='linear')
                line, = ax.plot(range(len(sorted_x_labels)), y_series.values, marker='', linestyle='-', label=key, color=color_list[i % len(color_list)])
                for x_idx, y_val in enumerate(pd.Series(y_values)):
                    if pd.notna(y_val):
                        ax.plot(x_idx, y_val, marker='o', color=line.get_color())
                        ax.annotate(f'{y_val:.2f}', (x_idx, y_val), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9)
        ymin, ymax = ax.get_ylim()
        span_alpha = 0.15
        if selected_metric == 'PBR (倍)': ax.axhspan(0, 1, facecolor='limegreen', alpha=span_alpha)
        elif selected_metric == 'PER (倍)': ax.axhspan(0, 10, facecolor='limegreen', alpha=span_alpha)
        elif selected_metric == 'ROE (%)': ax.axhspan(10, max(ymax, 11), facecolor='limegreen', alpha=span_alpha)
        elif selected_metric == 'PEG (実績)': ax.axhspan(0, 1, facecolor='limegreen', alpha=span_alpha)
        elif selected_metric == '自己資本比率 (%)': ax.axhspan(60, max(ymax, 11), facecolor='limegreen', alpha=span_alpha)
        elif selected_metric == 'EPS成長率 (対前年比) (%)': ax.axhspan(0, max(ymax, 1), facecolor='limegreen', alpha=span_alpha)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{selected_metric} の時系列比較", fontsize=16)
        ax.set_ylabel(selected_metric)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        ax.set_xticks(range(len(sorted_x_labels)))
        ax.set_xticklabels(sorted_x_labels, rotation=30, ha='right')
        st.pyplot(fig)
    else: st.warning("グラフを描画できる銘柄が選択されていません。")

else:
    st.info("サイドバーから銘柄コードまたは会社名を入力して「分析実行」ボタンを押すか、「AI類似銘柄検索」をご利用ください。")
