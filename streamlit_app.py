import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import logging
import time
import re
from datetime import datetime, date
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
# pyperclip はサーバー環境で動作しないため削除
# import pyperclip 
import unicodedata

# ==============================================================================
# 1. ログ設定
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 2. 銘柄検索用のヘルパー関数とデータロード
# ==============================================================================
JPX_STOCK_LIST_PATH = "jpx_list.xls"

@st.cache_data
def load_jpx_stock_list():
    """JPXの上場銘柄一覧を読み込み、キャッシュする"""
    try:
        df = pd.read_excel(JPX_STOCK_LIST_PATH, header=None)

        # ファイルに必要な列数があるか確認（業種区分は6列目にあるため6列以上必要）
        if df.shape[1] < 6:
            st.error(f"銘柄リストファイル({JPX_STOCK_LIST_PATH})の形式が想定と異なります。業種区分を含む列数が不足しています。")
            return pd.DataFrame(columns=['code', 'name', 'market', 'sector', 'normalized_name'])

        # ★変更点：33業種区分（6列目、インデックス5）も読み込む
        df = df.iloc[:, [1, 2, 3, 5]]
        df.columns = ['code', 'name', 'market', 'sector']
        
        df.dropna(subset=['code', 'name'], inplace=True)
        df = df[df['code'].apply(lambda x: isinstance(x, (int, float)) and 1000 <= x <= 9999)]
        df['code'] = df['code'].astype(int).astype(str)
        df['normalized_name'] = df['name'].apply(normalize_text)
        logger.info(f"銘柄リストをロードしました: {len(df)}件")
        return df
    except FileNotFoundError:
        st.error(f"銘柄リストファイル ({JPX_STOCK_LIST_PATH}) が見つかりません。JPXサイトからダウンロードして、コードと同じフォルダに配置してください。")
        return pd.DataFrame(columns=['code', 'name', 'market', 'sector', 'normalized_name'])
    except Exception as e:
        st.error(f"銘柄リストの読み込み中に予期せぬエラーが発生しました: {e}")
        return pd.DataFrame(columns=['code', 'name', 'market', 'sector', 'normalized_name'])

def normalize_text(text: str) -> str:
    """検索クエリと銘柄名を比較のために正規化する"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = "".join([chr(ord(c) + 96) if "ぁ" <= c <= "ん" else c for c in text])
    text = text.upper()
    remove_words = ['ホールディングス', 'グループ', '株式会社', '合同会社', '有限会社', '(株)', '(同)', '(有)']
    for word in remove_words:
        text = text.replace(word, '')
    return text.strip()

# ==============================================================================
# 3. データ処理クラス
# ==============================================================================
class IntegratedDataHandler:
    """両プログラムのデータ取得・分析ロジックを統合"""
    
    def __init__(self):
        """初期化時に銘柄リストを読み込む"""
        self.stock_list_df = load_jpx_stock_list()

    def get_ticker_info_from_query(self, query: str) -> dict | None:
        """★変更点：銘柄コードだけでなく業種などの情報も辞書で返す"""
        query = query.strip()

        if re.fullmatch(r'\d{4}', query):
            if not self.stock_list_df.empty:
                stock_data = self.stock_list_df[self.stock_list_df['code'] == query]
                if not stock_data.empty:
                    return stock_data.iloc[0].to_dict()
                else:
                    logger.warning(f"銘柄コード '{query}' はリストに存在しませんが、分析を試みます。")
                    return {'code': query, 'name': f'銘柄 {query}', 'sector': '業種不明'}
            return {'code': query, 'name': f'銘柄 {query}', 'sector': '業種不明'}

        if self.stock_list_df.empty:
            return None

        normalized_query = normalize_text(query)
        if not normalized_query:
            return None

        matches = self.stock_list_df[self.stock_list_df['normalized_name'].str.contains(normalized_query, na=False)]

        if not matches.empty:
            prime_matches = matches[matches['market'].str.contains('プライム', na=False)]
            if not prime_matches.empty:
                stock_data = prime_matches.iloc[0]
            else:
                stock_data = matches.iloc[0]
            
            logger.info(f"検索クエリ '{query}' から銘柄 '{stock_data['name']} ({stock_data['code']})' を見つけました。")
            return stock_data.to_dict()

        logger.warning(f"検索クエリ '{query}' に一致する銘柄が見つかりませんでした。")
        return None

    YFINANCE_TRANSLATION_MAP = {
        'Total Revenue': '売上高', 'Revenue': '売上高',
        'Operating Income': '営業利益', 'Operating Expense': '営業費用',
        'Cost Of Revenue': '売上原価', 'Gross Profit': '売上総利益',
        'Selling General And Administration': '販売費及び一般管理費',
        'Research And Development': '研究開発費',
        'Pretax Income': '税引前利益', 'Tax Provision': '法人税',
        'Net Income': '当期純利益', 'Net Income Common Stockholders': '親会社株主に帰属する当期純利益',
        'Basic EPS': '1株当たり利益 (EPS)', 'Diluted EPS': '希薄化後EPS',
        'Total Assets': '総資産', 'Current Assets': '流動資産',
        'Cash And Cash Equivalents': '現金及び現金同等物', 'Cash': '現金',
        'Receivables': '売上債権', 'Inventory': '棚卸資産',
        'Total Non Current Assets': '固定資産', 'Net PPE': '有形固定資産',
        'Goodwill And Other Intangible Assets': 'のれん及びその他無形固定資産',
        'Total Liabilities Net Minority Interest': '負債合計', 'Current Liabilities': '流動負債',
        'Payables And Accrued Expenses': '支払手形及び買掛金', 'Current Debt': '短期有利子負債',
        'Total Non Current Liabilities Net Minority Interest': '固定負債', 'Long Term Debt': '長期有利子負債',
        'Total Equity Gross Minority Interest': '純資産合計', 'Stockholders Equity': '株主資本',
        'Retained Earnings': '利益剰余金',
        'Cash Flow From Continuing Operating Activities': '営業キャッシュフロー',
        'Cash Flow From Continuing Investing Activities': '投資キャッシュフロー',
        'Cash Flow From Continuing Financing Activities': '財務キャッシュフロー',
        'Net Change In Cash': '現金の増減額', 'Free Cash Flow': 'フリーキャッシュフロー',
    }

    def get_html_soup(self, url: str) -> BeautifulSoup | None:
        logger.info(f"URLへのアクセスを開始: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8'
        }
        try:
            time.sleep(1.2)
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            logger.error(f"URLへのアクセス失敗: {url}, エラー: {e}")
            return None

    def get_risk_free_rate(self) -> float | None:
        url = "https://jp.investing.com/rates-bonds/japan-10-year-bond-yield"
        logger.info(f"リスクフリーレート取得試行: {url}")
        soup = self.get_html_soup(url)
        if soup:
            try:
                yield_div = soup.find('div', attrs={'data-test': 'instrument-price-last'})
                if yield_div:
                    return float(yield_div.get_text(strip=True)) / 100
            except Exception as e:
                logger.error(f"リスクフリーレートの解析に失敗: {e}")
        logger.warning("リスクフリーレートの自動取得に失敗しました。")
        return None
    
    def parse_financial_value(self, s: str) -> int | float | None:
        s = str(s).replace(',', '').strip()
        if s in ['-', '---', '']:
            return None
        is_negative = s.startswith(('△', '-'))
        s = s.lstrip('△-')
        try:
            total = 0
            if '兆' in s:
                total += float(re.findall(r'(\d+\.?\d*)', s)[0]) * 1000000
            elif '億' in s:
                total += float(re.findall(r'(\d+\.?\d*)', s)[0]) * 100
            elif '百万円' in s:
                total += float(re.findall(r'(\d+\.?\d*)', s)[0])
            elif '万円' in s:
                total += float(re.findall(r'(\d+\.?\d*)', s)[0]) * 0.01
            elif re.match(r'^\d+\.?\d*$', s):
                total = float(s)
            else:
                return s
            return -int(total) if is_negative else int(total)
        except (ValueError, TypeError, IndexError):
            return s

    def extract_all_financial_data(self, soup: BeautifulSoup) -> dict | None:
        financial_table = soup.find('table', class_='financial-table')
        if not financial_table:
            return None
        thead, tbody = financial_table.find('thead'), financial_table.find('tbody')
        if not thead or not tbody:
            return None
        period_headers = thead.find('tr').find_all('th')
        if len(period_headers) <= 1:
            return None
        valid_periods = []
        for i, th in enumerate(period_headers[1:]):
            header_text = th.text.strip()
            if header_text and "E" not in header_text.upper() and "C" not in header_text.upper():
                valid_periods.append({'name': header_text, 'index': i + 1})
        if not valid_periods:
            return None
        all_periods_data = OrderedDict()
        for row in tbody.find_all('tr'):
            cells = row.find_all(['th', 'td'])
            item_name = cells[0].text.strip()
            if not item_name or not re.search(r'[a-zA-Z\u3040-\u30FF\u4E00-\u9FFF]', item_name):
                continue
            for period in valid_periods:
                period_name = period['name']
                if period_name not in all_periods_data:
                    all_periods_data[period_name] = {}
                if len(cells) > period['index']:
                    display_value = cells[period['index']].get_text(strip=True)
                    if display_value not in ['-', '---', '']:
                        all_periods_data[period_name][item_name] = {
                            'display': display_value,
                            'raw': self.parse_financial_value(display_value)
                        }
        return all_periods_data

    def get_latest_financial_data(self, financial_data_dict: dict) -> dict:
        latest_year = -1
        latest_month = -1
        latest_data = {}
        if not financial_data_dict:
            return {}
            
        for period_name, data in financial_data_dict.items():
            match = re.search(r'(\d{2,4})[./](\d{1,2})', period_name)
            if match:
                year_str, month_str = match.groups()
                year = int(year_str)
                month = int(month_str)
                
                if year < 100:
                    year_full = 2000 + year if year < 50 else 1900 + year
                else:
                    year_full = year

                if year_full > latest_year or (year_full == latest_year and month > latest_month):
                    latest_year = year_full
                    latest_month = month
                    latest_data = data
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
        if df.empty:
            return df
        
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
        if pe is None and trailing_eps is not None and trailing_eps < 0:
            return {'score': 10, 'evaluation': '【赤字企業 (EPS基準)】'}
        is_profitable = keijo_rieki is not None and keijo_rieki > 0
        if not is_profitable:
            return {'score': 10, 'evaluation': '【赤字・要注意】'}
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
            score = 0
            evaluation = "【成長鈍化・赤字】" if peg_ratio is not None else "---"
        elif peg_ratio <= 0.5:
            score = 100
            evaluation = "【超割安な成長株】"
        elif peg_ratio <= 1.0:
            score = self._linear_interpolate(peg_ratio, 0.5, 100, 1.0, 70)
            evaluation = "【割安な成長株】"
        elif peg_ratio <= 1.5:
            score = self._linear_interpolate(peg_ratio, 1.0, 70, 1.5, 40)
            evaluation = "【適正価格】"
        elif peg_ratio < 2.0:
            score = self._linear_interpolate(peg_ratio, 1.5, 40, 2.0, 0)
            evaluation = "【やや割高】"
        else:
            score = 0
            evaluation = "【割高】"
        return {'score': int(score), 'evaluation': evaluation}

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
        minority_interest = self.get_value(latest_bs_data, ['非支配株主持分'], '非支配株主持分')
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
            'historical_pegs': {} 
        }
        
        try:
            current_per = info.get('trailingPE')
            if not current_per:
                for key in results:
                    if key != 'historical_pegs': results[key]['reason'] = '現在のPERが取得できません'
                return results

            financials = ticker_obj.financials
            if financials.empty or 'Basic EPS' not in financials.index:
                for key in results:
                    if key != 'historical_pegs': results[key]['reason'] = 'EPSデータが見つかりません'
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

                    if start_eps > 0 and end_eps > 0:
                        cagr = (end_eps / start_eps)**(1/years) - 1
                        results['cagr_growth']['growth'] = cagr
                        if cagr > 0:
                            results['cagr_growth']['value'] = current_per / (cagr * 100)
                            results['cagr_growth']['reason'] = f'{years}年間のCAGR'
                        else:
                            results['cagr_growth']['reason'] = f'{years}年CAGRがマイナス'
                    else:
                        results['cagr_growth']['reason'] = '開始または終了EPSがマイナス'
                else:
                    results['cagr_growth']['reason'] = '有効なEPSが2地点未満'

            history = ticker_obj.history(period="6y")
            if not history.empty and len(annual_eps_data) >= 2:
                history.index = history.index.tz_localize(None)
                for i in range(len(annual_eps_data) - 1):
                    eps_curr = annual_eps_data.iloc[i]
                    eps_prev = annual_eps_data.iloc[i+1]
                    year_date = annual_eps_data.index[i]
                    
                    if pd.notna(eps_curr) and pd.notna(eps_prev) and eps_prev > 0:
                        yoy_growth = (eps_curr - eps_prev) / eps_prev
                        if yoy_growth > 0:
                            price_at_fis_year = history.asof(year_date)['Close']
                            if price_at_fis_year:
                                historical_per = price_at_fis_year / eps_curr
                                peg = historical_per / (yoy_growth * 100)
                                results['historical_pegs'][f"{year_date.year}年度"] = peg

        except Exception as e:
            logger.error(f"PEGレシオ計算中にエラー: {e}", exc_info=True)

        return results

    def perform_full_analysis(self, ticker_code: str, options: dict) -> dict:
        result = {'ticker_code': ticker_code, 'warnings': [], 'buffett_code_data': {}, 'timeseries_df': pd.DataFrame()}
        try:
            logger.info(f"--- 銘柄 {ticker_code} の分析を開始 ---")
            
            info = None
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
            
            for statement, path in {"貸借対照表": "bs", "損益計算書": "pl"}.items():
                soup = self.get_html_soup(f"https://www.buffett-code.com/company/{ticker_code}/financial/{path}")
                if soup:
                    all_data = self.extract_all_financial_data(soup)
                    if all_data:
                        result['buffett_code_data'][statement] = all_data
                    else:
                        logger.warning(f"Buffett-Codeから{statement}のデータ解析に失敗。")
                        result['buffett_code_data'][statement] = {}
                else:
                    result['buffett_code_data'][statement] = {}
            
            yf_data_for_calc = {**info, **options}
            result['scoring_indicators'] = self._calculate_scoring_indicators(result['buffett_code_data'], yf_data_for_calc)
            result['warnings'].extend(result['scoring_indicators'].pop('calc_warnings', []))
            
            peg_results = self.calculate_peg_ratios(ticker_obj, info)
            result['peg_analysis'] = peg_results

            cagr_peg_value = peg_results['cagr_growth']['value']
            peg_score_dict = self._calculate_peg_score(cagr_peg_value)
            result['scoring_indicators']['peg'] = {'value': cagr_peg_value, 'reason': peg_results['cagr_growth']['reason'], **peg_score_dict}
            
            s1 = result['scoring_indicators']['net_cash_ratio']['score']
            s2 = result['scoring_indicators']['cn_per']['score']
            s3 = result['scoring_indicators']['roic']['score']
            s4 = result['scoring_indicators']['peg']['score']
            result['final_average_score'] = (s1 + s2 + s3 + s4) / 4

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

        except Exception as e:
            logger.error(f"銘柄 {ticker_code} の分析中にエラーが発生しました: {e}", exc_info=True)
            result['error'] = f"分析中にエラーが発生しました: {e}"
            if 'company_name' not in result:
                result['company_name'] = f"銘柄 {ticker_code} (エラー)"
        return result

# ==============================================================================
# 4. GUIアプリケーションクラス (Streamlit)
# ==============================================================================

# --- UI Helper Functions ---
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

    cn_per_comment = "\n\n<br><br>\n\n### キャッシュニュートラルPER（CN-PER）の評価\n\n"
    if cn_per is None: cn_per_comment += "評価不能 (純利益がゼロのため計算不可)"
    elif cn_per < 0:
        if net_income is not None and net_income > 0: cn_per_comment += "【究極の割安株】🤑 お宝株の可能性大。事業価値がマイナス（時価総額 < ネットキャッシュ）なのに利益は出ている状態。なぜ市場がこれほどまでに評価していないのか、隠れたリスクがないかを精査する価値が非常に高いです。"
        else: cn_per_comment += "【要注意株】🧐 「価値の罠」の可能性あり。事業が利益を生み出せていない赤字状態。どれだけ資産を持っていても、事業活動でそれを食いつぶしているかもしれません。赤字が一時的なものか、構造的なものか、その原因を詳しく調べる必要があります。"
    elif 0 <= cn_per < 2: cn_per_comment += "【現金より安い会社】🤯 💎\n\n> 「時価総額からネットキャッシュを引いた事業価値が、純利益の1～2年分しかないということ。これはもう、『ほぼタダ』で会社が手に入るのに等しい。なぜ市場がここまで見捨てているのか、何か特別な悪材料がないか徹底的に調べる必要があるが、そうでなければ『ありえない安値』だ。こういう会社は、誰かがその価値に気づけば、株価は簡単に2倍、3倍になる可能性を秘めている」\n\n**評価:** 最大限の買い評価。ただし、異常な安さの裏に隠れたリスク（訴訟、偶発債務など）がないかは慎重に確認する、というスタンスでしょう 🤔。"
    elif 2 <= cn_per < 4: cn_per_comment += "【私の投資のど真ん中】🎯 💪\n\n> 「実質PERが4倍以下。これが私の投資のど真ん中だ。 この水準であれば、多少の成長性の鈍化や業績のブレなど意に介さない。事業価値がこれだけ安ければ、下値リスクは限定的。市場参加者の多くがその価値に気づいていないだけで、放っておけばいずれ評価される。こういう銘柄こそ、安心して大きな金額を投じられる」\n\n**評価:** 最も信頼を置き、積極的に投資対象とする「コア・ゾーン」です。彼の投資術の神髄がこの価格帯にあると言えます ✅。"
    elif 4 <= cn_per < 7: cn_per_comment += "【まあ、悪くない水準】👍 🙂\n\n> 「実質PERが5倍、6倍ね…。まあ、悪くない水準だ。普通のバリュー投資家なら喜んで買うだろう。ただ、私に言わせれば、ここから先は『普通に安い会社』であって、驚くほどの安さではない。他に買うべきものがなければ検討するが、胸を張って『これは買いだ』と断言するには少し物足りなさを感じる」\n\n**評価:** 許容範囲ではあるものの、最高の投資対象とは見なしません。より割安な銘柄があれば、そちらを優先するでしょう。"
    elif 7 <= cn_per < 10: cn_per_comment += "【普通の会社】😐 📈\n\n> 「実質PERが10倍近くになってくると、もはや割安とは言えない。『普通の会社』の値段だ。 この水準の株を買うのであれば、ネットキャッシュの価値だけでは不十分で、将来の成長性がどれだけあるかという議論が不可欠になる。しかし、私にはその未来を正確に予測する能力はない」\n\n**評価:** 彼の得意とする「資産価値」を拠り所とした投資スタイルからは外れ始めます。成長性の評価という不確実な領域に入るため、投資対象としての魅力は大きく薄れます 🤷‍♂️。"
    elif 10 <= cn_per < 15: cn_per_comment += "【私には割高に思える】🤨 👎\n\n> 「多くの市場参加者が『適正水準だ』と言うかもしれないが、私にはもう割高に思える。ネットキャッシュを差し引いた事業価値ですら、利益の10年以上分を払うということ。それだけの価値があるというなら、よほど素晴らしい成長ストーリーと、それを実現できる経営陣が必要になる。私には博打にしか見えない 🎲」\n\n**評価:** 明確に「割高」と判断し、通常は投資対象としません。"
    else: cn_per_comment += "【論外。バブル以外の何物でもない】❌ 🤮\n\n> 「実質PERが20倍だの30倍だのというのは、はっきり言って論外だ。 どれだけ輝かしい未来を語られようと、それは単なる夢物語。株価は期待だけで形成されている。こういう会社がその後どうなるか、私は何度も見てきた。これは投資ではなく投機であり、バブル以外の何物でもない 💥。アナリストが全員で強気な推薦をしていたら、むしろ空売りを検討するくらいだ」\n\n**評価:** 投資対象として全く考えない水準です。むしろ市場の過熱を示すサインと捉え、警戒を強めるでしょう。"

    return nc_comment + cn_per_comment

# --- Main App ---
st.set_page_config(page_title="統合型 企業価値分析ツール", layout="wide")

st.sidebar.title("分析設定")
ticker_input = st.sidebar.text_area("銘柄コード or 会社名 (カンマ区切り)", "6758, トヨタ, 9984")

if 'rf_rate' not in st.session_state:
    st.session_state.rf_rate = 0.01

if 'rf_rate_fetched' not in st.session_state:
    with st.spinner("最新のリスクフリーレートを取得中..."):
        handler = IntegratedDataHandler()
        rate = handler.get_risk_free_rate()
        if rate is not None:
            st.session_state.rf_rate = rate
    st.session_state.rf_rate_fetched = True 

rf_rate_input = st.sidebar.number_input("リスクフリーレート(Rf)", value=st.session_state.rf_rate, format="%.4f")
st.session_state.rf_rate = rf_rate_input
mrp = st.sidebar.number_input("マーケットリスクプレミアム(MRP)", value=0.06, format="%.2f")
analyze_button = st.sidebar.button("分析実行")

st.title("統合型 企業価値分析ツール")

if 'results' not in st.session_state:
    st.session_state.results = None

# ★変更点：業種情報を分析結果に含めるように関数を修正
def run_analysis_for_all(stocks_to_analyze, options_str):
    options = eval(options_str)
    all_results = {}
    data_handler = IntegratedDataHandler()
    for stock_info in stocks_to_analyze:
        code = stock_info['code']
        result = data_handler.perform_full_analysis(code, options)
        result['sector'] = stock_info.get('sector', '業種不明') # 分析結果に業種を追加
        display_key = f"{result.get('company_name', code)} ({code})"
        all_results[display_key] = result
    return all_results

if analyze_button:
    input_queries = [q.strip() for q in ticker_input.split(',') if q.strip()]
    if not input_queries:
        st.error("銘柄コードまたは会社名を入力してください。")
    else:
        search_handler = IntegratedDataHandler()
        target_stocks = []
        not_found_queries = []
        
        with st.spinner("銘柄を検索しています..."):
            for query in input_queries:
                stock_info = search_handler.get_ticker_info_from_query(query)
                if stock_info:
                    target_stocks.append(stock_info)
                else:
                    not_found_queries.append(query)
        
        # 重複する銘柄コードを削除（辞書リストの重複削除）
        unique_target_stocks = list({stock['code']: stock for stock in target_stocks}.values())

        if not_found_queries:
            st.warning(f"以下の銘柄は見つかりませんでした: {', '.join(not_found_queries)}")

        if not unique_target_stocks:
            st.error("分析対象の銘柄が見つかりませんでした。入力内容を確認してください。")
            st.session_state.results = None
        else:
            display_codes = [s['code'] for s in unique_target_stocks]
            st.success(f"分析対象: {', '.join(display_codes)}")
            options = {'risk_free_rate': st.session_state.rf_rate, 'mkt_risk_premium': mrp}
            
            with st.spinner(f'分析中... ({len(unique_target_stocks)}件)'):
                all_results = run_analysis_for_all(unique_target_stocks, str(options))
            
            st.session_state.results = all_results


if st.session_state.results:
    all_results = st.session_state.results
    
    st.header("個別銘柄サマリー")
    sorted_results = sorted(all_results.items(), key=lambda item: item[1].get('final_average_score', -1), reverse=True)

    for display_key, result in sorted_results:
        if 'error' in result:
            with st.expander(f"▼ {display_key} - 分析エラー", expanded=True):
                st.error(f"分析中にエラーが発生しました。\n\n詳細: {result['error']}")
            continue 

        score = result.get('final_average_score')
        stars_text = "⭐" * int(get_recommendation(score)[0].count('★')) + "☆" * int(get_recommendation(score)[0].count('☆'))
        score_color = "#28a745" if score >= 70 else "#ffc107" if score >= 40 else "#dc3545"
        score_text = f"{score:.1f}" if score is not None else "N/A"
        
        st.markdown(f"<hr style='border: 2px solid {score_color}; border-radius: 2px;'>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([0.55, 0.3, 0.15])
        with col1:
            # ★変更点：銘柄名の横に業種を表示
            sector = result.get('sector', '')
            if sector and pd.notna(sector):
                 st.markdown(f"### {display_key} <span style='font-size: 16px; color: grey; font-weight: normal; margin-left: 10px;'>({sector})</span>", unsafe_allow_html=True)
            else:
                 st.markdown(f"### {display_key}")
        
        with col2:
            info = result.get('yf_info', {})
            price = info.get('regularMarketPrice')
            change = info.get('regularMarketChange')
            previous_close = info.get('regularMarketPreviousClose')
            
            change_pct_to_display = None
            
            if isinstance(price, (int, float)) and isinstance(previous_close, (int, float)) and previous_close > 0:
                change_pct_to_display = (price - previous_close) / previous_close
            else:
                change_pct_to_display = info.get('regularMarketChangePercent')

            if all(isinstance(x, (int, float)) for x in [price, change, change_pct_to_display]):
                st.metric(label="現在株価", value=f"{price:,.0f} 円", delta=f"前日比 {change:+.2f}円 ({change_pct_to_display:+.2%})", delta_color="normal")
            
        with col3:
            st.write("") 
            st.write("") 
            indicators = result.get('scoring_indicators', {})
            peg_data = indicators.get('peg', {})
            nc_data = indicators.get('net_cash_ratio', {})
            cnper_data = indicators.get('cn_per', {})
            roic_data = indicators.get('roic', {})

            def format_for_copy(data):
                val = data.get('value')
                return f"{val:.2f} ({data.get('evaluation', '')})" if val is not None else "N/A"
            
            change_pct_text = f"({change_pct_to_display:+.2%})" if isinstance(change_pct_to_display, (int, float)) else ""
            price_text = f"株価: {price:,.0f}円 (前日比 {change:+.2f}円, {change_pct_text})" if all(isinstance(x, (int, float)) for x in [price, change]) else ""

            copy_text = (
                f"■ {display_key}\n"
                f"{price_text}\n"
                f"総合スコア: {score:.1f}点 {stars_text}\n"
                f"--------------------\n"
                f"PEG (CAGR): {format_for_copy(peg_data)}\n"
                f"ネットキャッシュ比率: {format_for_copy(nc_data)}\n"
                f"CN-PER: {format_for_copy(cnper_data)}\n"
                f"ROIC: {format_for_copy(roic_data)}"
            )

            # --- ▼▼▼ ここからが修正箇所 ▼▼▼ ---
            # st.text_area を使った安定したコピー機能
            toggle_key = f"show_copy_area_{display_key}"
            if toggle_key not in st.session_state:
                st.session_state[toggle_key] = False

            if st.button("📋 結果をコピー用に表示/非表示", key=f"toggle_button_{display_key}"):
                st.session_state[toggle_key] = not st.session_state[toggle_key]
            
            if st.session_state[toggle_key]:
                st.text_area(
                    "以下のテキストをコピーしてください:",
                    copy_text,
                    height=200,
                    key=f"text_area_{display_key}",
                    help="右上のコピーボタンを押すか、テキスト全体を選択してコピーしてください。"
                )
            # --- ▲▲▲ ここまでが修正箇所 ▲▲▲ ---

        st.markdown(f"#### 総合スコア: <span style='font-size: 28px; font-weight: bold; color: {score_color};'>{score_text}点</span> <span style='font-size: 32px;'>{stars_text}</span>", unsafe_allow_html=True)
        
        if result.get('warnings'):
            st.info(f"注記: {'; '.join(list(set(result.get('warnings',[]))))}。詳細は各計算タブを確認してください。")

        with st.container():
            cols = st.columns(4)
            
            def show_metric(column, title, data, warnings):
                with column:
                    note = ""
                    if title == "PEG (CAGR)" and any("PEG" in w for w in warnings): note = " *"
                    if title == "ネットキャッシュ比率" and any(k in w for w in warnings for k in ["NC比率", "負債", "有価証券"]): note = " *"
                    if title == "CN-PER" and any(k in w for w in warnings for k in ["NC比率", "負債", "有価証券"]): note = " *"
                    if title == "ROIC" and any("ROIC" in w for w in warnings): note = " *"
                    
                    val = data.get('value')
                    val_str = f"{val:.2f}" if val is not None else "N/A"
                    score_val = data.get('score', 0)
                    color = "#28a745" if score_val >= 70 else "#ffc107" if score_val >= 40 else "#dc3545"
                    
                    st.markdown(f"<p style='font-size: 14px; color: #555; font-weight: bold; text-align: center; margin-bottom: 0;'>{title}{note}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 28px; color: {color}; font-weight: bold; text-align: center; margin: 0;'>{val_str}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; font-weight: bold; font-size: 14px;'>スコア: <span style='color:{color};'>{score_val:.1f}点</span></p>", unsafe_allow_html=True)
                    
                    if val is None:
                        st.markdown(f"<p style='text-align: center; font-size: 12px; color: red;'>({data.get('reason', '計算不能')})</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='text-align: center; font-size: 12px; color: #777;'>{data.get('evaluation', '---')}</p>", unsafe_allow_html=True)
            
            show_metric(cols[0], "PEG (CAGR)", indicators.get('peg', {}), result.get('warnings', []))
            show_metric(cols[1], "ネットキャッシュ比率", indicators.get('net_cash_ratio', {}), result.get('warnings', []))
            show_metric(cols[2], "CN-PER", indicators.get('cn_per', {}), result.get('warnings', []))
            show_metric(cols[3], "ROIC", indicators.get('roic', {}), result.get('warnings', []))
            
            with st.expander("詳細データを見る"):
                tabs = st.tabs([
                    "時系列指標", "PEG計算", "NC比率計算", "CN-PER計算", "ROIC計算", "WACC計算",
                    "PEGレシオコメント", "専門家コメント", "財務諸表(バフェットコード)", "ヤフーファイナンス財務"
                ])
                
                with tabs[0]: # 時系列指標
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

                with tabs[1]: # PEG計算
                    st.subheader("PEG (CAGR) の計算過程")
                    peg_analysis = result.get('peg_analysis', {})
                    peg_data = peg_analysis.get('cagr_growth', {})
                    
                    st.markdown(f"**計算式:** `PER / (EPSのCAGR * 100)`")
                    per_val = indicators.get('variables', {}).get('PER (実績)')
                    if peg_data.get('value') is not None and isinstance(per_val, (int, float)):
                        st.text(f"PER {per_val:.2f} / (CAGR {peg_data.get('growth', 0)*100:.2f} %) = {peg_data.get('value'):.2f}")
                        
                        st.markdown(f"**CAGR ({peg_data.get('years', 'N/A')}年) 計算:** `(最終EPS / 初期EPS) ** (1 / 年数) - 1`")
                        if all(isinstance(x, (int, float)) for x in [peg_data.get('end_eps'), peg_data.get('start_eps'), peg_data.get('years')]) and peg_data.get('years', 0) > 0:
                            st.text(f"({peg_data['end_eps']:.2f} / {peg_data['start_eps']:.2f}) ** (1 / {peg_data['years']}) - 1 = {peg_data.get('growth', 0):.4f}")
                    else:
                        st.error(f"計算不能。理由: {peg_data.get('reason', '不明')}")
                    
                    st.markdown("**計算に使用したEPSデータ (新しい順):**")
                    eps_points = peg_data.get('eps_points', [])
                    if eps_points:
                        st.text(str([f"{p:.2f}" if isinstance(p, (int, float)) else "N/A" for p in eps_points]))
                    else:
                        st.warning('EPSデータが不足しています。')

                with tabs[2]: # NC比率計算
                    st.subheader("清原式ネットキャッシュ比率の計算過程")
                    nc_warnings = [w for w in result.get('warnings', []) if "NC比率" in w or "純有利子負債" in w or "有価証券" in w or "負債" in w]
                    if nc_warnings:
                        st.info(" ".join(list(set(nc_warnings))))
                    st.markdown(f"**計算式:** `(流動資産 + 有価証券*0.7 - 負債合計) / 時価総額`")
                    formula = indicators.get('formulas', {}).get('ネットキャッシュ比率', indicators.get('net_cash_ratio', {}).get('reason'))
                    st.text(formula)
                    
                    st.json({
                        "流動資産": f"{indicators.get('variables', {}).get('流動資産', 'N/A'):,.0f} 百万円" if isinstance(indicators.get('variables', {}).get('流動資産'), (int, float)) else "N/A",
                        "有価証券": f"{indicators.get('variables', {}).get('有価証券', 'N/A'):,.0f} 百万円" if isinstance(indicators.get('variables', {}).get('有価証券'), (int, float)) else "N/A",
                        "負債合計": f"{indicators.get('variables', {}).get('負債合計', 'N/A'):,.0f} 百万円" if isinstance(indicators.get('variables', {}).get('負債合計'), (int, float)) else "N/A",
                        "時価総額": f"{indicators.get('variables', {}).get('時価総額', 0)/1e6:,.0f} 百万円" if isinstance(indicators.get('variables', {}).get('時価総額'), (int, float)) else "N/A"
                    })
                
                with tabs[3]: # CN-PER計算
                    st.subheader("CN-PERの計算過程")
                    st.markdown(f"**計算式:** `実績PER * (1 - ネットキャッシュ比率)`")
                    formula = indicators.get('formulas', {}).get('キャッシュニュートラルPER', indicators.get('cn_per', {}).get('reason'))
                    st.text(formula)
                    
                    per_val = indicators.get('variables', {}).get('PER (実績)')
                    nc_ratio_val = indicators.get('net_cash_ratio', {}).get('value')
                    
                    st.json({
                        "実績PER": f"{per_val:.2f} 倍" if isinstance(per_val, (int, float)) else f"N/A ({'データなし'})",
                        "ネットキャッシュ比率": f"{nc_ratio_val:.2f}" if isinstance(nc_ratio_val, (int, float)) else f"N/A ({indicators.get('net_cash_ratio', {}).get('reason')})"
                    })

                with tabs[4]: # ROIC計算
                    st.subheader("ROICの計算過程")
                    roic_warnings = [w for w in result.get('warnings', []) if "ROIC" in w]
                    if roic_warnings:
                        st.info(" ".join(list(set(roic_warnings))))
                    st.markdown(f"**計算式:** `NOPAT (税引後営業利益) / 投下資本 (純資産 + 有利子負債)`")
                    formula = indicators.get('formulas', {}).get('ROIC', indicators.get('roic', {}).get('reason'))
                    st.text(formula)
                    
                    def format_value(value, is_currency=True, is_percent=False):
                        if isinstance(value, (int, float)):
                            if is_percent: return f"{value:.2%}"
                            return f"{value:,.0f} 百万円" if is_currency else f"{value}"
                        return "N/A"
                    
                    op_income_val = indicators.get('variables', {}).get(f"NOPAT計算用利益 ({indicators.get('roic_source_key', '')})")
                    tax_rate_val = indicators.get('variables', {}).get('税率')
                    net_assets_val = indicators.get('variables', {}).get('純資産')
                    debt_val = indicators.get('variables', {}).get('有利子負債')
                    net_debt_val = indicators.get('variables', {}).get('純有利子負債')

                    roic_vars = {
                        "NOPAT計算用利益": format_value(op_income_val),
                        "税率": format_value(tax_rate_val, is_percent=True),
                        "純資産": format_value(net_assets_val),
                    }
                    if debt_val is not None:
                        roic_vars["有利子負債"] = format_value(debt_val)
                    else:
                        roic_vars["純有利子負債(代用)"] = format_value(net_debt_val)
                    st.json(roic_vars)
                
                with tabs[5]: # WACC計算
                    st.subheader("WACC (加重平均資本コスト) の計算過程")
                    wacc_warnings = [w for w in result.get('warnings', []) if "β値" in w]
                    if wacc_warnings:
                        st.info(" ".join(list(set(wacc_warnings))))
                    st.markdown(f"**計算式:** `株主資本コスト * 自己資本比率 + 負債コスト * (1 - 税率) * 負債比率`")
                    formula = indicators.get('formulas', {}).get('WACC', indicators.get('wacc', {}).get('reason'))
                    st.text(formula)
                    
                    st.json({
                        "WACC計算結果": format_value(indicators.get('wacc', {}).get('value'), is_percent=True),
                        "株主資本コスト (Ke)": format_value(indicators.get('variables', {}).get('株主資本コスト'), is_percent=True),
                        "負債コスト (Kd)": format_value(indicators.get('variables', {}).get('負債コスト'), is_percent=True),
                        "税率": format_value(indicators.get('variables', {}).get('税率'), is_percent=True),
                        "ベータ値": f"{indicators.get('variables', {}).get('ベータ値'):.2f}" if isinstance(indicators.get('variables', {}).get('ベータ値'), (int,float)) else "N/A",
                        "リスクフリーレート": f"{st.session_state.rf_rate:.2%}",
                        "マーケットリスクプレミアム": f"{mrp:.2%}"
                    })
                
                with tabs[6]: # PEGレシオコメント
                    st.subheader("PEGレシオに基づく投資家コメント (リンチ / クレイマー)")
                    peg_data = indicators.get('peg', {})
                    commentary = get_peg_investor_commentary(peg_data.get('value'))
                    st.markdown(commentary, unsafe_allow_html=True)
                
                with tabs[7]: # 専門家コメント
                    st.subheader("専門家コメント")
                    nc_ratio = indicators.get('net_cash_ratio', {}).get('value')
                    cn_per = indicators.get('cn_per', {}).get('value')
                    net_income = indicators.get('variables', {}).get('当期純利益')
                    commentary = get_kiyohara_commentary(nc_ratio, cn_per, net_income)
                    st.markdown(commentary, unsafe_allow_html=True)

                with tabs[8]: # 財務諸表(バフェットコード)
                    st.subheader("財務諸表 (バフェットコード)")
                    bc_data = result.get('buffett_code_data', {})
                    pl_data = bc_data.get('損益計算書', {})
                    bs_data = bc_data.get('貸借対照表', {})
                    if pl_data or bs_data:
                        all_items = set()
                        for period_data in pl_data.values(): all_items.update(period_data.keys())
                        for period_data in bs_data.values(): all_items.update(period_data.keys())
                        periods = sorted(list(set(pl_data.keys()) | set(bs_data.keys())), reverse=True)
                        display_df = pd.DataFrame(index=sorted(list(all_items)), columns=periods)
                        for period in periods:
                            for item in display_df.index:
                                val_pl = pl_data.get(period, {}).get(item, {}).get('display')
                                val_bs = bs_data.get(period, {}).get(item, {}).get('display')
                                display_df.loc[item, period] = val_pl or val_bs or "-"
                        st.dataframe(display_df)
                    else:
                        st.warning("財務諸表データを取得できませんでした。")
                
                with tabs[9]: # ヤフーファイナンス財務
                    st.subheader("ヤフーファイナンス財務データ")
                    yf_statements = result.get('yfinance_statements', {})
                    if yf_statements:
                        for title, df in yf_statements.items():
                            if not df.empty:
                                st.markdown(f"**{title}** (単位: 百万円)")
                                st.dataframe(df.style.format("{:,.0f}", na_rep="-"))
                            else:
                                st.markdown(f"**{title}**: データなし")
                    else:
                        st.warning("Yahoo Financeから財務データを取得できませんでした。")

    st.markdown("---") 

    st.header("時系列グラフ比較")
    metrics_to_plot = ['EPS (円)', 'EPS成長率 (対前年比) (%)', 'PER (倍)', 'PBR (倍)', 'ROE (%)', '自己資本比率 (%)', '年間1株配当 (円)', 'PEG (実績)']
    selected_metric = st.selectbox("比較する指標を選択してください", metrics_to_plot)

    visible_stocks = []
    cols_check = st.columns(4) 
    for i, key in enumerate(all_results.keys()):
        is_visible = cols_check[i % 4].checkbox(key, value=True, key=f"check_{key}")
        if is_visible:
            visible_stocks.append(key)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    color_list = plt.get_cmap('tab10').colors

    all_x_labels = set()
    for key in visible_stocks:
        res = all_results.get(key, {})
        df = res.get('timeseries_df')
        if df is not None and not df.empty and '年度' in df.columns:
            all_x_labels.update(df['年度'].dropna().tolist())
    
    if all_x_labels and visible_stocks:
        sorted_x_labels = sorted(list(all_x_labels), key=lambda x: (x == '最新', x))

        for i, key in enumerate(visible_stocks):
            res = all_results[key]
            df = res.get('timeseries_df')
            if df is not None and not df.empty:
                temp_df = df.set_index('年度')
                if selected_metric not in temp_df.columns:
                    temp_df[selected_metric] = np.nan
                
                y_values = [temp_df.loc[label, selected_metric] if label in temp_df.index and pd.notna(temp_df.loc[label, selected_metric]) else np.nan for label in sorted_x_labels]
                
                y_series = pd.Series(y_values, index=range(len(sorted_x_labels)))
                y_interpolated = y_series.interpolate(method='linear')
                
                line, = ax.plot(range(len(sorted_x_labels)), y_interpolated.values, marker='', linestyle='-', label=key, color=color_list[i % len(color_list)])
                
                for x_idx, y_val in enumerate(y_series):
                    if pd.notna(y_val):
                        ax.plot(x_idx, y_val, marker='o', color=line.get_color())
                        ax.annotate(f'{y_val:.2f}', (x_idx, y_val), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9)

        ymin, ymax = ax.get_ylim()
        if selected_metric == 'PBR (倍)': ax.axhspan(0, 1, facecolor='limegreen', alpha=0.15)
        elif selected_metric == 'PER (倍)': ax.axhspan(0, 10, facecolor='limegreen', alpha=0.15)
        elif selected_metric == 'ROE (%)': ax.axhspan(10, max(ymax, 11), facecolor='limegreen', alpha=0.15)
        elif selected_metric == 'PEG (実績)': ax.axhspan(0, 1, facecolor='limegreen', alpha=0.15)
        elif selected_metric == '自己資本比率 (%)': ax.axhspan(60, max(ymax, 11), facecolor='limegreen', alpha=0.15)
        elif selected_metric == 'EPS成長率 (対前年比) (%)': ax.axhspan(0, max(ymax, 1), facecolor='limegreen', alpha=0.15)
        ax.set_ylim(ymin, ymax) 

        ax.set_title(f"{selected_metric} の時系列比較", fontsize=16)
        ax.set_ylabel(selected_metric)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        ax.set_xticks(range(len(sorted_x_labels)))
        ax.set_xticklabels(sorted_x_labels, rotation=30, ha='right')
        st.pyplot(fig)
    else:
        st.warning("グラフを描画できる銘柄が選択されていません。")

else:
    st.info("サイドバーから銘柄コードまたは会社名を入力して「分析実行」ボタンを押してください。")