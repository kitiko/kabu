# ▼▼▼▼▼ ここからデバッグコードを追加 ▼▼▼▼▼
import subprocess
import sys
import streamlit as st

# Streamlit Cloudのログに、インストールされているライブラリの一覧を表示させる
try:
    st.write("### インストール済みライブラリ一覧")
    result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, check=True)
    st.code(result.stdout)
except subprocess.CalledProcessError as e:
    st.error(f"pip listの実行中にエラーが発生しました: Return code {e.returncode}")
    st.code(e.stderr)
except Exception as e:
    st.error(f"pip listの実行中に予期せぬエラーが発生しました: {e}")
# ▲▲▲▲▲ ここまでデバッグコードを追加 ▲▲▲▲▲


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
from streamlit_copy_to_clipboard import st_copy_to_clipboard
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
        if df.shape[1] < 6:
            st.error(f"銘柄リストファイル({JPX_STOCK_LIST_PATH})の形式が想定と異なります。業種区分を含む列数が不足しています。")
            return pd.DataFrame(columns=['code', 'name', 'market', 'sector', 'normalized_name'])

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
        """銘柄コードまたは会社名から銘柄コードや業種を辞書で返す"""
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
            stock_data = prime_matches.iloc[0] if not prime_matches.empty else matches.iloc[0]
            logger.info(f"検索クエリ '{query}' から銘柄 '{stock_data['name']} ({stock_data['code']})' を見つけました。")
            return stock_data.to_dict()
        logger.warning(f"検索クエリ '{query}' に一致する銘柄が見つかりませんでした。")
        return None

    YFINANCE_TRANSLATION_MAP = {
        'Total Revenue': '売上高', 'Revenue': '売上高', 'Operating Income': '営業利益', 'Operating Expense': '営業費用',
        'Cost Of Revenue': '売上原価', 'Gross Profit': '売上総利益', 'Selling General And Administration': '販売費及び一般管理費',
        'Research And Development': '研究開発費', 'Pretax Income': '税引前利益', 'Tax Provision': '法人税',
        'Net Income': '当期純利益', 'Net Income Common Stockholders': '親会社株主に帰属する当期純利益',
        'Basic EPS': '1株当たり利益 (EPS)', 'Diluted EPS': '希薄化後EPS', 'Total Assets': '総資産', 'Current Assets': '流動資産',
        'Cash And Cash Equivalents': '現金及び現金同等物', 'Cash': '現金', 'Receivables': '売上債権', 'Inventory': '棚卸資産',
        'Total Non Current Assets': '固定資産', 'Net PPE': '有形固定資産', 'Goodwill And Other Intangible Assets': 'のれん及びその他無形固定資産',
        'Total Liabilities Net Minority Interest': '負債合計', 'Current Liabilities': '流動負債',
        'Payables And Accrued Expenses': '支払手形及び買掛金', 'Current Debt': '短期有利子負債',
        'Total Non Current Liabilities Net Minority Interest': '固定負債', 'Long Term Debt': '長期有利子負債',
        'Total Equity Gross Minority Interest': '純資産合計', 'Stockholders Equity': '株主資本', 'Retained Earnings': '利益剰余金',
        'Cash Flow From Continuing Operating Activities': '営業キャッシュフロー', 'Cash Flow From Continuing Investing Activities': '投資キャッシュフロー',
        'Cash Flow From Continuing Financing Activities': '財務キャッシュフロー', 'Net Change In Cash': '現金の増減額', 'Free Cash Flow': 'フリーキャッシュフロー',
    }

    def get_html_soup(self, url: str) -> BeautifulSoup | None:
        logger.info(f"URLへのアクセスを開始: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7', 'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8'}
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
                if (yield_div := soup.find('div', attrs={'data-test': 'instrument-price-last'})):
                    return float(yield_div.get_text(strip=True)) / 100
            except Exception as e:
                logger.error(f"リスクフリーレートの解析に失敗: {e}")
        logger.warning("リスクフリーレートの自動取得に失敗しました。")
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
        if not (financial_table := soup.find('table', class_='financial-table')): return None
        if not (thead := financial_table.find('thead')) or not (tbody := financial_table.find('tbody')): return None
        period_headers = thead.find('tr').find_all('th')
        if len(period_headers) <= 1: return None
        valid_periods = [{'name': th.text.strip(), 'index': i + 1} for i, th in enumerate(period_headers[1:]) if th.text.strip() and "E" not in th.text.strip().upper() and "C" not in th.text.strip().upper()]
        if not valid_periods: return None
        all_periods_data = OrderedDict()
        for row in tbody.find_all('tr'):
            cells = row.find_all(['th', 'td'])
            if not (item_name := cells[0].text.strip()) or not re.search(r'[a-zA-Z\u3040-\u30FF\u4E00-\u9FFF]', item_name): continue
            for period in valid_periods:
                if (period_name := period['name']) not in all_periods_data: all_periods_data[period_name] = {}
                if len(cells) > period['index'] and (display_value := cells[period['index']].get_text(strip=True)) not in ['-', '---', '']:
                    all_periods_data[period_name][item_name] = {'display': display_value, 'raw': self.parse_financial_value(display_value)}
        return all_periods_data

    def get_latest_financial_data(self, financial_data_dict: dict) -> dict:
        latest_year, latest_month, latest_data = -1, -1, {}
        if not financial_data_dict: return {}
        for period_name, data in financial_data_dict.items():
            if (match := re.search(r'(\d{2,4})[./](\d{1,2})', period_name)):
                year_str, month_str = match.groups()
                year_full = 2000 + int(year_str) if int(year_str) < 100 else int(year_str)
                month = int(month_str)
                if year_full > latest_year or (year_full == latest_year and month > latest_month):
                    latest_year, latest_month, latest_data = year_full, month, data
        return latest_data

    def get_value(self, data_dict: dict, keys: list[str], log_name: str) -> any:
        for key in keys:
            if key in data_dict and (value := data_dict[key].get('raw')) is not None:
                logger.info(f"✅ {log_name}: 項目 '{key}' から値 ({value}) を取得しました。")
                return value
        logger.warning(f"⚠️ {log_name}: 項目が見つかりませんでした (試行キー: {keys})")
        return None

    def format_yfinance_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        df_copy = df.copy().rename(index=self.YFINANCE_TRANSLATION_MAP)
        df_copy = df_copy.loc[df_copy.index.isin(self.YFINANCE_TRANSLATION_MAP.values())]
        df_copy.columns = [f"{col.year}.{col.month}" for col in df_copy.columns]
        exclude_rows = [name for name in df_copy.index if any(s in name for s in ['EPS', '比率', 'Rate'])]
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
        if not (keijo_rieki and keijo_rieki > 0): return {'score': 10, 'evaluation': '【赤字・要注意】'}
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
            return {'score': self._linear_interpolate(roic_percent, 0, 40, wacc * 100, 60), 'evaluation': '【価値破壊】'}
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

    def _calculate_scoring_indicators(self, all_fin_data: dict, yf_data: dict) -> dict:
        indicators = {'calc_warnings': [], 'formulas': {}, 'variables': {}}
        latest_bs_data = self.get_latest_financial_data(all_fin_data.get('貸借対照表', {}))
        latest_pl_data = self.get_latest_financial_data(all_fin_data.get('損益計算書', {}))
        market_cap, pe, rf_rate, mrp, trailing_eps = (yf_data.get(k) for k in ['marketCap', 'trailingPE', 'risk_free_rate', 'mkt_risk_premium', 'trailingEps'])
        beta = yf_data.get('beta')
        indicators['variables'].update({'時価総額': market_cap, 'PER (実績)': pe, 'ベータ値': beta})
        if beta is None: beta = 1.0; indicators['calc_warnings'].append("注記: β値の代わりに1.0で代用")
        
        securities = self.get_value(latest_bs_data, ['有価証券', '投資有価証券', 'その他の金融資産'], '有価証券')
        if securities is not None and securities < 0: securities = 0; indicators['calc_warnings'].append("注記: 有価証券がマイナスだったため0として計算")
        if securities is None: securities = 0; indicators['calc_warnings'].append("注記: 有価証券が見つからないため0として計算")
        indicators['variables']['有価証券'] = securities
        
        op_income_source = '営業利益'
        op_income = self.get_value(latest_pl_data, ['営業利益'], '営業利益')
        if op_income is None: op_income_source = '税引前利益'; op_income = self.get_value(latest_pl_data, ['税引前利益', '税金等調整前当期純利益'], '税引前利益(代替)')
        if op_income is None: op_income_source = '当期純利益'; op_income = self.get_value(latest_pl_data, ['当期純利益', '親会社株主に帰属する当期純利益'], '当期純利益(代替)')
        indicators['roic_source_key'] = op_income_source
        if op_income_source != '営業利益' and op_income is not None: indicators['calc_warnings'].append(f"信頼性警告: 営業利益の代わりに「{op_income_source}」を使用")
        indicators['variables'][f'NOPAT計算用利益 ({op_income_source})'] = op_income
        
        net_assets = self.get_value(latest_bs_data, ['純資産合計', '純資産'], '純資産')
        corp_tax = self.get_value(latest_pl_data, ['法人税等', '法人税、住民税及び事業税'], '法人税等')
        pretax_income = self.get_value(latest_pl_data, ['税引前利益', '税金等調整前当期純利益'], '税引前利益')
        keijo_rieki = self.get_value(latest_pl_data, ['経常利益'], '経常利益')
        net_income = self.get_value(latest_pl_data, ['当期純利益', '親会社株主に帰属する当期純利益'], '当期純利益')
        indicators['variables'].update({'純資産': net_assets, '経常利益': keijo_rieki, '当期純利益': net_income})
        
        def check_reqs(reqs, names):
            missing = [name for req, name in zip(reqs, names) if req is None]
            return None if not missing else f"不足: {', '.join(missing)}"
        
        current_assets = self.get_value(latest_bs_data, ['流動資産合計', '流動資産'], '流動資産')
        total_liabilities = self.get_value(latest_bs_data, ['負債合計'], '負債')
        if total_liabilities is None and (total_liabilities := self.get_value(latest_bs_data, ['負債'], '負債')) is not None:
            indicators['calc_warnings'].append("注記: NC比率計算で「負債合計」の代わりに「負債」で代用")
        indicators['variables'].update({'流動資産': current_assets, '負債合計': total_liabilities})
        
        nc_ratio, nc_error = None, check_reqs([market_cap, current_assets, securities, total_liabilities], ["時価総額", "流動資産", "有価証券", "負債合計"])
        if not nc_error:
            if market_cap > 0:
                nc_ratio = (current_assets + (securities * 0.7) - total_liabilities) / (market_cap / 1_000_000)
                indicators['formulas']['ネットキャッシュ比率'] = f"({current_assets:,.0f} + {securities:,.0f}*0.7 - {total_liabilities:,.0f}) / {market_cap/1e6:,.0f}"
            else: nc_error = "時価総額がゼロです"
        
        cn_per, cnper_error = None, check_reqs([pe, nc_ratio], ["PER", "ネットキャッシュ比率"])
        if not cnper_error: cn_per = pe * (1 - nc_ratio); indicators['formulas']['キャッシュニュートラルPER'] = f"{pe:.2f} * (1 - {nc_ratio:.2f})"
        
        tax_rate = corp_tax / pretax_income if all(v is not None and v > 0 for v in [corp_tax, pretax_income]) else 0.3062
        indicators['variables']['税率'] = tax_rate
        
        debt = self.get_value(latest_bs_data, ['有利子負債合計', '有利子負債'], '有利子負債')
        net_debt = self.get_value(latest_bs_data, ['純有利子負債'], '純有利子負債')
        cash = self.get_value(latest_bs_data, ['現金', '現金及び預金'], '現金同等物')
        indicators['variables'].update({'有利子負債': debt, '純有利子負債': net_debt, '現金同等物': cash})
        
        interest_expense = self.get_value(latest_pl_data, ['支払利息', '金融費用'], '支払利息')
        cost_of_equity = rf_rate + beta * mrp if all(v is not None for v in [beta, rf_rate, mrp]) else None
        indicators['variables']['株主資本コスト'] = cost_of_equity
        
        effective_debt = debt
        if debt is None and net_debt is not None and cash is not None: effective_debt = max(0, net_debt + cash)
        
        cost_of_debt = interest_expense / effective_debt if all(v is not None and v > 0 for v in [interest_expense, effective_debt]) else 0.0
        indicators['variables']['負債コスト'] = cost_of_debt
        
        wacc, wacc_error = None, check_reqs([cost_of_equity, market_cap, effective_debt], ["株主資本コスト", "時価総額", "有利子負債(または代用値)"])
        if not wacc_error and (v := market_cap + effective_debt * 1_000_000) > 0:
            e, d_yen = market_cap, effective_debt * 1_000_000
            wacc = cost_of_equity * (e / v) + cost_of_debt * (1 - tax_rate) * (d_yen / v)
            indicators['formulas']['WACC'] = f"Ke {cost_of_equity:.2%} * (E/V {(e/v):.2%}) + Kd {cost_of_debt:.2%} * (1-T {tax_rate:.2%}) * (D/V {(d_yen/v):.2%})"

        roic, roic_error = None, check_reqs([op_income, net_assets, debt], [op_income_source, "純資産", "有利子負債"])
        if not roic_error and (invested_capital := net_assets + debt) > 0:
            nopat = op_income * (1 - tax_rate)
            roic = nopat / invested_capital
            indicators['formulas']['ROIC'] = f"{nopat:,.0f} / {invested_capital:,.0f}"

        indicators['net_cash_ratio'] = {'value': nc_ratio, 'reason': nc_error, **self._score_net_cash_ratio(nc_ratio)}
        indicators['cn_per'] = {'value': cn_per, 'reason': cnper_error, **self._score_cn_per(cn_per, keijo_rieki, pe, trailing_eps)}
        indicators['roic'] = {'value': roic, 'reason': roic_error, **self._score_roic(roic, wacc)}
        indicators['wacc'] = {'value': wacc, 'reason': wacc_error}
        return indicators

    # ... (以降の IntegratedDataHandler クラスのメソッドは変更なし)
    
# ==============================================================================
# 4. GUIアプリケーション (Streamlit)
# ==============================================================================

# --- UI Helper Functions ---
# (get_recommendation, get_peg_investor_commentary, get_kiyohara_commentary は変更なし)

# --- Main App ---
# (メインのアプリ部分も変更なし)