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
# import pyperclip  <- ★削除
from streamlit_copy_to_clipboard import st_copy_to_clipboard # ★追加
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
                year_full = (2000 + year) if year < 100 else year
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
            score = 0; evaluation = "【成長鈍化・赤字】" if peg_ratio is not None else "---"
        elif peg_ratio <= 0.5: score = 100; evaluation = "【超割安な成長株】"
        elif peg_ratio <= 1.0: score = self._linear_interpolate(peg_ratio, 0.5, 100, 1.0, 70); evaluation = "【割安な成長株】"
        elif peg_ratio <= 1.5: score = self._linear_interpolate(peg_ratio, 1.0, 70, 1.5, 40); evaluation = "【適正価格】"
        elif peg_ratio < 2.0: score = self._linear_interpolate(peg_ratio, 1.5, 40, 2.0, 0); evaluation = "【やや割高】"
        else: score = 0; evaluation = "【割高】"
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
        op_income = self.get_value(latest_pl_data, ['営業利益'], '営業利益')
        op_income_source = '営業利益'
        if op_income is None:
            op_income = self.get_value(latest_pl_data, ['税引前利益', '税金等調整前当期純利益'], '税引前利益(代替)'); op_income_source = '税引前利益'
        if op_income is None:
            op_income = self.get_value(latest_pl_data, ['当期純利益', '親会社株主に帰属する当期純利益'], '当期純利益(代替)'); op_income_source = '当期純利益'
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
        if total_liabilities is None:
            total_liabilities = self.get_value(latest_bs_data, ['負債'], '負債')
            if total_liabilities is not None: indicators['calc_warnings'].append("注記: NC比率計算で「負債合計」の代わりに「負債」で代用")
        indicators['variables'].update({'流動資産': current_assets, '負債合計': total_liabilities})
        nc_error = check_reqs([market_cap, current_assets, securities, total_liabilities], ["時価総額", "流動資産", "有価証券", "負債合計"])
        nc_ratio = None
        if not nc_error:
            if market_cap > 0:
                nc_ratio = (current_assets + (securities * 0.7) - total_liabilities) / (market_cap / 1_000_000)
                indicators['formulas']['ネットキャッシュ比率'] = f"({current_assets:,.0f} + {securities:,.0f}*0.7 - {total_liabilities:,.0f}) / {market_cap/1e6:,.0f}"
            else: nc_error = "時価総額がゼロです"
        cnper_error = check_reqs([pe, nc_ratio], ["PER", "ネットキャッシュ比率"])
        cn_per = None
        if not cnper_error:
            cn_per = pe * (1 - nc_ratio)
            indicators['formulas']['キャッシュニュートラルPER'] = f"{pe:.2f} * (1 - {nc_ratio:.2f})"
        tax_rate = corp_tax / pretax_income if all(v is not None for v in [corp_tax, pretax_income]) and pretax_income > 0 else 0.3062
        indicators['variables']['税率'] = tax_rate
        debt = self.get_value(latest_bs_data, ['有利子負債合計', '有利子負債'], '有利子負債')
        net_debt = self.get_value(latest_bs_data, ['純有利子負債'], '純有利子負債')
        cash = self.get_value(latest_bs_data, ['現金', '現金及び預金'], '現金同等物')
        indicators['variables'].update({'有利子負債': debt, '純有利子負債': net_debt, '現金同等物': cash})
        interest_expense = self.get_value(latest_pl_data, ['支払利息', '金融費用'], '支払利息')
        cost_of_equity = rf_rate + beta * mrp if all(v is not None for v in [beta, rf_rate, mrp]) else None
        indicators['variables']['株主資本コスト'] = cost_of_equity
        effective_debt_for_wacc = debt
        if debt is None and net_debt is not None and cash is not None:
            effective_debt_for_wacc = net_debt + cash
            if effective_debt_for_wacc < 0: effective_debt_for_wacc = 0
        cost_of_debt = interest_expense / effective_debt_for_wacc if all(v is not None for v in [interest_expense, effective_debt_for_wacc]) and effective_debt_for_wacc > 0 else 0.0
        indicators['variables']['負債コスト'] = cost_of_debt
        wacc_error = check_reqs([cost_of_equity, market_cap, effective_debt_for_wacc], ["株主資本コスト", "時価総額", "有利子負債(または代用値)"])
        wacc = None
        if not wacc_error:
            e, d_yen = market_cap, effective_debt_for_wacc * 1_000_000
            v = e + d_yen
            if v > 0:
                wacc = cost_of_equity * (e / v) + cost_of_debt * (1 - tax_rate) * (d_yen / v)
                indicators['formulas']['WACC'] = f"Ke {cost_of_equity:.2%} * (E/V {(e/v):.2%}) + Kd {cost_of_debt:.2%} * (1-T {tax_rate:.2%}) * (D/V {(d_yen/v):.2%})"
        roic_error = check_reqs([op_income, net_assets, debt], [op_income_source, "純資産", "有利子負債"])
        roic = None
        if not roic_error:
            invested_capital = net_assets + debt
            nopat = op_income * (1 - tax_rate)
            if invested_capital > 0:
                roic = nopat / invested_capital
                indicators['formulas']['ROIC'] = f"{nopat:,.0f} / {invested_capital:,.0f}"
        indicators['net_cash_ratio'] = {'value': nc_ratio, 'reason': nc_error, **self._score_net_cash_ratio(nc_ratio)}
        indicators['cn_per'] = {'value': cn_per, 'reason': cnper_error, **self._score_cn_per(cn_per, keijo_rieki, pe, trailing_eps)}
        indicators['roic'] = {'value': roic, 'reason': roic_error, **self._score_roic(roic, wacc)}
        indicators['wacc'] = {'value': wacc, 'reason': wacc_error}
        return indicators

    def get_yfinance_statements(self, ticker_obj):
        return {
            "年次損益計算書": self.format_yfinance_df(ticker_obj.financials),
            "四半期損益計算書": self.format_yfinance_df(ticker_obj.quarterly_financials),
            "年次貸借対照表": self.format_yfinance_df(ticker_obj.balance_sheet),
            "四半期貸借対照表": self.format_yfinance_df(ticker_obj.quarterly_balance_sheet),
            "年次CF計算書": self.format_yfinance_df(ticker_obj.cashflow),
            "四半期CF計算書": self.format_yfinance_df(ticker_obj.quarterly_cashflow),
        }

    def get_timeseries_financial_metrics(self, ticker_obj, info) -> pd.DataFrame:
        financials = ticker_obj.financials
        balance_sheet = ticker_obj.balance_sheet
        hist = ticker_obj.history(period="5y")
        dividends = ticker_obj.dividends
        if hasattr(hist.index.dtype, 'tz') and hist.index.dtype.tz is not None: hist.index = hist.index.tz_localize(None)
        if hasattr(dividends.index.dtype, 'tz') and dividends.index.dtype.tz is not None: dividends.index = dividends.index.tz_localize(None)
        equity_keys = ['Total Stockholder Equity', 'Stockholders Equity', 'Total Equity']; assets_keys = ['Total Assets']; shares_keys = ['Share Issued', 'Ordinary Shares Number', 'Basic Average Shares']; revenue_keys = ['Total Revenue', 'Revenues', 'Total Sales']; net_income_keys = ['Net Income', 'Net Income From Continuing Operations']; eps_keys = ['Basic EPS']
        def find_yf_value(df, keys, col):
            if df.empty or col not in df.columns: return None
            for key in keys:
                if key in df.index: return df.loc[key, col]
            return None
        metrics = []
        for date_col in financials.columns[:min(4, financials.shape[1])]:
            stockholder_equity = find_yf_value(balance_sheet, equity_keys, date_col)
            total_assets = find_yf_value(balance_sheet, assets_keys, date_col)
            net_income = find_yf_value(financials, net_income_keys, date_col)
            shares_outstanding = find_yf_value(balance_sheet, shares_keys, date_col)
            total_revenue = find_yf_value(financials, revenue_keys, date_col)
            price = hist.asof(date_col)['Close'] if not hist.empty else None
            eps = find_yf_value(financials, eps_keys, date_col)
            annual_dividends = 0
            if not dividends.empty:
                dividends_in_year = dividends[dividends.index.year == date_col.year]
                if not dividends_in_year.empty: annual_dividends = dividends_in_year.sum()
            metrics.append({
                '決算日': date_col.strftime('%Y-%m-%d'), '年度': f"{date_col.year}年度", 'EPS (円)': eps, 'PER (倍)': price / eps if all(pd.notna(v) and v != 0 for v in [price, eps]) else None,
                'PBR (倍)': price / (stockholder_equity / shares_outstanding) if all(pd.notna(v) and v != 0 for v in [price, stockholder_equity, shares_outstanding]) else None,
                'PSR (倍)': price / (total_revenue / shares_outstanding) if all(pd.notna(v) and v != 0 for v in [price, total_revenue, shares_outstanding]) else None,
                'ROE (%)': (net_income / stockholder_equity) * 100 if all(pd.notna(v) and v != 0 for v in [net_income, stockholder_equity]) else None,
                '自己資本比率 (%)': (stockholder_equity / total_assets) * 100 if all(pd.notna(v) and v > 0 for v in [stockholder_equity, total_assets]) else None,
                '年間1株配当 (円)': annual_dividends, '配当利回り (%)': (annual_dividends / price) * 100 if all(pd.notna(v) and v > 0 for v in [annual_dividends, price]) else None
            })
        latest_equity = find_yf_value(balance_sheet, equity_keys, balance_sheet.columns[0]) if not balance_sheet.empty else None
        latest_assets = find_yf_value(balance_sheet, assets_keys, balance_sheet.columns[0]) if not balance_sheet.empty else None
        metrics.append({
            '決算日': date.today().strftime('%Y-%m-%d'), '年度': '最新', 'EPS (円)': info.get('trailingEps'), 'PER (倍)': info.get('trailingPE'),
            'PBR (倍)': info.get('priceToBook'), 'PSR (倍)': info.get('priceToSalesTrailing12Months'), 'ROE (%)': info.get('returnOnEquity', 0) * 100,
            '自己資本比率 (%)': (latest_equity / latest_assets) * 100 if all(pd.notna(v) and v > 0 for v in [latest_equity, latest_assets]) else None,
            '年間1株配当 (円)': info.get('trailingAnnualDividendRate'), '配当利回り (%)': info.get('trailingAnnualDividendYield', 0) * 100
        })
        df = pd.DataFrame(metrics).set_index('決算日').sort_index(ascending=True)
        df['EPS成長率 (対前年比) (%)'] = df['EPS (円)'].pct_change(fill_method=None) * 100
        return df.sort_index(ascending=False)

    def calculate_peg_ratios(self, ticker_obj, info: dict) -> dict:
        results = {'cagr_growth': {'value': None, 'growth': None, 'reason': 'データ不足', 'eps_points': [], 'start_eps': None, 'end_eps': None, 'years': 0}, 'single_year': {'value': None, 'growth': None, 'reason': 'データ不足'}, 'historical_pegs': {}}
        try:
            current_per = info.get('trailingPE')
            if not current_per: return {**results, 'cagr_growth': {**results['cagr_growth'], 'reason': '現在のPERが取得できません'}, 'single_year': {**results['single_year'], 'reason': '現在のPERが取得できません'}}
            financials = ticker_obj.financials
            if financials.empty or 'Basic EPS' not in financials.index: return {**results, 'cagr_growth': {**results['cagr_growth'], 'reason': 'EPSデータが見つかりません'}, 'single_year': {**results['single_year'], 'reason': 'EPSデータが見つかりません'}}
            annual_eps_data = financials.loc['Basic EPS'].dropna().sort_index(ascending=False)
            if len(annual_eps_data) >= 2:
                latest_eps, prev_eps = annual_eps_data.iloc[0], annual_eps_data.iloc[1]
                if pd.notna(latest_eps) and pd.notna(prev_eps) and prev_eps > 0:
                    growth = (latest_eps - prev_eps) / prev_eps
                    results['single_year'] = {'growth': growth, 'value': current_per / (growth * 100) if growth > 0 else None, 'reason': None if growth > 0 else '単年成長率がマイナス'}
                else: results['single_year']['reason'] = 'EPSデータ欠損または前期がマイナス'
            trailing_eps = info.get('trailingEps')
            if trailing_eps is not None:
                points = [p for p in [trailing_eps] + annual_eps_data.tolist() if pd.notna(p)]
                results['cagr_growth']['eps_points'] = points
                if len(points) >= 2:
                    start_eps, end_eps, years = points[-1], points[0], len(points) - 1
                    results['cagr_growth'].update({'start_eps': start_eps, 'end_eps': end_eps, 'years': years})
                    if start_eps > 0 and end_eps > 0:
                        cagr = (end_eps / start_eps)**(1/years) - 1
                        results['cagr_growth']['growth'] = cagr
                        if cagr > 0: results['cagr_growth'].update({'value': current_per / (cagr * 100), 'reason': f'{years}年間のCAGR'})
                        else: results['cagr_growth']['reason'] = f'{years}年CAGRがマイナス'
                    else: results['cagr_growth']['reason'] = '開始または終了EPSがマイナス'
                else: results['cagr_growth']['reason'] = '有効なEPSが2地点未満'
            history = ticker_obj.history(period="6y")
            if not history.empty and len(annual_eps_data) >= 2:
                history.index = history.index.tz_localize(None)
                for i in range(len(annual_eps_data) - 1):
                    eps_curr, eps_prev, year_date = annual_eps_data.iloc[i], annual_eps_data.iloc[i+1], annual_eps_data.index[i]
                    if pd.notna(eps_curr) and pd.notna(eps_prev) and eps_prev > 0:
                        yoy_growth = (eps_curr - eps_prev) / eps_prev
                        if yoy_growth > 0:
                            price = history.asof(year_date)['Close']
                            if price: results['historical_pegs'][f"{year_date.year}年度"] = (price / eps_curr) / (yoy_growth * 100)
        except Exception as e: logger.error(f"PEGレシオ計算中にエラー: {e}", exc_info=True)
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
                        logger.info(f"銘柄 {ticker_code} の情報取得に成功しました。 ({attempt + 1}回目)"); break
                except Exception as e:
                    logger.warning(f"銘柄 {ticker_code} の情報取得に失敗 ({attempt + 1}/3回目): {e}"); time.sleep(5)
            if not info or info.get('quoteType') is None: raise ValueError("yfinanceから有効な情報を取得できませんでした。(3回試行後)")
            result['company_name'] = info.get('shortName') or info.get('longName') or f"銘柄 {ticker_code}"
            result['yf_info'] = info
            for statement, path in {"貸借対照表": "bs", "損益計算書": "pl"}.items():
                soup = self.get_html_soup(f"https://www.buffett-code.com/company/{ticker_code}/financial/{path}")
                if soup and (all_data := self.extract_all_financial_data(soup)): result['buffett_code_data'][statement] = all_data
                else: logger.warning(f"Buffett-Codeから{statement}のデータ解析に失敗。"); result['buffett_code_data'][statement] = {}
            result['scoring_indicators'] = self._calculate_scoring_indicators(result['buffett_code_data'], {**info, **options})
            result['warnings'].extend(result['scoring_indicators'].pop('calc_warnings', []))
            peg_results = self.calculate_peg_ratios(ticker_obj, info)
            result['peg_analysis'] = peg_results
            peg_score_dict = self._calculate_peg_score(peg_results['cagr_growth']['value'])
            result['scoring_indicators']['peg'] = {'value': peg_results['cagr_growth']['value'], 'reason': peg_results['cagr_growth']['reason'], **peg_score_dict}
            scores = [result['scoring_indicators'][k]['score'] for k in ['net_cash_ratio', 'cn_per', 'roic', 'peg']]
            result['final_average_score'] = sum(scores) / len(scores)
            ts_df = self.get_timeseries_financial_metrics(ticker_obj, info)
            if not ts_df.empty:
                peg_df = pd.DataFrame(peg_results['historical_pegs'].items(), columns=['年度', 'PEG (実績)'])
                ts_df = ts_df.reset_index().merge(peg_df, on='年度', how='left').set_index('決算日')
                if not (latest_index := ts_df[ts_df['年度'] == '最新'].index).empty:
                    ts_df.loc[latest_index, 'PEG (実績)'] = peg_results['single_year']['value']
            result['timeseries_df'] = ts_df
            result['yfinance_statements'] = self.get_yfinance_statements(ticker_obj)
        except Exception as e:
            logger.error(f"銘柄 {ticker_code} の分析中にエラーが発生しました: {e}", exc_info=True)
            result['error'] = f"分析中にエラーが発生しました: {e}"
            if 'company_name' not in result: result['company_name'] = f"銘柄 {ticker_code} (エラー)"
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

if 'rf_rate' not in st.session_state: st.session_state.rf_rate = 0.01
if 'rf_rate_fetched' not in st.session_state:
    with st.spinner("最新のリスクフリーレートを取得中..."):
        handler = IntegratedDataHandler()
        if (rate := handler.get_risk_free_rate()) is not None:
            st.session_state.rf_rate = rate
    st.session_state.rf_rate_fetched = True 

st.session_state.rf_rate = st.sidebar.number_input("リスクフリーレート(Rf)", value=st.session_state.rf_rate, format="%.4f")
mrp = st.sidebar.number_input("マーケットリスクプレミアム(MRP)", value=0.06, format="%.2f")
analyze_button = st.sidebar.button("分析実行")

st.title("統合型 企業価値分析ツール")

if 'results' not in st.session_state: st.session_state.results = None

def run_analysis_for_all(stocks_to_analyze, options_str):
    options = eval(options_str)
    all_results = {}
    data_handler = IntegratedDataHandler()
    for stock_info in stocks_to_analyze:
        code = stock_info['code']
        result = data_handler.perform_full_analysis(code, options)
        result['sector'] = stock_info.get('sector', '業種不明')
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
                if (stock_info := search_handler.get_ticker_info_from_query(query)):
                    target_stocks.append(stock_info)
                else:
                    not_found_queries.append(query)
        unique_target_stocks = list({stock['code']: stock for stock in target_stocks}.values())
        if not_found_queries: st.warning(f"以下の銘柄は見つかりませんでした: {', '.join(not_found_queries)}")
        if not unique_target_stocks:
            st.error("分析対象の銘柄が見つかりませんでした。入力内容を確認してください。")
            st.session_state.results = None
        else:
            display_codes = [s['code'] for s in unique_target_stocks]
            st.success(f"分析対象: {', '.join(display_codes)}")
            options = {'risk_free_rate': st.session_state.rf_rate, 'mkt_risk_premium': mrp}
            with st.spinner(f'分析中... ({len(unique_target_stocks)}件)'):
                st.session_state.results = run_analysis_for_all(unique_target_stocks, str(options))

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
            sector = result.get('sector', '')
            if sector and pd.notna(sector):
                 st.markdown(f"### {display_key} <span style='font-size: 16px; color: grey; font-weight: normal; margin-left: 10px;'>({sector})</span>", unsafe_allow_html=True)
            else: st.markdown(f"### {display_key}")
        
        with col2:
            info = result.get('yf_info', {})
            price = info.get('regularMarketPrice'); change = info.get('regularMarketChange'); previous_close = info.get('regularMarketPreviousClose')
            if isinstance(price, (int, float)) and isinstance(previous_close, (int, float)) and previous_close > 0:
                change_pct = (price - previous_close) / previous_close
            else: change_pct = info.get('regularMarketChangePercent')
            if all(isinstance(x, (int, float)) for x in [price, change, change_pct]):
                st.metric(label="現在株価", value=f"{price:,.0f} 円", delta=f"前日比 {change:+.2f}円 ({change_pct:+.2%})", delta_color="normal")
            
        with col3:
            st.write(""); st.write("")
            indicators = result.get('scoring_indicators', {}); peg_data = indicators.get('peg', {}); nc_data = indicators.get('net_cash_ratio', {}); cnper_data = indicators.get('cn_per', {}); roic_data = indicators.get('roic', {})
            def format_for_copy(data):
                val = data.get('value'); return f"{val:.2f} ({data.get('evaluation', '')})" if val is not None else "N/A"
            change_pct_text = f"({change_pct:+.2%})" if isinstance(change_pct, (int, float)) else ""
            price_text = f"株価: {price:,.0f}円 (前日比 {change:+.2f}円, {change_pct_text})" if all(isinstance(x, (int, float)) for x in [price, change]) else ""
            copy_text = (f"■ {display_key}\n{price_text}\n総合スコア: {score:.1f}点 {stars_text}\n"
                         f"--------------------\nPEG (CAGR): {format_for_copy(peg_data)}\n"
                         f"ネットキャッシュ比率: {format_for_copy(nc_data)}\nCN-PER: {format_for_copy(cnper_data)}\n"
                         f"ROIC: {format_for_copy(roic_data)}")
            
            # ★★★ 修正点：クリップボードボタンを新しいコンポーネントに置き換え ★★★
            st_copy_to_clipboard(copy_text, "📋 結果をコピー", key=f"copy_{display_key}")

        st.markdown(f"#### 総合スコア: <span style='font-size: 28px; font-weight: bold; color: {score_color};'>{score_text}点</span> <span style='font-size: 32px;'>{stars_text}</span>", unsafe_allow_html=True)
        if result.get('warnings'): st.info(f"注記: {'; '.join(list(set(result.get('warnings',[]))))}。詳細は各計算タブを確認してください。")

        with st.container():
            cols = st.columns(4)
            def show_metric(column, title, data, warnings):
                with column:
                    note = ""
                    if title == "PEG (CAGR)" and any("PEG" in w for w in warnings): note = " *"
                    if title == "ネットキャッシュ比率" and any(k in w for w in warnings for k in ["NC比率", "負債", "有価証券"]): note = " *"
                    if title == "CN-PER" and any(k in w for w in warnings for k in ["NC比率", "負債", "有価証券"]): note = " *"
                    if title == "ROIC" and any("ROIC" in w for w in warnings): note = " *"
                    val = data.get('value'); val_str = f"{val:.2f}" if val is not None else "N/A"
                    score_val = data.get('score', 0); color = "#28a745" if score_val >= 70 else "#ffc107" if score_val >= 40 else "#dc3545"
                    st.markdown(f"<p style='font-size: 14px; color: #555; font-weight: bold; text-align: center; margin-bottom: 0;'>{title}{note}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 28px; color: {color}; font-weight: bold; text-align: center; margin: 0;'>{val_str}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; font-weight: bold; font-size: 14px;'>スコア: <span style='color:{color};'>{score_val:.1f}点</span></p>", unsafe_allow_html=True)
                    if val is None: st.markdown(f"<p style='text-align: center; font-size: 12px; color: red;'>({data.get('reason', '計算不能')})</p>", unsafe_allow_html=True)
                    else: st.markdown(f"<p style='text-align: center; font-size: 12px; color: #777;'>{data.get('evaluation', '---')}</p>", unsafe_allow_html=True)
            show_metric(cols[0], "PEG (CAGR)", indicators.get('peg', {}), result.get('warnings', []))
            show_metric(cols[1], "ネットキャッシュ比率", indicators.get('net_cash_ratio', {}), result.get('warnings', []))
            show_metric(cols[2], "CN-PER", indicators.get('cn_per', {}), result.get('warnings', []))
            show_metric(cols[3], "ROIC", indicators.get('roic', {}), result.get('warnings', []))
            
            with st.expander("詳細データを見る"):
                tabs = st.tabs(["時系列指標", "PEG計算", "NC比率計算", "CN-PER計算", "ROIC計算", "WACC計算", "PEGレシオコメント", "専門家コメント", "財務諸表(バフェットコード)", "ヤフーファイナンス財務"])
                with tabs[0]:
                    if (ts_df := result.get('timeseries_df')) is not None and not ts_df.empty:
                        display_columns = ['年度', 'EPS (円)', 'EPS成長率 (対前年比) (%)', 'PER (倍)', 'PBR (倍)', 'PEG (実績)', 'PSR (倍)', 'ROE (%)', '自己資本比率 (%)', '年間1株配当 (円)', '配当利回り (%)']
                        df_to_display = ts_df.copy().reset_index()
                        existing_cols = [col for col in display_columns if col in df_to_display.columns]
                        st.dataframe(df_to_display[['決算日'] + existing_cols].style.format({col: "{:.2f}" for col in df_to_display.select_dtypes(include=np.number).columns}, na_rep="-"))
                    else: st.warning("時系列データを取得できませんでした。")
                with tabs[1]:
                    st.subheader("PEG (CAGR) の計算過程")
                    peg_data = result.get('peg_analysis', {}).get('cagr_growth', {})
                    st.markdown(f"**計算式:** `PER / (EPSのCAGR * 100)`")
                    per_val = indicators.get('variables', {}).get('PER (実績)')
                    if peg_data.get('value') is not None and isinstance(per_val, (int, float)):
                        st.text(f"PER {per_val:.2f} / (CAGR {peg_data.get('growth', 0)*100:.2f} %) = {peg_data.get('value'):.2f}")
                        st.markdown(f"**CAGR ({peg_data.get('years', 'N/A')}年) 計算:** `(最終EPS / 初期EPS) ** (1 / 年数) - 1`")
                        if all(isinstance(x, (int, float)) for x in [peg_data.get('end_eps'), peg_data.get('start_eps'), peg_data.get('years')]) and peg_data.get('years', 0) > 0:
                            st.text(f"({peg_data['end_eps']:.2f} / {peg_data['start_eps']:.2f}) ** (1 / {peg_data['years']}) - 1 = {peg_data.get('growth', 0):.4f}")
                    else: st.error(f"計算不能。理由: {peg_data.get('reason', '不明')}")
                    st.markdown("**計算に使用したEPSデータ (新しい順):**")
                    if (eps_points := peg_data.get('eps_points', [])): st.text(str([f"{p:.2f}" if isinstance(p, (int, float)) else "N/A" for p in eps_points]))
                    else: st.warning('EPSデータが不足しています。')
                with tabs[2]:
                    st.subheader("清原式ネットキャッシュ比率の計算過程")
                    if (nc_warnings := [w for w in result.get('warnings', []) if any(k in w for k in ["NC比率", "純有利子負債", "有価証券", "負債"])]): st.info(" ".join(list(set(nc_warnings))))
                    st.markdown(f"**計算式:** `(流動資産 + 有価証券*0.7 - 負債合計) / 時価総額`")
                    st.text(indicators.get('formulas', {}).get('ネットキャッシュ比率', indicators.get('net_cash_ratio', {}).get('reason')))
                    st.json({k: (f"{v:,.0f} 百万円" if isinstance(v, (int, float)) else "N/A") for k, v in {"流動資産": indicators.get('variables', {}).get('流動資産'), "有価証券": indicators.get('variables', {}).get('有価証券'), "負債合計": indicators.get('variables', {}).get('負債合計'), "時価総額": indicators.get('variables', {}).get('時価総額', 0)/1e6}.items()})
                with tabs[3]:
                    st.subheader("CN-PERの計算過程")
                    st.markdown(f"**計算式:** `実績PER * (1 - ネットキャッシュ比率)`")
                    st.text(indicators.get('formulas', {}).get('キャッシュニュートラルPER', indicators.get('cn_per', {}).get('reason')))
                    st.json({"実績PER": f"{v:.2f} 倍" if isinstance(v := indicators.get('variables', {}).get('PER (実績)'), (int, float)) else "N/A (データなし)", "ネットキャッシュ比率": f"{v2:.2f}" if isinstance(v2 := indicators.get('net_cash_ratio', {}).get('value'), (int, float)) else f"N/A ({indicators.get('net_cash_ratio', {}).get('reason')})"})
                with tabs[4]:
                    st.subheader("ROICの計算過程")
                    if (roic_warnings := [w for w in result.get('warnings', []) if "ROIC" in w]): st.info(" ".join(list(set(roic_warnings))))
                    st.markdown(f"**計算式:** `NOPAT (税引後営業利益) / 投下資本 (純資産 + 有利子負債)`")
                    st.text(indicators.get('formulas', {}).get('ROIC', indicators.get('roic', {}).get('reason')))
                    def format_val(v, is_curr=True, is_pct=False):
                        if isinstance(v, (int, float)): return f"{v:.2%}" if is_pct else f"{v:,.0f} 百万円" if is_curr else f"{v}"
                        return "N/A"
                    st.json({"NOPAT計算用利益": format_val(indicators.get('variables', {}).get(f"NOPAT計算用利益 ({indicators.get('roic_source_key', '')})")), "税率": format_val(indicators.get('variables', {}).get('税率'), is_pct=True), "純資産": format_val(indicators.get('variables', {}).get('純資産')), "有利子負債": format_val(indicators.get('variables', {}).get('有利子負債'))})
                with tabs[5]:
                    st.subheader("WACC (加重平均資本コスト) の計算過程")
                    if (wacc_warnings := [w for w in result.get('warnings', []) if "β値" in w]): st.info(" ".join(list(set(wacc_warnings))))
                    st.markdown(f"**計算式:** `株主資本コスト * 自己資本比率 + 負債コスト * (1 - 税率) * 負債比率`")
                    st.text(indicators.get('formulas', {}).get('WACC', indicators.get('wacc', {}).get('reason')))
                    st.json({"WACC計算結果": format_val(indicators.get('wacc', {}).get('value'), is_pct=True), "株主資本コスト (Ke)": format_val(indicators.get('variables', {}).get('株主資本コスト'), is_pct=True), "負債コスト (Kd)": format_val(indicators.get('variables', {}).get('負債コスト'), is_pct=True), "税率": format_val(indicators.get('variables', {}).get('税率'), is_pct=True), "ベータ値": f"{v:.2f}" if isinstance(v := indicators.get('variables', {}).get('ベータ値'), (int,float)) else "N/A", "リスクフリーレート": f"{st.session_state.rf_rate:.2%}", "マーケットリスクプレミアム": f"{mrp:.2%}"})
                with tabs[6]: st.subheader("PEGレシオに基づく投資家コメント (リンチ / クレイマー)"); st.markdown(get_peg_investor_commentary(indicators.get('peg', {}).get('value')), unsafe_allow_html=True)
                with tabs[7]: st.subheader("専門家コメント"); st.markdown(get_kiyohara_commentary(indicators.get('net_cash_ratio', {}).get('value'), indicators.get('cn_per', {}).get('value'), indicators.get('variables', {}).get('当期純利益')), unsafe_allow_html=True)
                with tabs[8]:
                    st.subheader("財務諸表 (バフェットコード)")
                    bc_data = result.get('buffett_code_data', {})
                    if (pl_data := bc_data.get('損益計算書', {})) or (bs_data := bc_data.get('貸借対照表', {})):
                        all_items = set(key for d in pl_data.values() for key in d) | set(key for d in bs_data.values() for key in d)
                        periods = sorted(list(set(pl_data.keys()) | set(bs_data.keys())), reverse=True)
                        display_df = pd.DataFrame(index=sorted(list(all_items)), columns=periods)
                        for period in periods:
                            for item in display_df.index:
                                display_df.loc[item, period] = pl_data.get(period, {}).get(item, {}).get('display') or bs_data.get(period, {}).get(item, {}).get('display') or "-"
                        st.dataframe(display_df)
                    else: st.warning("財務諸表データを取得できませんでした。")
                with tabs[9]:
                    st.subheader("ヤフーファイナンス財務データ")
                    if (yf_statements := result.get('yfinance_statements', {})):
                        for title, df in yf_statements.items():
                            if not df.empty:
                                st.markdown(f"**{title}** (単位: 百万円)"); st.dataframe(df.style.format("{:,.0f}", na_rep="-"))
                            else: st.markdown(f"**{title}**: データなし")
                    else: st.warning("Yahoo Financeから財務データを取得できませんでした。")

    st.markdown("---") 

    st.header("時系列グラフ比較")
    metrics_to_plot = ['EPS (円)', 'EPS成長率 (対前年比) (%)', 'PER (倍)', 'PBR (倍)', 'ROE (%)', '自己資本比率 (%)', '年間1株配当 (円)', 'PEG (実績)']
    selected_metric = st.selectbox("比較する指標を選択してください", metrics_to_plot)
    visible_stocks = [key for i, key in enumerate(all_results.keys()) if st.columns(4)[i % 4].checkbox(key, value=True, key=f"check_{key}")]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    color_list = plt.get_cmap('tab10').colors
    all_x_labels = sorted(list(set(label for key in visible_stocks for label in all_results[key].get('timeseries_df', pd.DataFrame(columns=['年度']))['年度'].dropna().tolist())), key=lambda x: (x == '最新', x))

    if all_x_labels and visible_stocks:
        for i, key in enumerate(visible_stocks):
            df = all_results[key].get('timeseries_df')
            if df is not None and not df.empty:
                temp_df = df.set_index('年度')
                y_values = pd.Series([temp_df.loc[label, selected_metric] if label in temp_df.index and pd.notna(temp_df.loc[label, selected_metric]) else np.nan for label in all_x_labels], index=range(len(all_x_labels)))
                y_interpolated = y_values.interpolate(method='linear')
                line, = ax.plot(range(len(all_x_labels)), y_interpolated.values, marker='', linestyle='-', label=key, color=color_list[i % len(color_list)])
                for x_idx, y_val in enumerate(y_values):
                    if pd.notna(y_val):
                        ax.plot(x_idx, y_val, marker='o', color=line.get_color())
                        ax.annotate(f'{y_val:.2f}', (x_idx, y_val), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=9)
        ymin, ymax = ax.get_ylim()
        span_options = {'PBR (倍)': (0, 1), 'PER (倍)': (0, 10), 'ROE (%)': (10, max(ymax, 11)), 'PEG (実績)': (0, 1), '自己資本比率 (%)': (60, max(ymax, 11)), 'EPS成長率 (対前年比) (%)': (0, max(ymax, 1))}
        if selected_metric in span_options: ax.axhspan(*span_options[selected_metric], facecolor='limegreen', alpha=0.15)
        ax.set_ylim(ymin, ymax) 
        ax.set_title(f"{selected_metric} の時系列比較", fontsize=16); ax.set_ylabel(selected_metric); ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.legend()
        ax.set_xticks(range(len(all_x_labels))); ax.set_xticklabels(all_x_labels, rotation=30, ha='right')
        st.pyplot(fig)
    else: st.warning("グラフを描画できる銘柄が選択されていません。")
else: st.info("サイドバーから銘柄コードまたは会社名を入力して「分析実行」ボタンを押してください。")