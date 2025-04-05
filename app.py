import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import time
from openai import OpenAI
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
from typing import Dict, List, Any, Tuple, Optional
import json
import concurrent.futures
import os
import schedule
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta


# 设置页面配置
st.set_page_config(
    page_title="分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API配置
OPENAI_API_KEY = "xxxxxxx"  # 更新为自己在tu-zi中的API
BINANCE_API_URL = "https://api-gcp.binance.com"  # 更新为官方推荐的现货API端点
BINANCE_FUTURES_URL = "https://fapi.binance.com"

client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.tu-zi.com/v1")

# 时间周期配置
TIMEFRAMES = {
    "5m": {"interval": "5m", "name": "5分钟", "weight": 0.1},
    "15m": {"interval": "15m", "name": "15分钟", "weight": 0.15},
    "1h": {"interval": "1h", "name": "1小时", "weight": 0.25},
    "4h": {"interval": "4h", "name": "4小时", "weight": 0.25},
    "1d": {"interval": "1d", "name": "日线", "weight": 0.25}
}

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加自定义CSS样式
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    div.stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .metric-container {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .report-container {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# 费率监控常量
UPDATE_INTERVAL = 10  # 数据更新间隔（秒）
MAX_DATA_POINTS = 240  # 最大数据点数量 (4小时 = 240分钟)
HOURS_TO_DISPLAY = 4  # 显示过去多少小时的数据
STATS_FILE = "funding_rates_stats.json"  # 统计数据文件
class RateLimiter:
    """请求频率限制器"""
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            self.requests = [req for req in self.requests if now - req < self.time_window]
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.time_window - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.requests.append(now)

def fetch_data(url: str, params: dict = None, retries: int = 3) -> dict:
    """通用数据获取函数，带重试机制"""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)  # 设置10秒超时
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                logger.error(f"请求最终失败: {url}")
                return None

def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> tuple:
    """计算支撑和阻力位"""
    rolling_low = df['low'].rolling(window=window).min()
    rolling_high = df['high'].rolling(window=window).max()
    support = rolling_low.iloc[-1]
    resistance = rolling_high.iloc[-1]
    return support, resistance

def get_binance_klines(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
    """获取币安K线数据"""
    url = f"{BINANCE_API_URL}/api/v3/klines"  # 使用 /api/v3/klines 端点
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        data = fetch_data(url, params)
        if not data or not isinstance(data, list):
            logger.error(f"获取K线数据失败: 无效响应数据 - {symbol}, {interval}")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"K线数据缺少必要列: {df.columns.tolist()}")
            return pd.DataFrame()

        df[required_columns] = df[required_columns].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"处理K线数据时发生错误: {str(e)}")
        return pd.DataFrame()

# 费率监控相关函数
def get_spot_price(symbol):
    """获取现货价格"""
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "price" in data:
            return float(data["price"])
        else:
            logger.error(f"无法获取现货价格: {data}")
            return None
    except Exception as e:
        logger.error(f"获取现货价格时出错: {e}")
        return None

def get_futures_price(symbol):
    """获取期货价格"""
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "price" in data:
            return float(data["price"])
        else:
            logger.error(f"无法获取期货价格: {data}")
            return None
    except Exception as e:
        logger.error(f"获取期货价格时出错: {e}")
        return None

def get_funding_rate(symbol):
    """获取资金费率"""
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "lastFundingRate" in data:
            return float(data["lastFundingRate"])
        else:
            logger.error(f"无法获取资金费率: {data}")
            return None
    except Exception as e:
        logger.error(f"获取资金费率时出错: {e}")
        return None

def get_open_interest(symbol):
    """获取持仓量"""
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "openInterest" in data:
            return float(data["openInterest"])
        else:
            logger.error(f"无法获取持仓量: {data}")
            return None
    except Exception as e:
        logger.error(f"获取持仓量时出错: {e}")
        return None

def get_historical_klines(symbol, interval, limit):
    """获取历史K线数据"""
    try:
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(hours=HOURS_TO_DISPLAY)).timestamp() * 1000)

        spot_url = "https://api.binance.com/api/v3/klines"
        spot_params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        spot_response = requests.get(spot_url, params=spot_params)
        spot_response.raise_for_status()
        spot_data = spot_response.json()

        futures_url = "https://fapi.binance.com/fapi/v1/klines"
        futures_params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        futures_response = requests.get(futures_url, params=futures_params)
        futures_response.raise_for_status()
        futures_data = futures_response.json()

        historical_timestamps = []
        historical_spot_prices = []
        historical_futures_prices = []
        historical_premiums = []

        min_length = min(len(spot_data), len(futures_data))
        for i in range(min_length):
            timestamp = datetime.fromtimestamp(spot_data[i][0] / 1000, tz=timezone.utc)
            spot_close = float(spot_data[i][4])
            futures_close = float(futures_data[i][4])
            premium = (futures_close - spot_close) / spot_close * 100

            historical_timestamps.append(timestamp)
            historical_spot_prices.append(spot_close)
            historical_futures_prices.append(futures_close)
            historical_premiums.append(premium)

        return historical_timestamps, historical_spot_prices, historical_futures_prices, historical_premiums
    except Exception as e:
        logger.error(f"获取历史K线数据时出错: {e}")
        return [], [], [], []

def get_historical_funding_rates(symbol, limit=MAX_DATA_POINTS):
    """获取历史资金费率数据"""
    try:
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(hours=HOURS_TO_DISPLAY)).timestamp() * 1000)

        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        timestamps = []
        funding_rates = []
        for item in data:
            timestamps.append(datetime.fromtimestamp(item["fundingTime"] / 1000, tz=timezone.utc))
            funding_rates.append(float(item["fundingRate"]) * 100)
        return timestamps, funding_rates
    except Exception as e:
        logger.error(f"获取历史资金费率数据时出错: {e}")
        return [], []

def get_historical_open_interest(symbol, period="5m", limit=MAX_DATA_POINTS):
    """获取历史持仓量数据"""
    try:
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(hours=HOURS_TO_DISPLAY)).timestamp() * 1000)

        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        timestamps = []
        open_interests = []
        for item in data:
            timestamps.append(datetime.fromtimestamp(item["timestamp"] / 1000, tz=timezone.utc))
            open_interests.append(float(item["sumOpenInterest"]))
        return timestamps, open_interests
    except Exception as e:
        logger.error(f"获取历史持仓量数据时出错: {e}")
        return [], []
class BinanceFuturesAnalyzer:
    """币安期货分析器"""
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.rate_limiter = RateLimiter(max_requests=5, time_window=1.0)

    def get_usdt_symbols(self) -> List[str]:
        """获取所有USDT合约交易对"""
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return [symbol['symbol'] for symbol in data['symbols']
                    if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING']
        except Exception as e:
            logger.error(f"获取交易对失败: {e}")
            return []

    def get_open_interest(self, symbol: str, period: str = "5m") -> Dict:
        """获取单个交易对的持仓量数据"""
        self.rate_limiter.acquire()
        url = f"{self.base_url}/futures/data/openInterestHist"
        params = {
            "symbol": symbol,
            "period": period,
            "limit": 500
        }
        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return {"symbol": symbol, "data": data}
        except Exception as e:
            logger.error(f"获取{symbol}持仓数据失败: {e}")
            return None

    def analyze_positions(self) -> pd.DataFrame:
        """分析所有交易对的持仓情况"""
        symbols = self.get_usdt_symbols()
        historical_data = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self.get_open_interest, symbol): symbol
                for symbol in symbols
            }
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result and result['data']:
                        historical_data[symbol] = result['data']
                except Exception as e:
                    logger.error(f"处理{symbol}数据失败: {e}")

        analysis_results = []
        for symbol, data in historical_data.items():
            if not data:
                continue
            df = pd.DataFrame(data)
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
            except Exception as e:
                logger.error(f"数据处理失败: {e}")
                continue

            current_oi = float(df['sumOpenInterest'].iloc[-1])
            changes = {}
            for period, hours in [('1小时', 1), ('4小时', 4), ('24小时', 24)]:
                try:
                    past_oi = float(df[df['timestamp'] <=
                                     df['timestamp'].max() - pd.Timedelta(hours=hours)]
                                  ['sumOpenInterest'].iloc[-1])
                    change = current_oi - past_oi
                    change_percentage = (change / past_oi) * 100
                    changes[period] = {'change': change, 'change_percentage': change_percentage}
                except (IndexError, ValueError):
                    changes[period] = {'change': 0, 'change_percentage': 0}

            try:
                percentile = (df['sumOpenInterest'].rank(pct=True).iloc[-1]) * 100
            except Exception:
                percentile = 0

            analysis_results.append({
                'symbol': symbol,
                'current_oi': current_oi,
                'percentile': percentile,
                'changes': changes
            })

        df = pd.DataFrame(analysis_results)
        if 'changes' in df.columns:
            df['change_percentage'] = df['changes'].apply(
                lambda x: x.get('24小时', {}).get('change_percentage', 0)
            )
        else:
            df['change_percentage'] = 0
        return df

    def analyze_market_behavior(self, symbol: str, position_data: dict) -> str:
        """分析市场行为并生成AI报告"""
        price_data = {}
        for period, hours in [('1h', 1), ('4h', 4), ('1d', 24)]:
            df = get_binance_klines(symbol, period, limit=2)
            if not df.empty:
                price_change = ((float(df['close'].iloc[-1]) - float(df['open'].iloc[0])) 
                              / float(df['open'].iloc[0])) * 100
                price_data[f'{hours}h_change'] = price_change

        daily_klines = get_binance_klines(symbol, '1d', limit=1)
        volatility = 0
        if not daily_klines.empty:
            high = float(daily_klines['high'].iloc[0])
            low = float(daily_klines['low'].iloc[0])
            volatility = ((high - low) / low) * 100

        prompt = f"""
        作为一位专业的期货交易分析师，请基于以下{symbol}的数据进行深度市场行为分析：

        当前市场状态：
        - 当前持仓量：{position_data['current_oi']:,.0f}
        - 持仓分位数：{position_data['percentile']:.2f}%（高位>80%，低位<20%）
        - 24小时波动率：{volatility:.2f}%

        价格与持仓量变化对比：
        1小时周期：
        - 价格变化：{price_data.get('1h_change', 0):.2f}%
        - 持仓变化：{position_data['changes']['1小时']['change_percentage']:.2f}%
        4小时周期：
        - 价格变化：{price_data.get('4h_change', 0):.2f}%
        - 持仓变化：{position_data['changes']['4小时']['change_percentage']:.2f}%
        24小时周期：
        - 价格变化：{price_data.get('24h_change', 0):.2f}%
        - 持仓变化：{position_data['changes']['24小时']['change_percentage']:.2f}%

        请提供以下分析（使用markdown格式）：
        ## 市场行为分析
        [基于价格与持仓量的变化关系分析]
        ## 持仓水平研判
        [分析当前持仓量水平]
        ## 多空博弈分析
        - 主导方向：
        - 主要特征：
        - 市场情绪：
        ## 交易建议
        - 操作思路：
        - 关注重点：
        - 风险提示：
        """
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"生成AI分析报告失败: {e}")
            return "AI分析生成失败，请稍后重试"

    def generate_position_report(self, df: pd.DataFrame) -> str:
        """生成持仓分析AI报告"""
        df['change_percentage'] = df.apply(lambda x: x['changes']['24小时']['change_percentage'], axis=1)
        top_increase = df.nlargest(10, 'change_percentage')
        top_decrease = df.nsmallest(10, 'change_percentage')

        prompt = f"""作为一位专业的期货交易分析师，请基于以下持仓数据变化提供简洁的市场分析报告：
        持仓增加最显著的前10个交易对：
        {top_increase[['symbol', 'change_percentage']].to_string()}
        持仓减少最显著的前10个交易对：
        {top_decrease[['symbol', 'change_percentage']].to_string()}
        请提供以下分析（使用markdown格式）：
        ## 市场情绪分析
        [分析整体市场情绪]
        ## 主要变动解读
        - 大额持仓变动分析
        - 潜在市场方向
        ## 交易建议
        - 重点关注品种
        - 风险提示
        """
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"生成持仓分析报告失败: {e}")
            return "AI分析生成失败，请稍后重试"

class BinanceFundingRateTracker:
    """币安资金费率跟踪器"""
    def __init__(self, data_file="funding_rates_stats.json"):
        self.data_file = data_file
        self.previous_rates = {}
        self.current_rates = {}

        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    if 'previous_rates' in data:
                        self.previous_rates = data['previous_rates']
            except Exception as e:
                logger.error(f"加载历史费率数据失败: {e}")

    def get_usdt_perpetual_symbols(self) -> List[str]:
        """获取所有USDT结尾的永续合约交易对"""
        try:
            response = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
            data = response.json()
            usdt_symbols = [symbol_info['symbol'] for symbol_info in data['symbols']
                            if symbol_info['symbol'].endswith('USDT') and 
                               symbol_info['status'] == 'TRADING' and 
                               symbol_info['contractType'] == 'PERPETUAL']
            return usdt_symbols
        except Exception as e:
            logger.error(f"获取永续合约交易对失败: {e}")
            return []

    def get_funding_rates(self) -> Dict[str, float]:
        """获取所有USDT交易对的资金费率"""
        try:
            response = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex")
            data = response.json()
            funding_rates = {item['symbol']: float(item['lastFundingRate'])
                            for item in data if item['symbol'].endswith('USDT')}
            return funding_rates
        except Exception as e:
            logger.error(f"获取资金费率失败: {e}")
            return {}

    def get_top_n(self, rates: Dict[str, float], n: int, reverse: bool = True) -> List[Tuple[str, float]]:
        """获取费率最高/最低的n个交易对"""
        sorted_rates = sorted(rates.items(), key=lambda x: x[1], reverse=reverse)
        return sorted_rates[:n]

    def get_biggest_changes(self, current: Dict[str, float], previous: Dict[str, float], n: int,
                            increasing: bool = True) -> List[Tuple[str, float]]:
        """获取费率变化最大的n个交易对"""
        changes = {}
        for symbol, rate in current.items():
            if symbol in previous:
                change = rate - previous[symbol]
                if (increasing and change > 0) or (not increasing and change < 0):
                    changes[symbol] = change
        sorted_changes = sorted(changes.items(), key=lambda x: x[1], reverse=increasing)
        return sorted_changes[:n]

    def run_task(self):
        """执行资金费率统计任务"""
        logger.info(f"运行费率统计任务于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.current_rates = self.get_funding_rates()
        if not self.current_rates:
            logger.error("无法获取资金费率，跳过本次运行")
            return

        highest_rates = self.get_top_n(self.current_rates, 5, reverse=True)
        lowest_rates = self.get_top_n(self.current_rates, 5, reverse=False)
        increasing_rates = decreasing_rates = []

        if self.previous_rates:
            increasing_rates = self.get_biggest_changes(self.current_rates, self.previous_rates, 5, increasing=True)
            decreasing_rates = self.get_biggest_changes(self.current_rates, self.previous_rates, 5, increasing=False)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        stats = {
            "timestamp": timestamp,
            "highest_rates": [{"symbol": s, "rate": r} for s, r in highest_rates],
            "lowest_rates": [{"symbol": s, "rate": r} for s, r in lowest_rates],
            "biggest_increases": [{"symbol": s, "change": c} for s, c in increasing_rates],
            "biggest_decreases": [{"symbol": s, "change": c} for s, c in decreasing_rates],
            "previous_rates": self.current_rates
        }

        try:
            with open(self.data_file, 'w') as f:
                json.dump(stats, f, indent=4)
            logger.info(f"费率数据已保存至 {self.data_file}")
        except Exception as e:
            logger.error(f"保存费率数据失败: {e}")

        self.previous_rates = self.current_rates.copy()
def multi_timeframe_analysis(symbol: str) -> dict:
    """多周期分析功能"""
    results = {}
    trends = {}
    for tf, info in TIMEFRAMES.items():
        df = get_binance_klines(symbol, info['interval'], limit=100)
        if df.empty:
            logger.warning(f"时间周期 {tf} 数据获取失败")
            continue

        df['rsi'] = calculate_rsi(df['close'])
        support, resistance = calculate_support_resistance(df)
        sma20 = df['close'].rolling(window=20).mean()
        sma50 = df['close'].rolling(window=50).mean()
        current_price = float(df['close'].iloc[-1])

        if current_price > sma20.iloc[-1] > sma50.iloc[-1]:
            trend = "上升"
        elif current_price < sma20.iloc[-1] < sma50.iloc[-1]:
            trend = "下降"
        else:
            trend = "震荡"

        volume_sma = df['volume'].rolling(window=20).mean()
        volume_trend = "放量" if df['volume'].iloc[-1] > volume_sma.iloc[-1] else "缩量"

        trends[tf] = {
            "trend": trend,
            "rsi": df['rsi'].iloc[-1],
            "support": support,
            "resistance": resistance,
            "volume_trend": volume_trend
        }

    if not trends:
        return {"error": "无法获取任何时间周期的数据"}

    df_current = get_binance_klines(symbol, '1m', limit=1)
    if df_current.empty or 'close' not in df_current.columns:
        logger.error(f"获取当前价格失败: 数据为空或缺少'close'列")
        return {"error": "无法获取当前价格数据"}
    current_price = float(df_current['close'].iloc[0])

    short_term_trend = trends.get('5m', {}).get('trend', '') + "/" + trends.get('15m', {}).get('trend', '')
    medium_term_trend = trends.get('1h', {}).get('trend', '') + "/" + trends.get('4h', {}).get('trend', '')
    rsi_values = [data['rsi'] for data in trends.values()]
    avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else 0

    risk = {"level": "中等", "factors": []}
    if avg_rsi > 70:
        risk["factors"].append("RSI超买")
    elif avg_rsi < 30:
        risk["factors"].append("RSI超卖")

    for tf, data in trends.items():
        if abs(current_price - data['resistance']) / current_price < 0.02:
            risk["factors"].append(f"{TIMEFRAMES[tf]['name']}接近阻力位")
        if abs(current_price - data['support']) / current_price < 0.02:
            risk["factors"].append(f"{TIMEFRAMES[tf]['name']}接近支撑位")

    risk["level"] = "高" if len(risk["factors"]) >= 3 else "低" if len(risk["factors"]) <= 1 else "中等"

    prompt = f"""作为专业的加密货币分析师，请基于以下数据提供详细的市场分析报告：
    技术指标数据：
    - 当前价格：{current_price}
    - 短期趋势：{short_term_trend}
    - 中期趋势：{medium_term_trend}
    - RSI指标：{avg_rsi:.2f}
    - 支撑位：{trends['1h']['support']}
    - 阻力位：{trends['1h']['resistance']}
    - 成交量趋势：{trends['1h']['volume_trend']}
    风险评估：
    - 风险等级：{risk['level']}
    - 风险因素：{', '.join(risk['factors']) if risk['factors'] else '无重大风险'}
    请提供以下分析（使用markdown格式）：
    ## 市场综述
    [基于多周期分析框架的整体判断]
    ## 技术面分析
    - 趋势状态：
    - 支撑阻力分析：
    - 动量指标解读：
    - 成交量分析：
    ## 操作建议
    - 短期策略：
    - 中期布局：
    - 风险提示：
    请确保分析专业、客观，并注意风险提示。
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        ai_analysis = response.choices[0].message.content
    except Exception as e:
        logger.error(f"生成多周期分析报告失败: {e}")
        ai_analysis = "AI分析生成失败，请稍后重试"

    return {
        "current_price": current_price,
        "trends": trends,
        "risk": risk,
        "ai_analysis": ai_analysis
    }

class FundFlowAnalyzer:
    """资金流向分析器"""
    def __init__(self):
        self.spot_base_url = "https://api.binance.com/api/v3"
        self.futures_base_url = "https://fapi.binance.com/fapi/v1"
        self.stablecoins = {'USDC', 'TUSD', 'BUSD', 'DAI', 'USDP', 'EUR', 'GYEN'}

    def get_all_usdt_symbols(self, is_futures=False):
        """获取所有以USDT结尾的交易对，剔除稳定币对"""
        base_url = self.futures_base_url if is_futures else self.spot_base_url
        endpoint = "/exchangeInfo"
        response = requests.get(f"{base_url}{endpoint}")
        data = response.json()
        symbols = []
        if is_futures:
            for item in data['symbols']:
                symbol = item['symbol']
                base_asset = item['baseAsset']
                if (item['status'] == 'TRADING' and
                    item['quoteAsset'] == 'USDT' and
                    base_asset not in self.stablecoins):
                    symbols.append(symbol)
        else:
            for item in data['symbols']:
                symbol = item['symbol']
                base_asset = item['baseAsset']
                if (item['status'] == 'TRADING' and
                    item['quoteAsset'] == 'USDT' and
                    base_asset not in self.stablecoins):
                    symbols.append(symbol)
        return symbols

    def format_number(self, value):
        """将数值格式化为K/M表示，保留两位小数"""
        if abs(value) >= 1000000:
            return f"{value / 1000000:.2f}M"
        elif abs(value) >= 1000:
            return f"{value / 1000:.2f}K"
        else:
            return f"{value:.2f}"

    def get_klines_parallel(self, symbols, is_futures=False, max_workers=20, include_latest=False):
        """使用线程池并行获取多个交易对的K线数据（基于最近完成的 4H K线，可选最新价格）"""
        results = []
        failed_symbols = []
        def fetch_kline(symbol):
            try:
                base_url = self.futures_base_url if is_futures else self.spot_base_url
                endpoint = "/klines"
                now = datetime.utcnow()
                # 计算最近完成的 4H 周期
                hours_since_day_start = now.hour + now.minute / 60 + now.second / 3600
                last_4h_offset = int(hours_since_day_start // 4) * 4
                last_4h_start = now.replace(minute=0, second=0, microsecond=0, hour=0) + timedelta(hours=last_4h_offset)
                if last_4h_start > now:
                    last_4h_start -= timedelta(hours=4)
                end_time = int(last_4h_start.timestamp() * 1000)
                start_time = int((last_4h_start - timedelta(hours=4)).timestamp() * 1000)
                params = {
                    'symbol': symbol,
                    'interval': '4h',
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 2
                }
                response = requests.get(f"{base_url}{endpoint}", params=params)
                response.raise_for_status()
                data = response.json()
                if not data or len(data) < 2:
                    logger.error(f"数据不足 {symbol}: 返回 {len(data)} 根K线")
                    return None
                k = data[-1]
                result = {
                    'symbol': symbol,
                    'open_time': datetime.fromtimestamp(k[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'close_time': datetime.fromtimestamp(k[6] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'quote_volume': float(k[7]),
                    'trades': int(k[8]),
                    'taker_buy_base_volume': float(k[9]),
                    'taker_buy_quote_volume': float(k[10]),
                    'net_inflow': 2 * float(k[10]) - float(k[7])
                }
                # 获取最新价格（可选）
                if include_latest:
                    latest_params = {
                        'symbol': symbol,
                        'interval': '1m',
                        'limit': 1
                    }
                    latest_response = requests.get(f"{base_url}{endpoint}", params=latest_params)
                    latest_response.raise_for_status()
                    latest_data = latest_response.json()
                    if latest_data and len(latest_data) > 0:
                        result['latest_price'] = float(latest_data[0][4])  # 最新收盘价
                    else:
                        result['latest_price'] = result['close']  # 回退到 4H 收盘价
                return result
            except Exception as e:
                logger.error(f"获取 {symbol} 数据出错: {e}")
                return None
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_kline, symbol): symbol for symbol in symbols}
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed_symbols.append(symbol)
        if failed_symbols:
            logger.warning(f"以下交易对数据获取失败: {failed_symbols}")
        if not results:
            logger.error("所有交易对数据获取失败")
            return []
        return results

    def send_to_deepseek(self, data):
        """发送数据到 DeepSeek API 生成分析报告"""
        prompt = (
            "基于Binance现货和期货市场USDT交易对前4H已完成收盘数据的资金流入流出情况，生成专业资金流向分析报告。数据如下：\n" +
            json.dumps(data, indent=2, ensure_ascii=False) +
            "\n\n"
            "### 分析要求\n"
            "1. **主力资金行为解读**：\n"
            "   - 识别现货和期货市场中相同交易对的资金流向特征（净流入/流出）。\n"
            "   - 分析潜在主力行为（如吸筹、拉抬、对冲、对倒），结合量价关系（高成交量低波动等）。\n"
            "   - 评估订单簿压力（volume imbalance）和资金流向与价格的相关性。\n"
            "2. **价格阶段判断**：\n"
            "   - 根据资金流向推断市场阶段（整理/上涨/下跌），量化置信度（如相关性或趋势强度）。\n"
            "   - 对比现货和期货市场的趋势一致性及主导关系。\n"
            "3. **短期趋势预判（4-8小时）**：\n"
            "   - 预测主要交易对的短期方向（看涨/看跌/震荡），标示关键支撑/阻力位。\n"
            "4. **交易策略建议**：\n"
            "   - 针对主要交易对（如BTCUSDT、ETHUSDT）提出具体操作策略（方向、入场点、止损、目标）。\n"
            "   - 提供风险收益比和对冲建议。\n"
            "\n"
            "### 输出规范\n"
            "- 使用Markdown格式，结构为：[主力资金行为解读]→[价格阶段判断]→[短期趋势预判]→[交易策略建议]。\n"
            "- 语言简洁专业，突出可操作信号，避免冗余描述。\n"
            "- 数据呈现：\n"
            "  - 关键指标（如volume imbalance、correlation）用**粗体**标注。\n"
            "  - 表格对比现货/期货资金流向及策略要点。\n"
            "  - 趋势置信度以百分比或p值表述。\n"
            "- 以中文回复，确保逻辑清晰，适用于交易决策。\n"
        )
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek API错误: {e}")
            return "无法获取DeepSeek分析结果"

    def analyze_fund_flow(self):
        """分析资金流向"""
        spot_symbols = self.get_all_usdt_symbols(is_futures=False)
        futures_symbols = self.get_all_usdt_symbols(is_futures=True)
        spot_data = self.get_klines_parallel(spot_symbols, is_futures=False, max_workers=20, include_latest=True)
        futures_data = self.get_klines_parallel(futures_symbols, is_futures=True, max_workers=20, include_latest=True)
        spot_df = pd.DataFrame(spot_data)
        futures_df = pd.DataFrame(futures_data)
        if spot_df.empty or 'net_inflow' not in spot_df.columns:
            logger.error("现货数据为空或缺少 'net_inflow' 列")
            spot_df = pd.DataFrame(columns=['symbol', 'net_inflow', 'quote_volume', 'latest_price'])
        if futures_df.empty or 'net_inflow' not in futures_df.columns:
            logger.error("期货数据为空或缺少 'net_inflow' 列")
            futures_df = pd.DataFrame(columns=['symbol', 'net_inflow', 'quote_volume', 'latest_price'])
        spot_inflow_top20 = spot_df.sort_values(by='net_inflow', ascending=False).head(20)
        futures_inflow_top20 = futures_df.sort_values(by='net_inflow', ascending=False).head(20)
        spot_outflow_top20 = spot_df.sort_values(by='net_inflow', ascending=True).head(20)
        futures_outflow_top20 = futures_df.sort_values(by='net_inflow', ascending=True).head(20)
        deepseek_data = {
            "spot_inflow_top20": spot_inflow_top20[['symbol', 'net_inflow', 'quote_volume', 'latest_price']].to_dict('records'),
            "futures_inflow_top20": futures_inflow_top20[['symbol', 'net_inflow', 'quote_volume', 'latest_price']].to_dict('records'),
            "spot_outflow_top20": spot_outflow_top20[['symbol', 'net_inflow', 'quote_volume', 'latest_price']].to_dict('records'),
            "futures_outflow_top20": futures_outflow_top20[['symbol', 'net_inflow', 'quote_volume', 'latest_price']].to_dict('records'),
            "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        analysis = self.send_to_deepseek(deepseek_data)
        return {
            "spot_inflow_top20": spot_inflow_top20,
            "futures_inflow_top20": futures_inflow_top20,
            "spot_outflow_top20": spot_outflow_top20,
            "futures_outflow_top20": futures_outflow_top20,
            "analysis": analysis
        }

# 费率监控相关函数
def update_data(symbol):
    """更新费率监控数据"""
    now = datetime.now(timezone.utc)
    spot_price = get_spot_price(symbol)
    futures_price = get_futures_price(symbol)
    funding_rate = get_funding_rate(symbol)
    open_interest = get_open_interest(symbol)

    if spot_price is not None and futures_price is not None:
        premium = (futures_price - spot_price) / spot_price * 100
        st.session_state.timestamps.append(now)
        st.session_state.spot_prices.append(spot_price)
        st.session_state.futures_prices.append(futures_price)
        st.session_state.premiums.append(premium)

        if funding_rate is not None:
            st.session_state.funding_rates.append(funding_rate * 100)
            st.session_state.last_funding_rate = funding_rate
        elif st.session_state.funding_rates:
            st.session_state.funding_rates.append(st.session_state.funding_rates[-1])
        else:
            st.session_state.funding_rates.append(0)

        if open_interest is not None:
            st.session_state.open_interest.append(open_interest)
        elif st.session_state.open_interest:
            st.session_state.open_interest.append(st.session_state.open_interest[-1])
        else:
            st.session_state.open_interest.append(0)

        if len(st.session_state.timestamps) > 1:
            cutoff_time = now - timedelta(hours=HOURS_TO_DISPLAY)
            if st.session_state.timestamps[0] < cutoff_time:
                valid_indices = [i for i, ts in enumerate(st.session_state.timestamps) if ts >= cutoff_time]
                if valid_indices:
                    start_idx = valid_indices[0]
                    st.session_state.timestamps = st.session_state.timestamps[start_idx:]
                    st.session_state.spot_prices = st.session_state.spot_prices[start_idx:]
                    st.session_state.futures_prices = st.session_state.futures_prices[start_idx:]
                    st.session_state.premiums = st.session_state.premiums[start_idx:]
                    st.session_state.funding_rates = st.session_state.funding_rates[start_idx:]
                    st.session_state.open_interest = st.session_state.open_interest[start_idx:]

        return spot_price, futures_price, premium, funding_rate, open_interest
    return None, None, None, funding_rate, open_interest

def create_premium_chart():
    """创建溢价率图表（中国时区 CST）"""
    if not st.session_state.timestamps:
        return None
    # 将 UTC 时间转换为 CST (UTC+8)
    cst_timestamps = [ts + timedelta(hours=8) for ts in st.session_state.timestamps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cst_timestamps, y=st.session_state.premiums, mode='lines',
                             name='期现溢价率 (%)', line=dict(color='green')))
    fig.update_layout(
        height=300,
        title_text=f"{st.session_state.symbol} 期现溢价率 (%) (CST)",
        margin=dict(l=40, r=40, t=50, b=30),
        xaxis_title="时间 (CST)",
        yaxis_title="期现溢价率 (%)",
        xaxis=dict(range=[datetime.now(timezone.utc) - timedelta(hours=HOURS_TO_DISPLAY - 8), 
                         datetime.now(timezone.utc) + timedelta(hours=8)])
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    return fig

def create_funding_rate_chart():
    """创建资金费率图表（中国时区 CST）"""
    if not st.session_state.timestamps:
        return None
    # 将 UTC 时间转换为 CST (UTC+8)
    cst_timestamps = [ts + timedelta(hours=8) for ts in st.session_state.timestamps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cst_timestamps, y=st.session_state.funding_rates, mode='lines',
                             name='资金费率 (%)', line=dict(color='red')))
    fig.update_layout(
        height=300,
        title_text=f"{st.session_state.symbol} 资金费率 (%) (CST)",
        margin=dict(l=40, r=40, t=50, b=30),
        xaxis_title="时间 (CST)",
        yaxis_title="资金费率 (%)",
        xaxis=dict(range=[datetime.now(timezone.utc) - timedelta(hours=HOURS_TO_DISPLAY - 8), 
                         datetime.now(timezone.utc) + timedelta(hours=8)])
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    return fig

def create_open_interest_chart():
    """创建持仓量图表（中国时区 CST）"""
    if not st.session_state.timestamps:
        return None
    # 将 UTC 时间转换为 CST (UTC+8)
    cst_timestamps = [ts + timedelta(hours=8) for ts in st.session_state.timestamps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cst_timestamps, y=st.session_state.open_interest, mode='lines',
                             name='持仓量', line=dict(color='blue')))
    fig.update_layout(
        height=300,
        title_text=f"{st.session_state.symbol} 持仓量 (CST)",
        margin=dict(l=40, r=40, t=50, b=30),
        xaxis_title="时间 (CST)",
        yaxis_title="持仓量",
        xaxis=dict(range=[datetime.now(timezone.utc) - timedelta(hours=HOURS_TO_DISPLAY - 8), 
                         datetime.now(timezone.utc) + timedelta(hours=8)])
    )
    return fig
def load_stats_data():
    """读取资金费率统计数据"""
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                data = json.load(f)
                st.session_state.stats_data = data
                st.session_state.last_stats_update = datetime.now()
                return data
        return None
    except Exception as e:
        logger.error(f"读取统计数据出错: {e}")
        return None

def load_historical_data(symbol):
    """加载费率监控历史数据"""
    if not st.session_state.historical_data_loaded:
        with st.spinner("正在加载历史数据..."):
            timestamps, spot_prices, futures_prices, premiums = get_historical_klines(symbol, "1m", MAX_DATA_POINTS)
            funding_timestamps, funding_rates = get_historical_funding_rates(symbol)
            oi_timestamps, open_interests = get_historical_open_interest(symbol)

            if timestamps:
                st.session_state.timestamps = timestamps
                st.session_state.spot_prices = spot_prices
                st.session_state.futures_prices = futures_prices
                st.session_state.premiums = premiums

                if funding_rates:
                    mapped_funding_rates = []
                    for ts in timestamps:
                        closest_idx = min(range(len(funding_timestamps)), 
                                        key=lambda i: abs((ts - funding_timestamps[i]).total_seconds()))
                        mapped_funding_rates.append(funding_rates[closest_idx] if closest_idx < len(funding_rates) else 0)
                    st.session_state.funding_rates = mapped_funding_rates
                else:
                    st.session_state.funding_rates = [0] * len(timestamps)

                if open_interests:
                    mapped_open_interests = []
                    for ts in timestamps:
                        closest_idx = min(range(len(oi_timestamps)), 
                                        key=lambda i: abs((ts - oi_timestamps[i]).total_seconds()))
                        mapped_open_interests.append(open_interests[closest_idx] if closest_idx < len(open_interests) else 0)
                    st.session_state.open_interest = mapped_open_interests
                else:
                    st.session_state.open_interest = [0] * len(timestamps)

                funding_rate = get_funding_rate(symbol)
                open_interest = get_open_interest(symbol)
                if funding_rate is not None:
                    st.session_state.last_funding_rate = funding_rate
                    if st.session_state.funding_rates:
                        st.session_state.funding_rates[-1] = funding_rate * 100
                if open_interest is not None and st.session_state.open_interest:
                    st.session_state.open_interest[-1] = open_interest

                st.session_state.historical_data_loaded = True
                return True
            return False
    return True

def display_stats_data():
    """显示资金费率统计数据"""
    if (st.session_state.last_stats_update is None or
            (datetime.now() - st.session_state.last_stats_update).total_seconds() > 60):
        load_stats_data()

    container = st.container()
    with container:
        if st.session_state.stats_data:
            data = st.session_state.stats_data
            timestamp = data.get("timestamp", "未知")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.subheader("费率最高的交易对")
                if "highest_rates" in data and data["highest_rates"]:
                    df_highest = pd.DataFrame([{"交易对": item.get("symbol", ""), "费率": f"{item.get('rate', 0) * 100:.2f}%"}
                                              for item in data["highest_rates"]])
                    st.dataframe(df_highest, hide_index=True)
                else:
                    st.write("暂无数据")

            with col2:
                st.subheader("费率最低的交易对")
                if "lowest_rates" in data and data["lowest_rates"]:
                    df_lowest = pd.DataFrame([{"交易对": item.get("symbol", ""), "费率": f"{item.get('rate', 0) * 100:.2f}%"}
                                             for item in data["lowest_rates"]])
                    st.dataframe(df_lowest, hide_index=True)
                else:
                    st.write("暂无数据")

            with col3:
                st.subheader("费率上升最快")
                if "biggest_increases" in data and data["biggest_increases"]:
                    df_increases = pd.DataFrame([{"交易对": item.get("symbol", ""), "变化": f"{item.get('change', 0) * 100:.4f}%"}
                                                for item in data["biggest_increases"]])
                    st.dataframe(df_increases, hide_index=True)
                else:
                    st.write("暂无数据")

            with col4:
                st.subheader("费率下降最快")
                if "biggest_decreases" in data and data["biggest_decreases"]:
                    df_decreases = pd.DataFrame([{"交易对": item.get("symbol", ""), "变化": f"{item.get('change', 0) * 100:.4f}%"}
                                                for item in data["biggest_decreases"]])
                    st.dataframe(df_decreases, hide_index=True)
                else:
                    st.write("暂无数据")

            st.caption(f"更新时间: {timestamp}")
        else:
            st.error("未能加载数据，请检查API连接")
    return container

# 初始化会话状态（在全局作用域中定义，避免重复初始化）
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = "市场行为分析"
if 'symbol' not in st.session_state:
    st.session_state.symbol = "BTCUSDT"
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []
    st.session_state.spot_prices = []
    st.session_state.futures_prices = []
    st.session_state.premiums = []
    st.session_state.funding_rates = []
    st.session_state.open_interest = []
    st.session_state.last_funding_rate = None
    st.session_state.running = False
    st.session_state.charts = [None, None, None]
    st.session_state.historical_data_loaded = False
    st.session_state.stats_data = None
    st.session_state.last_stats_update = None
def main():
    st.title("分析系统")
    st.sidebar.header("功能导航")

    # 初始化分析选择
    if st.sidebar.button("市场行为分析",
                         type="primary" if st.session_state.current_analysis == "市场行为分析" else "secondary",
                         use_container_width=True):
        st.session_state.current_analysis = "市场行为分析"
    if st.sidebar.button("持仓分析",
                         type="primary" if st.session_state.current_analysis == "持仓分析" else "secondary",
                         use_container_width=True):
        st.session_state.current_analysis = "持仓分析"
    if st.sidebar.button("多周期分析",
                         type="primary" if st.session_state.current_analysis == "多周期分析" else "secondary",
                         use_container_width=True):
        st.session_state.current_analysis = "多周期分析"
    if st.sidebar.button("资金流向分析",
                         type="primary" if st.session_state.current_analysis == "资金流向分析" else "secondary",
                         use_container_width=True):
        st.session_state.current_analysis = "资金流向分析"
    if st.sidebar.button("费率监控",
                         type="primary" if st.session_state.current_analysis == "费率监控" else "secondary",
                         use_container_width=True):
        st.session_state.current_analysis = "费率监控"

    # 市场行为分析
    if st.session_state.current_analysis == "市场行为分析":
        st.header("市场行为分析")
        analyzer = BinanceFuturesAnalyzer()
        symbols = analyzer.get_usdt_symbols()
        selected_symbol = st.selectbox("选择交易对", symbols)
        if st.button("开始分析"):
            with st.spinner("正在分析市场行为..."):
                position_data = analyzer.get_open_interest(selected_symbol)
                if position_data and position_data['data']:
                    df = pd.DataFrame(position_data['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
                    current_oi = float(df['sumOpenInterest'].iloc[-1])
                    percentile = (df['sumOpenInterest'].rank(pct=True).iloc[-1]) * 100
                    changes = {}
                    for period, hours in [('1小时', 1), ('4小时', 4), ('24小时', 24)]:
                        past_oi = float(df[df['timestamp'] <=
                                         df['timestamp'].max() - pd.Timedelta(hours=hours)]
                                      ['sumOpenInterest'].iloc[-1])
                        change = current_oi - past_oi
                        change_percentage = (change / past_oi) * 100
                        changes[period] = {'change': change, 'change_percentage': change_percentage}
                    position_info = {'current_oi': current_oi, 'percentile': percentile, 'changes': changes}
                    analysis_report = analyzer.analyze_market_behavior(selected_symbol, position_info)
                    st.markdown(analysis_report)

    # 持仓分析
    elif st.session_state.current_analysis == "持仓分析":
        st.header("持仓分析")
        analyzer = BinanceFuturesAnalyzer()
        if st.button("分析所有交易对持仓"):
            with st.spinner("正在分析持仓数据..."):
                df = analyzer.analyze_positions()
                st.subheader("持仓变化Top10")
                increase_top10 = df.nlargest(10, 'change_percentage')[['symbol', 'change_percentage', 'current_oi']]
                decrease_top10 = df.nsmallest(10, 'change_percentage')[['symbol', 'change_percentage', 'current_oi']]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**增持Top10**")
                    increase_display = increase_top10.copy()
                    increase_display.columns = ['交易对', '变化率(%)', '当前持仓']
                    increase_display['变化率(%)'] = increase_display['变化率(%)'].round(2)
                    increase_display['当前持仓'] = increase_display['当前持仓'].round(2)
                    st.table(increase_display)
                with col2:
                    st.markdown("**减持Top10**")
                    decrease_display = decrease_top10.copy()
                    decrease_display.columns = ['交易对', '变化率(%)', '当前持仓']
                    decrease_display['变化率(%)'] = decrease_display['变化率(%)'].round(2)
                    decrease_display['当前持仓'] = decrease_display['当前持仓'].round(2)
                    st.table(decrease_display)
                st.subheader("总体持仓分析")
                report = analyzer.generate_position_report(df)
                st.markdown(report)

    # 多周期分析
    elif st.session_state.current_analysis == "多周期分析":
        st.header("多周期分析")
        analyzer = BinanceFuturesAnalyzer()
        symbols = analyzer.get_usdt_symbols()
        selected_symbol = st.selectbox("选择交易对", symbols)
        if st.button("开始多周期分析"):
            with st.spinner("正在进行多周期分析..."):
                analysis_result = multi_timeframe_analysis(selected_symbol)
                if "error" in analysis_result:
                    st.error(analysis_result["error"])
                else:
                    st.markdown(analysis_result["ai_analysis"])

    # 资金流向分析
    elif st.session_state.current_analysis == "资金流向分析":
        st.header("资金流向分析")
        fund_flow_analyzer = FundFlowAnalyzer()
        if st.button("开始资金流向分析"):
            with st.spinner("正在分析资金流向..."):
                analysis_result = fund_flow_analyzer.analyze_fund_flow()
                st.subheader("现货交易对净流入TOP20")
                st.table(analysis_result["spot_inflow_top20"][['symbol', 'net_inflow', 'quote_volume']])
                st.subheader("期货交易对净流入TOP20")
                st.table(analysis_result["futures_inflow_top20"][['symbol', 'net_inflow', 'quote_volume']])
                st.subheader("现货交易对净流出TOP20")
                st.table(analysis_result["spot_outflow_top20"][['symbol', 'net_inflow', 'quote_volume']])
                st.subheader("期货交易对净流出TOP20")
                st.table(analysis_result["futures_outflow_top20"][['symbol', 'net_inflow', 'quote_volume']])
                st.subheader("DeepSeek分析结果")
                st.markdown(analysis_result["analysis"])

    # 费率监控
    elif st.session_state.current_analysis == "费率监控":
        st.header("加密货币期现溢价监控")
        
        # 侧边栏控件
        with st.sidebar:
            st.title("监控设置")
            new_symbol = st.text_input("输入交易对", value=st.session_state.symbol, placeholder="例如: BTCUSDT, ETHUSDT")
            if new_symbol != st.session_state.symbol:
                st.session_state.symbol = new_symbol
                st.session_state.timestamps = []
                st.session_state.spot_prices = []
                st.session_state.futures_prices = []
                st.session_state.premiums = []
                st.session_state.funding_rates = []
                st.session_state.open_interest = []
                st.session_state.last_funding_rate = None
                st.session_state.historical_data_loaded = False
                st.session_state.charts = [None, None, None]
                if st.session_state.running:
                    st.session_state.running = False
                st.experimental_rerun()

            col1, col2 = st.columns(2)
            with col1:
                start_stop = st.button('开始监控' if not st.session_state.running else '停止监控', use_container_width=True)
                if start_stop:
                    st.session_state.running = not st.session_state.running
                    if st.session_state.running:
                        success = load_historical_data(st.session_state.symbol)
                        if not success:
                            st.error("无法加载历史数据，请检查交易对是否正确")
                            st.session_state.running = False
                        st.experimental_rerun()
            with col2:
                if st.button('清除数据', use_container_width=True):
                    st.session_state.timestamps = []
                    st.session_state.spot_prices = []
                    st.session_state.futures_prices = []
                    st.session_state.premiums = []
                    st.session_state.funding_rates = []
                    st.session_state.open_interest = []
                    st.session_state.last_funding_rate = None
                    st.session_state.historical_data_loaded = False
                    st.session_state.charts = [None, None, None]
                    st.experimental_rerun()

        # 显示统计数据
        stats_placeholder = st.empty()
        with stats_placeholder:
            display_stats_data()

        # 显示最新数据
        metrics_placeholder = st.empty()

        # 图表布局
        chart_col1, chart_col2, chart_col3 = st.columns(3)

        if st.session_state.running:
            progress_placeholder = st.empty()
            if not st.session_state.historical_data_loaded:
                success = load_historical_data(st.session_state.symbol)
                if not success:
                    st.error("无法加载历史数据，请检查交易对是否正确")
                    st.session_state.running = False
                    st.experimental_rerun()

            if st.session_state.charts[0] is None:
                with chart_col1:
                    st.session_state.charts[0] = st.empty()
            if st.session_state.charts[1] is None:
                with chart_col2:
                    st.session_state.charts[1] = st.empty()
            if st.session_state.charts[2] is None:
                with chart_col3:
                    st.session_state.charts[2] = st.empty()

            last_stats_refresh = time.time()
            while st.session_state.running:
                spot_price, futures_price, premium, funding_rate, open_interest = update_data(st.session_state.symbol)
                current_time = time.time()
                if current_time - last_stats_refresh > 60:
                    with stats_placeholder:
                        display_stats_data()
                    last_stats_refresh = current_time

                if spot_price is not None and futures_price is not None:
                    # 使用 CST 时间显示当前数据
                    china_time = datetime.now(timezone.utc) + timedelta(hours=8)
                    current_time_cst = china_time.strftime("%Y-%m-%d %H:%M:%S CST")
                    metrics_placeholder.markdown(f"""
                    ### 当前数据 - {st.session_state.symbol} ({current_time_cst})
                    | 现货价格 | 期货价格 | 期现溢价 | 资金费率 | 持仓量 |
                    | --- | --- | --- | --- | --- |
                    | {spot_price:.6f} | {futures_price:.6f} | {premium:.4f}% | {funding_rate * 100:.6f}% | {open_interest:.2f} |
                    """)

                premium_fig = create_premium_chart()
                funding_fig = create_funding_rate_chart()
                open_interest_fig = create_open_interest_chart()
                if premium_fig and st.session_state.charts[0]:
                    st.session_state.charts[0].plotly_chart(premium_fig, use_container_width=True)
                if funding_fig and st.session_state.charts[1]:
                    st.session_state.charts[1].plotly_chart(funding_fig, use_container_width=True)
                if open_interest_fig and st.session_state.charts[2]:
                    st.session_state.charts[2].plotly_chart(open_interest_fig, use_container_width=True)

                for i in range(UPDATE_INTERVAL, 0, -1):
                    progress_placeholder.progress(1 - i / UPDATE_INTERVAL, text=f"下次更新倒计时: {i}秒")
                    time.sleep(1)

if __name__ == "__main__":
    # 启动资金费率跟踪器（后台任务）
    tracker = BinanceFundingRateTracker()
    schedule.every(5).minutes.do(tracker.run_task)
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(10)
    threading.Thread(target=run_scheduler, daemon=True).start()
    
    # 运行主程序
    main()
