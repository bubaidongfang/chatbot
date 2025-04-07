# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import time
import requests
import json
import os
import io
import fcntl
import errno
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from decimal import Decimal, InvalidOperation
import threading
import schedule
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义中国时区（UTC+8）
CHINA_TZ = timezone(timedelta(hours=8))

# 获取当前文件的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 定义统计文件的绝对路径
STATS_FILE = os.path.join(CURRENT_DIR, "funding_rates_stats.json")


# 文件锁定类，用于安全的文件读写
class FileLock:
    def __init__(self, file_path):
        self.file_path = file_path
        self.lock_file = f"{file_path}.lock"
        self.lock_fd = None

    def __enter__(self):
        self.lock_fd = open(self.lock_file, 'w')
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError as e:
            if e.errno == errno.EAGAIN:
                logger.warning(f"文件 {self.file_path} 已被锁定，等待解锁...")
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX)  # 阻塞等待锁
            else:
                raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_fd:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            self.lock_fd.close()
            try:
                os.remove(self.lock_file)
            except:
                pass


# 通用函数：格式化价格
def format_price(price):
    if price is None or price == float('inf') or price == float('-inf'):
        return "N/A"
    try:
        price_decimal = Decimal(str(price))
        price_str = str(price_decimal).upper()
        if 'E' in price_str:
            exponent = abs(price_decimal.as_tuple().exponent)
            if exponent > 4:
                return f"{price_decimal:.0f}"
            elif exponent > 2:
                return f"{price_decimal:.2f}"
            return f"{price_decimal:.6f}"
        if abs(price_decimal) >= 10000:
            return f"{price_decimal:,.0f}"
        elif abs(price_decimal) >= 1000:
            return f"{price_decimal:,.0f}"
        elif abs(price_decimal) >= 100:
            return f"{price_decimal:.2f}"
        elif abs(price_decimal) >= 1:
            return f"{price_decimal:.3f}"
        elif abs(price_decimal) >= 0.1:
            return f"{price_decimal:.4f}"
        elif abs(price_decimal) >= 0.01:
            return f"{price_decimal:.5f}"
        elif abs(price_decimal) >= 0.001:
            return f"{price_decimal:.6f}"
        else:
            formatted = f"{price_decimal:.8f}"
            return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted
    except (InvalidOperation, ValueError, TypeError):
        return "N/A"


# 安全的JSON文件读写函数
def safe_read_json(file_path):
    """安全地读取JSON文件，处理各种可能的错误"""
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return None

    try:
        with FileLock(file_path):
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if not content:  # 文件为空
                    logger.warning(f"文件为空: {file_path}")
                    return None
                return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {e}, 文件: {file_path}")
        # 尝试修复文件
        try:
            with FileLock(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                # 如果文件损坏，创建一个新的空文件
                with open(file_path, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S'),
                        "highest_rates": [],
                        "lowest_rates": [],
                        "biggest_increases": [],
                        "biggest_decreases": [],
                        "previous_rates": {}
                    }, f, indent=4)
                logger.info(f"已创建新的统计文件: {file_path}")
        except Exception as e2:
            logger.error(f"修复文件失败: {e2}")
        return None
    except Exception as e:
        logger.error(f"读取文件错误: {e}, 文件: {file_path}")
        return None


def safe_write_json(file_path, data):
    """安全地写入JSON文件，确保数据完整性"""
    try:
        # 先写入临时文件
        temp_file = f"{file_path}.tmp"
        with FileLock(file_path):
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=4)
            # 成功写入后替换原文件
            os.replace(temp_file, file_path)
        logger.info(f"成功写入数据到: {file_path}")
        return True
    except Exception as e:
        logger.error(f"写入文件错误: {e}, 文件: {file_path}")
        if os.path.exists(f"{file_path}.tmp"):
            try:
                os.remove(f"{file_path}.tmp")
            except:
                pass
        return False


# 数据获取函数
def get_spot_price(symbol):
    try:
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return float(data["price"]) if "price" in data else None
    except Exception as e:
        logger.error(f"获取现货价格出错: {e}")
        return None


def get_futures_price(symbol):
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return float(data["price"]) if "price" in data else None
    except Exception as e:
        logger.error(f"获取期货价格出错: {e}")
        return None


def get_funding_rate(symbol):
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return float(data["lastFundingRate"]) if "lastFundingRate" in data else None
    except Exception as e:
        logger.error(f"获取资金费率出错: {e}")
        return None


def get_open_interest(symbol):
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return float(data["openInterest"]) if "openInterest" in data else None
    except Exception as e:
        logger.error(f"获取持仓量出错: {e}")
        return None


def get_historical_klines(symbol, interval, limit):
    try:
        end_time = int(datetime.now(CHINA_TZ).timestamp() * 1000)
        start_time = int((datetime.now(CHINA_TZ) - timedelta(hours=4)).timestamp() * 1000)
        spot_url = "https://api.binance.com/api/v3/klines"
        futures_url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "startTime": start_time, "endTime": end_time, "limit": limit}

        spot_response = requests.get(spot_url, params=params)
        futures_response = requests.get(futures_url, params=params)

        if spot_response.status_code != 200 or futures_response.status_code != 200:
            logger.warning(f"API响应错误: 现货={spot_response.status_code}, 期货={futures_response.status_code}")
            return [], [], [], []

        spot_data = spot_response.json()
        futures_data = futures_response.json()

        timestamps, spot_prices, futures_prices, premiums = [], [], [], []
        min_length = min(len(spot_data), len(futures_data))

        for i in range(min_length):
            timestamp = datetime.fromtimestamp(spot_data[i][0] / 1000, tz=CHINA_TZ)
            spot_close = float(spot_data[i][4])
            futures_close = float(futures_data[i][4])
            premium = (futures_close - spot_close) / spot_close * 100
            timestamps.append(timestamp)
            spot_prices.append(spot_close)
            futures_prices.append(futures_close)
            premiums.append(premium)

        return timestamps, spot_prices, futures_prices, premiums
    except Exception as e:
        logger.error(f"获取历史K线数据出错: {e}")
        return [], [], [], []


def get_historical_funding_rates(symbol, limit=240):
    try:
        end_time = int(datetime.now(CHINA_TZ).timestamp() * 1000)
        start_time = int((datetime.now(CHINA_TZ) - timedelta(hours=4)).timestamp() * 1000)
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {"symbol": symbol, "startTime": start_time, "endTime": end_time, "limit": limit}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            logger.warning(f"获取历史资金费率API响应错误: {response.status_code}")
            return [], []

        data = response.json()

        timestamps, funding_rates = [], []
        for item in data:
            timestamps.append(datetime.fromtimestamp(item["fundingTime"] / 1000, tz=CHINA_TZ))
            funding_rates.append(float(item["fundingRate"]) * 100)
        return timestamps, funding_rates
    except Exception as e:
        logger.error(f"获取历史资金费率出错: {e}")
        return [], []


def get_historical_open_interest(symbol, period="5m", limit=240):
    try:
        end_time = int(datetime.now(CHINA_TZ).timestamp() * 1000)
        start_time = int((datetime.now(CHINA_TZ) - timedelta(hours=4)).timestamp() * 1000)
        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {"symbol": symbol, "period": period, "startTime": start_time, "endTime": end_time, "limit": limit}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            logger.warning(f"获取历史持仓量API响应错误: {response.status_code}")
            return [], []

        data = response.json()

        timestamps, open_interests = [], []
        for item in data:
            timestamps.append(datetime.fromtimestamp(item["timestamp"] / 1000, tz=CHINA_TZ))
            open_interests.append(float(item["sumOpenInterest"]))
        return timestamps, open_interests
    except Exception as e:
        logger.error(f"获取历史持仓量出错: {e}")
        return [], []


def update_data(symbol, symbol_data):
    now = datetime.now(CHINA_TZ)
    spot_price = get_spot_price(symbol)
    futures_price = get_futures_price(symbol)
    funding_rate = get_funding_rate(symbol)
    open_interest = get_open_interest(symbol)

    if spot_price is not None and futures_price is not None:
        premium = (futures_price - spot_price) / spot_price * 100
        symbol_data["timestamps"].append(now)
        symbol_data["spot_prices"].append(spot_price)
        symbol_data["futures_prices"].append(futures_price)
        symbol_data["premiums"].append(premium)

        if funding_rate is not None:
            symbol_data["funding_rates"].append(funding_rate * 100)
            symbol_data["last_funding_rate"] = funding_rate
        elif symbol_data["funding_rates"]:
            symbol_data["funding_rates"].append(symbol_data["funding_rates"][-1])
        else:
            symbol_data["funding_rates"].append(0)

        if open_interest is not None:
            symbol_data["open_interest"].append(open_interest)
        elif symbol_data["open_interest"]:
            symbol_data["open_interest"].append(symbol_data["open_interest"][-1])
        else:
            symbol_data["open_interest"].append(0)

        if len(symbol_data["timestamps"]) > 1:
            cutoff_time = now - timedelta(hours=4)
            if symbol_data["timestamps"][0] < cutoff_time:
                valid_indices = [i for i, ts in enumerate(symbol_data["timestamps"]) if ts >= cutoff_time]
                if valid_indices:
                    start_idx = valid_indices[0]
                    symbol_data["timestamps"] = symbol_data["timestamps"][start_idx:]
                    symbol_data["spot_prices"] = symbol_data["spot_prices"][start_idx:]
                    symbol_data["futures_prices"] = symbol_data["futures_prices"][start_idx:]
                    symbol_data["premiums"] = symbol_data["premiums"][start_idx:]
                    symbol_data["funding_rates"] = symbol_data["funding_rates"][start_idx:]
                    symbol_data["open_interest"] = symbol_data["open_interest"][start_idx:]

        return spot_price, futures_price, premium, funding_rate, open_interest
    return None, None, None, funding_rate, open_interest


def create_premium_chart(symbol, symbol_data):
    if not symbol_data["timestamps"]:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=symbol_data["timestamps"], y=symbol_data["premiums"], mode='lines', line=dict(color='green')))
    fig.update_layout(height=300, margin=dict(l=40, r=40, t=50, b=30), yaxis_title="期现溢价率 (%)",
                      xaxis=dict(range=[datetime.now(CHINA_TZ) - timedelta(hours=4), datetime.now(CHINA_TZ)]))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    return fig


def create_funding_rate_chart(symbol, symbol_data):
    if not symbol_data["timestamps"]:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=symbol_data["timestamps"], y=symbol_data["funding_rates"], mode='lines', name='资金费率 (%)',
                   line=dict(color='red')))
    fig.update_layout(height=300, margin=dict(l=40, r=40, t=50, b=30), yaxis_title="资金费率 (%)",
                      xaxis=dict(range=[datetime.now(CHINA_TZ) - timedelta(hours=4), datetime.now(CHINA_TZ)]))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    return fig


def create_open_interest_chart(symbol, symbol_data):
    if not symbol_data["timestamps"]:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=symbol_data["timestamps"], y=symbol_data["open_interest"], mode='lines', name='持仓量',
                             line=dict(color='blue')))
    fig.update_layout(height=300, margin=dict(l=40, r=40, t=50, b=30), yaxis_title="持仓量",
                      xaxis=dict(range=[datetime.now(CHINA_TZ) - timedelta(hours=4), datetime.now(CHINA_TZ)]))
    return fig


def load_historical_data(symbol, symbol_data):
    if not symbol_data["historical_data_loaded"]:
        with st.spinner(f"正在加载 {symbol} 历史数据..."):
            timestamps, spot_prices, futures_prices, premiums = get_historical_klines(symbol, "1m", 240)
            funding_timestamps, funding_rates = get_historical_funding_rates(symbol)
            oi_timestamps, open_interests = get_historical_open_interest(symbol)

            if timestamps:
                symbol_data["timestamps"] = timestamps
                symbol_data["spot_prices"] = spot_prices
                symbol_data["futures_prices"] = futures_prices
                symbol_data["premiums"] = premiums

                if funding_rates:
                    mapped_funding_rates = []
                    for ts in timestamps:
                        closest_idx = min(range(len(funding_timestamps)),
                                          key=lambda i: abs((ts - funding_timestamps[i]).total_seconds()))
                        mapped_funding_rates.append(
                            funding_rates[closest_idx] if closest_idx < len(funding_rates) else 0)
                    symbol_data["funding_rates"] = mapped_funding_rates
                else:
                    symbol_data["funding_rates"] = [0] * len(timestamps)

                if open_interests:
                    mapped_open_interests = []
                    for ts in timestamps:
                        closest_idx = min(range(len(oi_timestamps)),
                                          key=lambda i: abs((ts - oi_timestamps[i]).total_seconds()))
                        mapped_open_interests.append(
                            open_interests[closest_idx] if closest_idx < len(open_interests) else 0)
                    symbol_data["open_interest"] = mapped_open_interests
                else:
                    symbol_data["open_interest"] = [0] * len(timestamps)

                funding_rate = get_funding_rate(symbol)
                open_interest = get_open_interest(symbol)
                if funding_rate is not None:
                    symbol_data["last_funding_rate"] = funding_rate
                    if symbol_data["funding_rates"]:
                        symbol_data["funding_rates"][-1] = funding_rate * 100
                if open_interest is not None and symbol_data["open_interest"]:
                    symbol_data["open_interest"][-1] = open_interest

                symbol_data["historical_data_loaded"] = True
                return True
            return False
    return True


# 后台费率跟踪器类
class BinanceFundingRateTracker:
    def __init__(self, data_file=STATS_FILE):
        self.data_file = data_file
        self.previous_rates = {}
        self.current_rates = {}

        # 确保文件存在并有效
        if not os.path.exists(self.data_file):
            logger.info(f"创建新的统计文件: {self.data_file}")
            self._create_empty_stats_file()
        else:
            data = safe_read_json(self.data_file)
            if data is None:
                logger.warning(f"统计文件无效，创建新文件: {self.data_file}")
                self._create_empty_stats_file()
            else:
                self.previous_rates = data.get('previous_rates', {})
                self.current_rates = self.previous_rates.copy()  # 初始化时加载上一次数据
                logger.info(f"成功加载历史数据，包含 {len(self.previous_rates)} 个交易对")

    def _create_empty_stats_file(self):
        """创建一个空的统计文件"""
        empty_stats = {
            "timestamp": datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "highest_rates": [],
            "lowest_rates": [],
            "biggest_increases": [],
            "biggest_decreases": [],
            "previous_rates": {}
        }
        safe_write_json(self.data_file, empty_stats)
        self.previous_rates = {}
        self.current_rates = {}

    def get_funding_rates(self):
        try:
            response = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex")
            if response.status_code != 200:
                logger.warning(f"获取资金费率API响应错误: {response.status_code}")
                return {}

            data = response.json()
            return {item['symbol']: float(item['lastFundingRate'])
                    for item in data if item['symbol'].endswith('USDT')}
        except Exception as e:
            logger.error(f"错误获取资金费率: {e}")
            return {}

    def get_top_n(self, rates, n, reverse=True):
        return sorted(rates.items(), key=lambda x: x[1], reverse=reverse)[:n]

    def get_biggest_changes(self, current, previous, n, increasing=True):
        changes = {s: current[s] - previous[s] for s in current if s in previous}
        if not changes:
            return []
        return sorted(changes.items(), key=lambda x: x[1], reverse=increasing)[:n]

    def run_task(self):
        logger.info(f"资金费率跟踪任务运行于 {datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
        self.current_rates = self.get_funding_rates()
        if not self.current_rates:
            logger.warning("获取资金费率失败，跳过本次运行")
            return

        highest_rates = self.get_top_n(self.current_rates, 5, reverse=True)
        lowest_rates = self.get_top_n(self.current_rates, 5, reverse=False)
        increasing_rates = decreasing_rates = []

        if self.previous_rates:
            increasing_rates = self.get_biggest_changes(self.current_rates, self.previous_rates, 5, increasing=True)
            decreasing_rates = self.get_biggest_changes(self.current_rates, self.previous_rates, 5, increasing=False)

        timestamp = datetime.now(CHINA_TZ).strftime('%Y-%m-%d %H:%M:%S')
        stats = {
            "timestamp": timestamp,
            "highest_rates": [{"symbol": s, "rate": r} for s, r in highest_rates],
            "lowest_rates": [{"symbol": s, "rate": r} for s, r in lowest_rates],
            "biggest_increases": [{"symbol": s, "change": c} for s, c in increasing_rates],
            "biggest_decreases": [{"symbol": s, "change": c} for s, c in decreasing_rates],
            "previous_rates": self.current_rates
        }

        if safe_write_json(self.data_file, stats):
            logger.info(f"数据已保存至 {self.data_file}")
            self.previous_rates = self.current_rates.copy()
        else:
            logger.error(f"保存数据失败")


# 前端 Streamlit 应用
def run_streamlit():
    st.set_page_config(page_title="加密货币费率监控系统", page_icon="📈", layout="wide",
                       initial_sidebar_state="expanded")

    # 初始化 session_state
    if 'symbol1' not in st.session_state:
        st.session_state.symbol1 = "AUCTIONUSDT"
        st.session_state.symbol1_data = {"timestamps": [], "spot_prices": [], "futures_prices": [], "premiums": [],
                                         "funding_rates": [], "open_interest": [], "last_funding_rate": None,
                                         "historical_data_loaded": False, "charts": [None, None, None],
                                         "running": False}
    if 'symbol2' not in st.session_state:
        st.session_state.symbol2 = "FUNUSDT"
        st.session_state.symbol2_data = {"timestamps": [], "spot_prices": [], "futures_prices": [], "premiums": [],
                                         "funding_rates": [], "open_interest": [], "last_funding_rate": None,
                                         "historical_data_loaded": False, "charts": [None, None, None],
                                         "running": False}
    if 'stats_data' not in st.session_state:
        st.session_state.stats_data = None
    if 'last_stats_update' not in st.session_state:
        st.session_state.last_stats_update = None
    # 添加一个新的session_state来跟踪按钮状态，避免重复key问题
    if 'button_keys' not in st.session_state:
        st.session_state.button_keys = {
            'symbol1_toggle': f"toggle_symbol1_{int(time.time())}",
            'symbol2_toggle': f"toggle_symbol2_{int(time.time())}",
            'reload_stats': f"reload_stats_{int(time.time())}"
        }

    UPDATE_INTERVAL = 10

    def load_stats_data():
        try:
            data = safe_read_json(STATS_FILE)
            if data:
                st.session_state.stats_data = data
                st.session_state.last_stats_update = datetime.now(CHINA_TZ)
                return data
            else:
                logger.warning("统计数据为空或无效")
                return None
        except Exception as e:
            logger.error(f"读取统计数据出错: {e}")
            return None

    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #ffffff;'>⚔️ 合约费率战法监控系统</h1>",
                    unsafe_allow_html=True)
        st.title("🛰️监控设置")

        col1, col2 = st.columns(2)
        with col1:
            new_symbol1 = st.text_input("交易对1", value=st.session_state.symbol1, placeholder="例如: FUNUSDT",
                                        key="symbol1_input", label_visibility="collapsed")
        with col2:
            new_symbol2 = st.text_input("交易对2", value=st.session_state.symbol2, placeholder="例如: AUCTIONUSDT",
                                        key="symbol2_input", label_visibility="collapsed")

        if new_symbol1 != st.session_state.symbol1:
            st.session_state.symbol1 = new_symbol1
            st.session_state.symbol1_data = {"timestamps": [], "spot_prices": [], "futures_prices": [], "premiums": [],
                                             "funding_rates": [], "open_interest": [], "last_funding_rate": None,
                                             "historical_data_loaded": False, "charts": [None, None, None],
                                             "running": False}
            # 更新按钮key以避免重复
            st.session_state.button_keys['symbol1_toggle'] = f"toggle_symbol1_{int(time.time())}"
            st.experimental_rerun()
        if new_symbol2 != st.session_state.symbol2:
            st.session_state.symbol2 = new_symbol2
            st.session_state.symbol2_data = {"timestamps": [], "spot_prices": [], "futures_prices": [], "premiums": [],
                                             "funding_rates": [], "open_interest": [], "last_funding_rate": None,
                                             "historical_data_loaded": False, "charts": [None, None, None],
                                             "running": False}
            # 更新按钮key以避免重复
            st.session_state.button_keys['symbol2_toggle'] = f"toggle_symbol2_{int(time.time())}"
            st.experimental_rerun()

        col1, col2 = st.columns(2)
        with col1:
            # 使用动态生成的唯一key
            if st.button('1️⃣停止监控' if st.session_state.symbol1_data["running"] else '1️⃣开始监控',
                         key=st.session_state.button_keys['symbol1_toggle']):
                st.session_state.symbol1_data["running"] = not st.session_state.symbol1_data["running"]
                if st.session_state.symbol1_data["running"]:
                    success = load_historical_data(st.session_state.symbol1, st.session_state.symbol1_data)
                    if not success:
                        st.error(f"无法加载 {st.session_state.symbol1} 历史数据")
                        st.session_state.symbol1_data["running"] = False
                # 更新按钮key以避免重复
                st.session_state.button_keys['symbol1_toggle'] = f"toggle_symbol1_{int(time.time())}"
                st.experimental_rerun()
        with col2:
            # 使用动态生成的唯一key
            if st.button('2️⃣停止监控' if st.session_state.symbol2_data["running"] else '2️⃣开始监控',
                         key=st.session_state.button_keys['symbol2_toggle']):
                st.session_state.symbol2_data["running"] = not st.session_state.symbol2_data["running"]
                if st.session_state.symbol2_data["running"]:
                    success = load_historical_data(st.session_state.symbol2, st.session_state.symbol2_data)
                    if not success:
                        st.error(f"无法加载 {st.session_state.symbol2} 历史数据")
                        st.session_state.symbol2_data["running"] = False
                # 更新按钮key以避免重复
                st.session_state.button_keys['symbol2_toggle'] = f"toggle_symbol2_{int(time.time())}"
                st.experimental_rerun()

        st.markdown("---")
        st.subheader("📊 统计数据")
        stats_placeholder = st.empty()
        if st.session_state.stats_data is None:
            load_stats_data()

        # 显示统计数据
        def display_stats_data(placeholder):
            with placeholder.container():
                if st.session_state.stats_data:
                    data = st.session_state.stats_data
                    timestamp = data.get("timestamp", "未知")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("😱 **费率最高的交易对**")
                        if "highest_rates" in data and data["highest_rates"]:
                            df_highest = pd.DataFrame(
                                [{"交易对": f"🟢 {item['symbol']}", "费率": f"{item['rate'] * 100:.2f}%"}
                                 for item in data["highest_rates"]])
                            st.dataframe(df_highest, hide_index=True)
                        else:
                            st.info("暂无数据")
                    with col2:
                        st.write("😍 **费率最低的交易对**")
                        if "lowest_rates" in data and data["lowest_rates"]:
                            df_lowest = pd.DataFrame(
                                [{"交易对": f"🔴 {item['symbol']}", "费率": f"{item['rate'] * 100:.2f}%"}
                                 for item in data["lowest_rates"]])
                            st.dataframe(df_lowest, hide_index=True)
                        else:
                            st.info("暂无数据")
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write("⬆️ **费率上升最快**")
                        if "biggest_increases" in data and data["biggest_increases"]:
                            df_increases = pd.DataFrame(
                                [{"交易对": item["symbol"], "变化": f"{item['change'] * 100:.4f}%"}
                                 for item in data["biggest_increases"]])
                            st.dataframe(df_increases, hide_index=True)
                        else:
                            st.info("暂无数据")
                    with col4:
                        st.write("⬇️ **费率下降最快**")
                        if "biggest_decreases" in data and data["biggest_decreases"]:
                            df_decreases = pd.DataFrame(
                                [{"交易对": item["symbol"], "变化": f"{item['change'] * 100:.4f}%"}
                                 for item in data["biggest_decreases"]])
                            st.dataframe(df_decreases, hide_index=True)
                        else:
                            st.info("暂无数据")
                    st.caption(f"更新时间: {timestamp}")
                else:
                    st.warning("未能加载统计数据，请稍后再试")
                    # 使用动态生成的唯一key
                    if st.button("重新加载统计数据", key=st.session_state.button_keys['reload_stats']):
                        load_stats_data()
                        # 更新按钮key以避免重复
                        st.session_state.button_keys['reload_stats'] = f"reload_stats_{int(time.time())}"
                        st.experimental_rerun()


        # 显示统计数据
        def display_stats_data(placeholder):
            with placeholder.container():
                if st.session_state.stats_data:
                    data = st.session_state.stats_data
                    timestamp = data.get("timestamp", "未知")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("😱 **费率最高的交易对**")
                        if "highest_rates" in data and data["highest_rates"]:
                            df_highest = pd.DataFrame(
                                [{"交易对": f"🟢 {item['symbol']}", "费率": f"{item['rate'] * 100:.2f}%"}
                                 for item in data["highest_rates"]])
                            st.dataframe(df_highest, hide_index=True)
                        else:
                            st.info("暂无数据")
                    with col2:
                        st.write("😍 **费率最低的交易对**")
                        if "lowest_rates" in data and data["lowest_rates"]:
                            df_lowest = pd.DataFrame(
                                [{"交易对": f"🔴 {item['symbol']}", "费率": f"{item['rate'] * 100:.2f}%"}
                                 for item in data["lowest_rates"]])
                            st.dataframe(df_lowest, hide_index=True)
                        else:
                            st.info("暂无数据")
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write("⬆️ **费率上升最快**")
                        if "biggest_increases" in data and data["biggest_increases"]:
                            df_increases = pd.DataFrame(
                                [{"交易对": item["symbol"], "变化": f"{item['change'] * 100:.4f}%"}
                                 for item in data["biggest_increases"]])
                            st.dataframe(df_increases, hide_index=True)
                        else:
                            st.info("暂无数据")
                    with col4:
                        st.write("⬇️ **费率下降最快**")
                        if "biggest_decreases" in data and data["biggest_decreases"]:
                            df_decreases = pd.DataFrame(
                                [{"交易对": item["symbol"], "变化": f"{item['change'] * 100:.4f}%"}
                                 for item in data["biggest_decreases"]])
                            st.dataframe(df_decreases, hide_index=True)
                        else:
                            st.info("暂无数据")
                    st.caption(f"更新时间: {timestamp}")
                else:
                    st.warning("未能加载统计数据，请稍后再试")
                    # 尝试重新加载
                    if st.button("重新加载统计数据"):
                        load_stats_data()
                        st.experimental_rerun()

        # 显示初始统计数据
        display_stats_data(stats_placeholder)

    title_placeholder1 = st.empty()
    metrics_placeholder1 = st.empty()
    symbol1_container = st.container()
    title_placeholder2 = st.empty()
    metrics_placeholder2 = st.empty()
    symbol2_container = st.container()

    with symbol1_container:
        chart_col1_1, chart_col1_2, chart_col1_3 = st.columns(3)
        with chart_col1_1:
            chart1_premium = st.empty()
        with chart_col1_2:
            chart1_funding = st.empty()
        with chart_col1_3:
            chart1_oi = st.empty()

    with symbol2_container:
        chart_col2_1, chart_col2_2, chart_col2_3 = st.columns(3)
        with chart_col2_1:
            chart2_premium = st.empty()
        with chart_col2_2:
            chart2_funding = st.empty()
        with chart_col2_3:
            chart2_oi = st.empty()

    last_stats_refresh = time.time()
    while True:
        current_time = time.time()
        # 每分钟刷新一次统计数据
        if current_time - last_stats_refresh > 60:
            logger.info("刷新统计数据")
            data = load_stats_data()
            last_stats_refresh = current_time
            display_stats_data(stats_placeholder)

        current_time_str = datetime.now(CHINA_TZ).strftime("%Y-%m-%d %H:%M:%S CST")

        # 更新交易对1数据
        if st.session_state.symbol1_data["running"]:
            if not st.session_state.symbol1_data["historical_data_loaded"]:
                success = load_historical_data(st.session_state.symbol1, st.session_state.symbol1_data)
                if not success:
                    st.error(f"无法加载 {st.session_state.symbol1} 历史数据")
                    st.session_state.symbol1_data["running"] = False
                    st.experimental_rerun()

            try:
                spot_price1, futures_price1, premium1, funding_rate1, open_interest1 = update_data(
                    st.session_state.symbol1, st.session_state.symbol1_data)
                if spot_price1 is not None and futures_price1 is not None:
                    title_placeholder1.markdown(f"### 1️⃣ {st.session_state.symbol1} 当前数据 - ({current_time_str})")
                    with metrics_placeholder1.container():
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric(label="现货价格", value=format_price(spot_price1))
                        col2.metric(label="期货价格", value=format_price(futures_price1))
                        col3.metric(label="期现溢价", value=f"{'🟢' if premium1 > 0 else '🔴'} {premium1:.2f}%")
                        col4.metric(label="资金费率",
                                    value=f"{'🟢' if funding_rate1 > 0 else '🔴'} {funding_rate1 * 100:.2f}%")
                        col5.metric(label="持仓量", value=f"{open_interest1:,.0f}")

                premium_fig1 = create_premium_chart(st.session_state.symbol1, st.session_state.symbol1_data)
                funding_fig1 = create_funding_rate_chart(st.session_state.symbol1, st.session_state.symbol1_data)
                open_interest_fig1 = create_open_interest_chart(st.session_state.symbol1, st.session_state.symbol1_data)

                if premium_fig1:
                    chart1_premium.plotly_chart(premium_fig1, use_container_width=True)
                if funding_fig1:
                    chart1_funding.plotly_chart(funding_fig1, use_container_width=True)
                if open_interest_fig1:
                    chart1_oi.plotly_chart(open_interest_fig1, use_container_width=True)
            except Exception as e:
                logger.error(f"更新交易对1数据出错: {e}")
                st.error(f"更新 {st.session_state.symbol1} 数据时出错")

        # 更新交易对2数据
        if st.session_state.symbol2_data["running"]:
            if not st.session_state.symbol2_data["historical_data_loaded"]:
                success = load_historical_data(st.session_state.symbol2, st.session_state.symbol2_data)
                if not success:
                    st.error(f"无法加载 {st.session_state.symbol2} 历史数据")
                    st.session_state.symbol2_data["running"] = False
                    st.experimental_rerun()

            try:
                spot_price2, futures_price2, premium2, funding_rate2, open_interest2 = update_data(
                    st.session_state.symbol2, st.session_state.symbol2_data)
                if spot_price2 is not None and futures_price2 is not None:
                    title_placeholder2.markdown(f"### 2️⃣ {st.session_state.symbol2} 当前数据 - ({current_time_str})")
                    with metrics_placeholder2.container():
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric(label="现货价格", value=format_price(spot_price2))
                        col2.metric(label="期货价格", value=format_price(futures_price2))
                        col3.metric(label="期现溢价", value=f"{'🟢' if premium2 > 0 else '🔴'} {premium2:.2f}%")
                        col4.metric(label="资金费率",
                                    value=f"{'🟢' if funding_rate2 > 0 else '🔴'} {funding_rate2 * 100:.2f}%")
                        col5.metric(label="持仓量", value=f"{open_interest2:,.0f}")

                premium_fig2 = create_premium_chart(st.session_state.symbol2, st.session_state.symbol2_data)
                funding_fig2 = create_funding_rate_chart(st.session_state.symbol2, st.session_state.symbol2_data)
                open_interest_fig2 = create_open_interest_chart(st.session_state.symbol2, st.session_state.symbol2_data)

                if premium_fig2:
                    chart2_premium.plotly_chart(premium_fig2, use_container_width=True)
                if funding_fig2:
                    chart2_funding.plotly_chart(funding_fig2, use_container_width=True)
                if open_interest_fig2:
                    chart2_oi.plotly_chart(open_interest_fig2, use_container_width=True)
            except Exception as e:
                logger.error(f"更新交易对2数据出错: {e}")
                st.error(f"更新 {st.session_state.symbol2} 数据时出错")

        time.sleep(UPDATE_INTERVAL)


# 后台线程运行函数
def run_tracker():
    logger.info("启动资金费率跟踪器后台线程")
    try:
        tracker = BinanceFundingRateTracker()
        tracker.run_task()  # 首次运行
        schedule.every(5).minutes.do(tracker.run_task)
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"跟踪器运行出错: {e}")
                time.sleep(60)  # 出错后等待一分钟再重试
    except Exception as e:
        logger.critical(f"跟踪器线程崩溃: {e}")


# 主函数：一个命令启动前端和后端
def main():
    # 确保统计文件目录存在
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)

    logger.info(f"启动加密货币费率监控系统，统计文件路径: {STATS_FILE}")

    # 启动后端线程
    tracker_thread = threading.Thread(target=run_tracker, daemon=True)
    tracker_thread.start()
    logger.info("后端跟踪器已在线程中启动")

    # 运行前端
    run_streamlit()


if __name__ == "__main__":
    main()
