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

# 设置页面配置
st.set_page_config(
    page_title="分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API配置
OPENAI_API_KEY = "xxx"  # 更新为自己在tu-zi中的API
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

        prompt = (
            f"基于{symbol}期货市场的以下关键数据进行专业市场行为分析：\n\n"
            f"**市场状态指标**\n"
            f"- 当前持仓量：{position_data['current_oi']:,.0f}\n"
            f"- 持仓分位数：{position_data['percentile']:.2f}%（高位>80%，低位<20%）\n"
            f"- 24小时波动率：{volatility:.2f}%\n\n"
            f"**多周期价格与持仓对比**\n"
            f"| 周期 | 价格变化 | 持仓变化 |\n"
            f"|------|----------|----------|\n"
            f"| 1小时 | {price_data.get('1h_change', 0):.2f}% | {position_data['changes']['1小时']['change_percentage']:.2f}% |\n"
            f"| 4小时 | {price_data.get('4h_change', 0):.2f}% | {position_data['changes']['4小时']['change_percentage']:.2f}% |\n"
            f"| 24小时 | {price_data.get('24h_change', 0):.2f}% | {position_data['changes']['24小时']['change_percentage']:.2f}% |\n\n"

            f"请提供专业的市场行为分析报告，包括以下部分：\n"
            f"1. **市场行为模式识别**：基于价格与持仓量变化关系识别当前市场行为模式\n"
            f"2. **多空力量对比**：确定当前主导方向及其强度\n"
            f"3. **交易策略建议**：提供具体的操作思路和关键价格水平\n\n"

            f"请使用Markdown格式，结构清晰，语言简洁专业，突出可操作信号，避免冗余描述。"
            f"使用表格对比数据，关键指标用**粗体**标注，明确标示操作建议的置信度。"
        )
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7
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

        prompt = (
            f"基于Binance期货市场USDT交易对的持仓量变化数据进行专业市场分析：\n\n"
            f"**持仓增加最显著的前10个交易对**\n"
            f"```\n{top_increase[['symbol', 'change_percentage']].to_string()}\n```\n\n"
            f"**持仓减少最显著的前10个交易对**\n"
            f"```\n{top_decrease[['symbol', 'change_percentage']].to_string()}\n```\n\n"

            f"请提供专业的持仓分析报告，包括以下部分：\n"
            f"1. **市场情绪分析**：基于持仓变化评估整体市场情绪\n"
            f"2. **资金流向解读**：分析主要资金流入/流出的币种类别\n"
            f"3. **交易策略建议**：提供重点关注币种及其选择理由\n\n"

            f"请使用Markdown格式，结构为：[市场情绪分析]→[资金流向解读]→[交易策略建议]。"
            f"语言简洁专业，突出可操作信号，避免冗余描述。"
            f"使用表格对比不同币种类别的资金流向特征，关键指标用**粗体**标注。"
        )
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"生成持仓分析报告失败: {e}")
            return "AI分析生成失败，请稍后重试"

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

    # 创建趋势表格数据
    trend_table_data = []
    for tf in ['5m', '15m', '1h', '4h', '1d']:
        if tf in trends:
            trend_table_data.append({
                "周期": TIMEFRAMES[tf]['name'],
                "趋势": trends[tf]['trend'],
                "RSI": f"{trends[tf]['rsi']:.2f}",
                "支撑位": f"{trends[tf]['support']:.2f}",
                "阻力位": f"{trends[tf]['resistance']:.2f}",
                "成交量": trends[tf]['volume_trend']
            })

    # 构建更结构化的提示
    prompt = (
        f"请生成一份关于{symbol}的多周期技术分析报告，格式必须严格按照以下结构：\n\n"

        f"## 市场结构\n"
        f"当前市场呈现 **[主要特征描述]** 特征：\n"
        f"- **短期动能**：[5分钟和15分钟周期分析]\n"
        f"- **中期走势**：[1小时和4小时周期分析]\n"
        f"- **长期趋势**：[日线周期分析]\n\n"
        f"市场结构处于 **[结构定位]** 阶段，[额外市场特征描述]\n\n"

        f"## 技术面分析\n"
        f"### 关键指标交互验证\n"
        f"1. **价格结构**：\n"
        f"   - [支撑位分析]\n"
        f"   - [阻力位分析]\n\n"
        f"2. **动量指标**：\n"
        f"   - [RSI分析]\n"
        f"   - [其他动量指标分析]\n\n"
        f"3. **量价配合**：\n"
        f"   - [成交量与价格关系分析]\n"
        f"   - [成交量趋势分析]\n\n"

        f"## 操作建议\n"
        f"### 短期策略（日内交易）\n"
        f"**多头策略**：\n"
        f"- 入场：[入场点位和条件]\n"
        f"- 目标：[目标价位]\n"
        f"- 止损：[止损价位]\n\n"
        f"**空头策略**：\n"
        f"- 触发条件：[触发条件]\n"
        f"- 目标：[目标价位]\n"
        f"- 止损：[止损价位]\n\n"
    )

    # 添加数据信息
    data_info = (
        f"基于以下数据进行分析：\n"
        f"- 当前价格：{current_price:.2f}\n"
        f"- 短期趋势（5m/15m）：{short_term_trend}\n"
        f"- 中期趋势（1h/4h）：{medium_term_trend}\n"
        f"- RSI指标：{avg_rsi:.2f}\n"
        f"- 支撑位：{trends.get('1h', {}).get('support', 'N/A'):.2f}\n"
        f"- 阻力位：{trends.get('1h', {}).get('resistance', 'N/A'):.2f}\n"
        f"- 风险等级：{risk['level']}\n"
        f"- 风险因素：{', '.join(risk['factors']) if risk['factors'] else '无重大风险'}\n"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system",
                 "content": "你是一位专业的加密货币技术分析师，请严格按照指定格式生成分析报告，不要添加任何额外的章节或标题。"},
                {"role": "user", "content": prompt + data_info}
            ],
            temperature=0.5,  # 降低温度以获得更一致的输出
            max_tokens=4000
        )
        ai_analysis = response.choices[0].message.content

        # 确保格式一致性
        if not ai_analysis.startswith(f"# {symbol} 技术分析报告"):
            ai_analysis = f"# {symbol} 技术分析报告\n\n" + ai_analysis

        # 确保主要部分存在
        required_sections = ["## 市场结构", "## 技术面分析", "## 操作建议"]
        for section in required_sections:
            if section not in ai_analysis:
                ai_analysis = ai_analysis.replace(section.replace("##", "#"), section)
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
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
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

# 运行主程序
if __name__ == "__main__":
    main()

