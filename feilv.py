import streamlit as st
import pandas as pd
import time
import requests
import json
import os
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from decimal import Decimal, InvalidOperation
import schedule
from typing import Dict, List, Tuple, Optional
import argparse
import sys

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

# 后台费率跟踪器类
class BinanceFundingRateTracker:
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
                print(f"Error loading previous data: {e}")

    def get_usdt_perpetual_symbols(self) -> List[str]:
        try:
            response = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
            data = response.json()
            return [s['symbol'] for s in data['symbols'] 
                    if s['symbol'].endswith('USDT') and s['status'] == 'TRADING' and s['contractType'] == 'PERPETUAL']
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []

    def get_funding_rates(self) -> Dict[str, float]:
        try:
            response = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex")
            data = response.json()
            return {item['symbol']: float(item['lastFundingRate']) 
                    for item in data if item['symbol'].endswith('USDT')}
        except Exception as e:
            print(f"Error fetching funding rates: {e}")
            return {}

    def get_top_n(self, rates: Dict[str, float], n: int, reverse: bool = True) -> List[Tuple[str, float]]:
        return sorted(rates.items(), key=lambda x: x[1], reverse=reverse)[:n]

    def get_biggest_changes(self, current: Dict[str, float], previous: Dict[str, float], n: int, 
                            increasing: bool = True) -> List[Tuple[str, float]]:
        changes = {s: r - previous[s] for s, r in current.items() 
                   if s in previous and ((increasing and r > previous[s]) or (not increasing and r < previous[s]))}
        return sorted(changes.items(), key=lambda x: x[1], reverse=increasing)[:n]

    def run_task(self):
        print(f"Running task at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.current_rates = self.get_funding_rates()
        if not self.current_rates:
            print("Failed to get funding rates, skipping this run")
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
            print(f"Data saved to {self.data_file}")
        except Exception as e:
            print(f"Error saving data: {e}")

        self.previous_rates = self.current_rates.copy()

# 前端 Streamlit 应用
def run_streamlit():
    st.set_page_config(page_title="加密货币费率监控系统", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    
    # 初始化会话状态
    if 'symbol1' not in st.session_state:
        st.session_state.symbol1 = "AUCTIONUSDT"
        st.session_state.symbol1_data = {
            "timestamps": [], "spot_prices": [], "futures_prices": [], "premiums": [], 
            "funding_rates": [], "open_interest": [], "last_funding_rate": None, 
            "historical_data_loaded": False, "charts": [None, None, None], "running": False
        }
    if 'symbol2' not in st.session_state:
        st.session_state.symbol2 = "FUNUSDT"
        st.session_state.symbol2_data = {
            "timestamps": [], "spot_prices": [], "futures_prices": [], "premiums": [], 
            "funding_rates": [], "open_interest": [], "last_funding_rate": None, 
            "historical_data_loaded": False, "charts": [None, None, None], "running": False
        }
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'stats_data' not in st.session_state:
        st.session_state.stats_data = None
    if 'last_stats_update' not in st.session_state:
        st.session_state.last_stats_update = None

    # 常量
    UPDATE_INTERVAL = 10
    MAX_DATA_POINTS = 240
    HOURS_TO_DISPLAY = 4
    STATS_FILE = "funding_rates_stats.json"

    # 数据获取函数（略，与原代码一致，需添加）
    def get_spot_price(symbol): pass  # 实现略
    def get_futures_price(symbol): pass  # 实现略
    def get_funding_rate(symbol): pass  # 实现略
    def get_open_interest(symbol): pass  # 实现略
    def get_historical_klines(symbol, interval, limit): pass  # 实现略
    def get_historical_funding_rates(symbol, limit=MAX_DATA_POINTS): pass  # 实现略
    def get_historical_open_interest(symbol, period="5m", limit=MAX_DATA_POINTS): pass  # 实现略
    def update_data(symbol, symbol_data): pass  # 实现略
    def create_premium_chart(symbol, symbol_data): pass  # 实现略
    def create_funding_rate_chart(symbol, symbol_data): pass  # 实现略
    def create_open_interest_chart(symbol, symbol_data): pass  # 实现略
    def load_historical_data(symbol, symbol_data): pass  # 实现略
    def load_stats_data():
        try:
            if os.path.exists(STATS_FILE):
                with open(STATS_FILE, 'r') as f:
                    data = json.load(f)
                    st.session_state.stats_data = data
                    st.session_state.last_stats_update = datetime.now()
                    return data
            return None
        except Exception as e:
            st.error(f"读取统计数据出错: {e}")
            return None

    # 侧边栏和主界面逻辑（略，与原代码一致，需添加）
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #ffffff;'>⚔️ 合约费率战法监控系统</h1>", unsafe_allow_html=True)
        st.title("🛰️监控设置")
        # 实现略，参考原代码

    # 主界面（略，需添加完整逻辑）
    # 这里仅占位，完整实现参考原 Streamlit 主循环
    st.write("Streamlit 前端运行中...（请补充完整逻辑）")

# 主函数
def main():
    parser = argparse.ArgumentParser(description="加密货币费率监控系统")
    parser.add_argument('--mode', choices=['frontend', 'backend'], default='frontend', 
                        help="运行模式：frontend (Streamlit) 或 backend (跟踪器)")
    args = parser.parse_args()

    if args.mode == 'frontend':
        run_streamlit()
    elif args.mode == 'backend':
        tracker = BinanceFundingRateTracker()
        tracker.run_task()
        schedule.every(5).minutes.do(tracker.run_task)
        print("Funding rate tracker started. Press Ctrl+C to exit.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("Tracker stopped by user.")

if __name__ == "__main__":
    main()
