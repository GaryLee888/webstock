import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from datetime import timedelta
import io
import matplotlib.font_manager as fm
import os

# --- 設定頁面與中文字型 ---
st.set_page_config(layout="wide", page_title="Gary's 決策系統 V60.10 (Web版)")

# 嘗試設定中文字型 (相容 Windows 開發與 Linux 部署)
plt.rcParams['axes.unicode_minus'] = False
font_path = None

# 檢查常見的 Linux 中文字型路徑 (Streamlit Cloud 適用)
possible_fonts = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
]
for p in possible_fonts:
    if os.path.exists(p):
        font_prop = fm.FontProperties(fname=p)
        plt.rcParams['font.family'] = font_prop.get_name()
        font_path = p
        break

# 如果找不到 Linux 字型，則嘗試 Windows 字型
if font_path is None:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']

# --- 風格配色 ---
COLORS = {
    "bull": "#e74c3c",          # 紅
    "bear": "#27ae60",          # 綠
    "neutral": "#7f8c8d", 
    "wave": "#2980b9", 
    "predict_optimistic": "#e74c3c", 
    "predict_median": "#8e44ad",      
    "predict_pessimistic": "#27ae60",
    "predict_fill": "#d7bde2",
}

# --- 核心邏輯區 ---

@st.cache_data(ttl=3600)
def load_all_tw_stocks():
    """下載上市櫃興櫃清單，含偽裝 Header 防止被證交所阻擋"""
    tw_stock_map = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        # 2=上市, 4=上櫃, 5=興櫃
        sources = [(2, '.TW', '上市'), (4, '.TWO', '上櫃'), (5, '.TWO', '興櫃')]
        for mode, suffix, _ in sources:
            url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
            try:
                res = requests.get(url, headers=headers, timeout=10)
                res.encoding = 'big5' 
                
                # 嘗試解析 HTML
                dfs = pd.read_html(io.StringIO(res.text))
                if dfs and len(dfs) > 0:
                    df = dfs[0]
                    if df.shape[1] > 1:
                        # 跳過標頭，通常從第2行開始資料
                        for _, row in df.iloc[2:].iterrows():
                            try:
                                code_name = row[0]
                                if not isinstance(code_name, str): continue
                                parts = code_name.split()
                                if len(parts) >= 2:
                                    code = parts[0].strip()
                                    name = parts[1].strip()
                                    if code.isdigit() and len(code) == 4:
                                        tw_stock_map[code] = f"{code}{suffix}"
                                        tw_stock_map[name] = f"{code}{suffix}"
                            except: continue
            except Exception as e:
                pass
        return tw_stock_map
    except:
        return {}

def adjust_to_tick(price, return_str=True):
    price = float(price)
    if price < 10: val = round(price, 2); fmt = "{:.2f}"
    elif price < 50: val = round(price * 20) / 20.0; fmt = "{:.2f}"
    elif price < 100: val = round(price * 10) / 10.0; fmt = "{:.1f}"
    elif price < 500: val = round(price * 2) / 2.0; fmt = "{:.1f}"
    elif price < 1000: val = round(price); fmt = "{:.0f}"
    else: val = round(price / 5) * 5; fmt = "{:.0f}"
    return fmt.format(val) if return_str else val

def calc_indicators(df):
    try:
        for w in [5, 10, 20, 60]: df[f'MA{w}'] = df['Close'].rolling(window=w).mean()
        df['VMA20'] = df['Volume'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BB_H'] = df['MA20'] + (std20 * 2)
        df['BB_L'] = df['MA20'] - (std20 * 2)
        df['BB_W'] = (df['BB_H'] - df['BB_L']) / df['MA20']
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_D'] = df['MACD'] - df['MACD_Signal']
        
        low14 = df['Low'].rolling(window=14).min()
        high14 = df['High'].rolling(window=14).max()
        rsv = (df['Close'] - low14) / (high14 - low14) * 100
        df['K'] = rsv.ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        df['WR'] = (high14 - df['Close']) / (high14 - low14) * -100
        df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        df['BIAS5'] = (df['Close'] - df['MA5']) / df['MA5'] * 100
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        return df
    except: return df

def calc_zigzag(df):
    n = 3 
    df = df.copy()
    highs = df['High'].values
    lows = df['Low'].values
    high_idx = argrelextrema(highs, np.greater_equal, order=n)[0]
    low_idx = argrelextrema(lows, np.less_equal, order=n)[0]
    
    df['Wave_High'] = np.nan
    df['Wave_Low'] = np.nan
    
    for idx in high_idx:
        df.iloc[idx, df.columns.get_loc('Wave_High')] = highs[idx]
    for idx in low_idx:
        df.iloc[idx, df.columns.get_loc('Wave_Low')] = lows[idx]

    pivots = []
    for i in range(len(df)):
        if pd.notna(df['Wave_High'].iloc[i]): pivots.append((i, df['Wave_High'].iloc[i], 'High'))
        elif pd.notna(df['Wave_Low'].iloc[i]): pivots.append((i, df['Wave_Low'].iloc[i], 'Low'))
    
    clean = []
    if not pivots: return []
    last_type = None
    for p in pivots:
        if p[2] != last_type: clean.append(p); last_type = p[2]
        else:
            if last_type == 'High': 
                if p[1] > clean[-1][1]: clean[-1] = p
            else: 
                if p[1] < clean[-1][1]: clean[-1] = p
    return clean

def predict_monte_carlo(prices, forecast_days=10, simulations=1000):
    try:
        log_returns = np.log(1 + prices.pct_change())
        u = log_returns.mean()
        var = log_returns.var()
        drift = u - (0.5 * var)
        stdev = log_returns.std()
        daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (forecast_days, simulations)))
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = prices.iloc[-1] * daily_returns[0]
        for t in range(1, forecast_days):
            price_paths[t] = price_paths[t-1] * daily_returns[t]
        p90 = np.percentile(price_paths, 90, axis=1)
        p50 = np.percentile(price_paths, 50, axis=1)
        p10 = np.percentile(price_paths, 10, axis=1)
        return p90, p50, p10
    except:
        return None, None, None

def resolve_symbol(query, tw_stock_map):
    query = query.strip().upper()
    
    # 1. 優先查快取
    if query in tw_stock_map: 
        return tw_stock_map[query], query
    
    # 2. 如果是數字，直接猜測
    if query.isdigit():
        return f"{query}.TW", query 
        
    # 3. 備援：用證交所 API 查詢 (處理離線或新股)
    try:
        url = f"https://www.twse.com.tw/rwd/zh/api/codeQuery?query={query}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3).json()
        if 'suggestions' in res and res['suggestions']:
            # 回傳格式範例: "3033\t威健\t上市" 或 "6695\t貝爾威勒\t興櫃"
            first_match = res['suggestions'][0]
            parts = first_match.split('\t')
            
            if len(parts) >= 2:
                code = parts[0]
                name = parts[1]
                
                # 判斷市場別 (如果有的話)
                suffix = ".TW" # 預設上市
                if len(parts) >= 3:
                    market_type = parts[2]
                    if "上櫃" in market_type or "興櫃" in market_type:
                        suffix = ".TWO"
                
                return f"{code}{suffix}", name
    except Exception as e:
        print(f"API Search Error: {e}")
        pass

    return query, query

# --- 主介面 ---

st.title("Gary's 決策系統 V60.10 - 興櫃彈性容錯 Web版")

# 側邊欄或頂部輸入
col_input, col_status = st.columns([3, 1])
with col_input:
    stock_input = st.text_input("輸入股票代號或名稱 (例如: 2330, 鴻海)", value="威健")
    
tw_map = load_all_tw_stocks()
with col_status:
    if tw_map:
        st.success(f"已連線 (資料庫: {len(tw_map)} 檔)")
    else:
        st.warning("⚠️ 離線模式 (啟用備援搜尋)")

if st.button("🔍 智能分析", type="primary"):
    with st.spinner('正在分析大數據與計算蒙地卡羅模擬...'):
        try:
            symbol, name_query = resolve_symbol(stock_input, tw_map)
            # 顯示名稱邏輯
            display_name = list(tw_map.keys())[list(tw_map.values()).index(symbol)] if symbol in tw_map.values() else name_query

            # 下載資料
            df = yf.download(symbol, period="1y", progress=False)
            if df.empty:
                st.error(f"找不到 {symbol} 的資料 (Yahoo Finance 回傳空值)。")
                st.info("提示：若為興櫃股票，可能因成交量過低導致資料抓取失敗。")
                st.stop()
            
            # 處理 MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            data_len = len(df)
            enable_prediction = True
            if data_len < 10:
                st.error("資料不足 10 筆，無法分析")
                st.stop()
            elif data_len < 60:
                st.warning(f"注意：資料僅 {data_len} 筆，已自動關閉預測功能")
                enable_prediction = False

            df = calc_indicators(df)
            plot_df = df.tail(100).copy()
            pivots = calc_zigzag(plot_df)
            
            last = df.iloc[-1].copy()
            cp = float(last['Close'])

            # 數值計算
            raw_entry = (last['Open'] + last['High'] + last['Low'] + (last['Close'] * 2)) / 5
            atr = float(last['ATR']) if last['ATR'] > 0 else cp*0.02
            raw_sl = raw_entry - (atr * 2.0)
            raw_tp = raw_entry + (atr * 3.2)
            
            smart_entry_str = adjust_to_tick(raw_entry)
            smart_sl_str = adjust_to_tick(raw_sl)
            smart_tp_str = adjust_to_tick(raw_tp)
            smart_sl_val = adjust_to_tick(raw_sl, return_str=False)
            smart_tp_val = adjust_to_tick(raw_tp, return_str=False)

            # 30 指標邏輯
            prev = df.iloc[-2]
            p_prev = df.iloc[-3] if len(df) > 3 else prev
            
            l30 = [
                ("股價站於月線上", cp > last['MA20']), ("均線呈金叉狀態", last['MA5'] > last['MA20']),
                ("短期五日線向上", last['MA5'] > prev['MA5']), ("MACD紅柱遞增", last['MACD_D'] > 0 and last['MACD_D'] > prev['MACD_D']),
                ("KD低檔黃金交叉", last['K'] > last['D'] and prev['K'] < prev['D'] and last['K'] < 50), ("RSI處多方強勢位", last['RSI'] > 50),
                ("威廉指標進入強勢", last['WR'] > -50), ("今日爆量攻擊", last['Volume'] > last['VMA20']*1.5),
                ("今日收盤實體紅K", last['Close'] > last['Open']), ("突破布林通道上限", cp > last['BB_H']),
                ("低點不破昨低", last['Low'] >= prev['Low']), ("三日累漲幅度>3%", cp / p_prev['Close'] > 1.03),
                ("十日均線向上", last['MA10'] > prev['MA10']), ("成交量高於均量", last['Volume'] > last['VMA20']),
                ("RSI位階未過熱", last['RSI'] < 75), ("今日收盤創新高", cp > prev['High']),
                ("五日均量向上", last['VMA20'] > prev['VMA20']), ("乖離率適中", abs(last['BIAS20']) < 10),
                ("高點刷新昨高", last['High'] > prev['High']), ("尾盤作價收高", last['Close'] > (last['High']+last['Low'])/2),
                ("開盤具備缺口", last['Open'] > prev['Close']), ("MACD零軸上發散", last['MACD_D'] > 0),
                ("突破三日高點", cp > max(prev['High'], p_prev['High'])), ("ATR波動放大", last['ATR'] > prev['ATR']),
                ("KD呈多方排列", last['K'] > last['D']), ("5日乖離修正", abs(last['BIAS5']) < 5),
                ("季線支撐強勁", cp > last['MA60']), ("威廉指標向上", last['WR'] > prev['WR']),
                ("創20日收盤新高", cp == df['Close'].tail(20).max()), ("布林開口擴張", last['BB_W'] > df['BB_W'].iloc[-5])
            ]
            
            gene = [
                ("成交量異常噴發", last['Volume'] > last['VMA20']*2), ("均線多頭發散", last['MA5']>last['MA10']>last['MA20']),
                ("沿布林上軌推升", cp > last['BB_H']*0.99), ("創半年新高", cp >= df['Close'].tail(120).max()*0.98),
                ("MACD動能連三增", last['MACD_D'] > prev['MACD_D'] > p_prev['MACD_D'] > 0), ("5日線陡峭", last['MA5'] > prev['MA5']*1.02)
            ]
            
            score_30 = sum(100/30 for _, s in l30 if s)
            score_gene = sum(100/6 for _, s in gene if s)
            final_score = (score_30 * 0.7) + (score_gene * 0.3)

            # 預測
            mc_p90, mc_p50, mc_p10 = (None, None, None)
            if enable_prediction:
                forecast_data = df['Close'].tail(60) 
                mc_p90, mc_p50, mc_p10 = predict_monte_carlo(forecast_data, 10, 1000)

            # --- UI 呈現 ---
            col_report, col_chart = st.columns([1, 1.5])
            
            with col_report:
                st.markdown(f"### {display_name} ({symbol})")
                st.markdown(f"**現價**: {adjust_to_tick(cp)} | **ATR**: {atr:.2f}")
                
                # 分數卡片
                score_color = COLORS["bull"] if final_score >= 60 else COLORS["bear"]
                cmt = "🚀 鑽石飆股" if final_score >= 80 else "🔥 黃金強勢" if final_score >= 65 else "⚖️ 白銀震盪" if final_score >= 50 else "🐻 青銅弱勢"
                
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd; padding:10px; border-radius:5px; text-align:center;">
                        <span style="color:gray;">綜合評分</span><br>
                        <span style="font-size:40px; font-weight:bold; color:{score_color}">{final_score:.1f}</span><br>
                        <span style="background-color:{score_color}; color:white; padding:2px 10px; border-radius:3px;">{cmt}</span>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                # 預測區
                if enable_prediction and mc_p50 is not None:
                    target_p50 = mc_p50[-1]
                    p_text = "看漲" if target_p50 > cp else "看跌"
                    p_color = "red" if target_p50 > cp else "green"
                    st.markdown(f"""
                    **10日後預測**: <span style='color:{p_color}'>{p_text}</span> (中位 {adjust_to_tick(target_p50)})  
                    區間: {adjust_to_tick(mc_p10[-1])} ~ {adjust_to_tick(mc_p90[-1])}
                    """, unsafe_allow_html=True)
                
                # 操盤規劃
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("建議進場", smart_entry_str)
                c2.metric("精密停損", smart_sl_str, delta_color="inverse")
                c3.metric("黃金停利", smart_tp_str)

                # 基因與指標
                with st.expander("📊 查看詳細基因與指標", expanded=False):
                    st.write("**飆股基因**")
                    for desc, passed in gene:
                        icon = "🔴" if passed else "⚫" 
                        st.write(f"{icon} {desc}")
                    st.write("**30項技術指標**")
                    for desc, passed in l30:
                        icon = "🔴" if passed else "⚫"
                        st.write(f"{icon} {desc}")

            with col_chart:
                # 繪圖
                fig, ax = plt.subplots(figsize=(10, 6))
                
                dates_idx = np.arange(len(plot_df))
                opens, highs, lows, closes = plot_df['Open'], plot_df['High'], plot_df['Low'], plot_df['Close']
                
                # K線
                for i in dates_idx:
                    color = COLORS["bull"] if closes.iloc[i] >= opens.iloc[i] else COLORS["bear"]
                    ax.plot([i, i], [lows.iloc[i], highs.iloc[i]], color='black', linewidth=1, zorder=1)
                    h = abs(closes.iloc[i] - opens.iloc[i]) or 0.01
                    rect = plt.Rectangle((i - 0.3, min(opens.iloc[i], closes.iloc[i])), 0.6, h, color=color, zorder=2)
                    ax.add_patch(rect)

                ax.plot(dates_idx, plot_df['MA20'].values, color='#f39c12', label='20MA', linewidth=1.5)
                ax.plot(dates_idx, plot_df['MA60'].values, color='#2980b9', label='60MA', linewidth=1.5)

                # ZigZag
                if pivots:
                    px, py = zip(*[(p[0], p[1]) for p in pivots])
                    ax.plot(px, py, color=COLORS["wave"], linewidth=2, alpha=0.7, label='波浪')
                
                # 預測通道
                if enable_prediction and mc_p50 is not None:
                    last_idx = dates_idx[-1]
                    future_x = np.arange(last_idx, last_idx + 11)
                    start_price = closes.iloc[-1]
                    y_p90 = np.concatenate(([start_price], mc_p90))
                    y_p50 = np.concatenate(([start_price], mc_p50))
                    y_p10 = np.concatenate(([start_price], mc_p10))
                    
                    ax.plot(future_x, y_p90, color=COLORS["predict_optimistic"], linestyle='--', alpha=0.5)
                    ax.plot(future_x, y_p50, color=COLORS["predict_median"], linestyle='--', label='預測中位')
                    ax.plot(future_x, y_p10, color=COLORS["predict_pessimistic"], linestyle='--', alpha=0.5)
                    ax.fill_between(future_x, y_p10, y_p90, color=COLORS["predict_fill"], alpha=0.2)

                # 停損停利線
                ax.axhline(smart_tp_val, color=COLORS["bull"], linestyle=':', alpha=0.6)
                ax.axhline(smart_sl_val, color=COLORS["bear"], linestyle=':', alpha=0.6)

                ax.set_title(f"{display_name} 技術分析與預測", fontproperties=font_prop if font_path else None)
                ax.legend(prop=font_prop if font_path else None)
                ax.grid(True, linestyle=':', alpha=0.3)
                
                # 調整 X 軸日期標籤
                date_labels = [d.strftime('%m-%d') for d in plot_df.index]
                if enable_prediction:
                    last_date = plot_df.index[-1]
                    future_dates = pd.bdate_range(start=last_date, periods=11)[1:]
                    date_labels += [d.strftime('%m-%d') for d in future_dates]
                
                step = max(1, len(date_labels) // 10)
                ax.set_xticks(range(0, len(date_labels), step))
                ax.set_xticklabels(date_labels[::step], rotation=0)

                st.pyplot(fig)

        except Exception as e:
            st.error(f"分析發生錯誤: {str(e)}")
            st.exception(e)

# 頁腳
st.markdown("---")
st.caption("Gary's 決策系統 V60.10 Web版 - 僅供技術研究參考，不作為投資建議")
