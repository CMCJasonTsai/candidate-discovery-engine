from __future__ import annotations

from datetime import datetime
from io import StringIO
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from src.candidate_engine import HORIZON_CFG, load_market_snapshot, load_valuation, run_engine

st.set_page_config(page_title="Candidate Discovery Engine", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

def check_access():
    try:
        expected = st.secrets.get("APP_ACCESS_CODE", "")
    except Exception:
        expected = ""
    if not expected or st.session_state.get("authorized"):
        return
    st.title("🔐 團隊存取驗證")
    code = st.text_input("請輸入團隊存取碼", type="password")
    if st.button("進入系統", type="primary"):
        if hashlib.sha256(code.encode()).hexdigest() == hashlib.sha256(str(expected).encode()).hexdigest():
            st.session_state["authorized"] = True
            st.rerun()
        else:
            st.error("存取碼不正確。")
    st.stop()

check_access()

@st.cache_data(ttl=900, show_spinner=False)
def cached_snapshot():
    return load_market_snapshot()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_valuation():
    return load_valuation()

@st.cache_data(ttl=1800, show_spinner=False)
def cached_run(universe_json, horizon, deep_scan_n, min_avg_amount):
    universe = pd.read_json(StringIO(universe_json), orient="split")
    result, _ = run_engine(universe, horizon, deep_scan_n, min_avg_amount)
    return result

st.title("📈 全市場 Candidate Discovery Engine")
st.caption("Web App v1.1｜TWSE＋TPEx｜機會分數 × 風險分級 × 候選型態｜研究排序工具")

with st.sidebar:
    st.header("候選篩選條件")
    horizon = st.radio("A. 投資期間", list(HORIZON_CFG.keys()), index=1)
    st.caption(HORIZON_CFG[horizon]["desc"])

    st.markdown("**B. 股價範圍（元）**")
    c1, c2 = st.columns(2)
    price_min = c1.number_input("最低", min_value=0.0, max_value=5000.0, value=20.0, step=5.0)
    price_max = c2.number_input("最高", min_value=0.0, max_value=5000.0, value=500.0, step=10.0)

    markets = st.multiselect("市場", ["上市", "上櫃"], default=["上市", "上櫃"])
    min_amount_m = st.selectbox("最低 20 日平均成交金額", [10, 30, 50, 100, 300], index=2, format_func=lambda x: f"{x:,} 百萬元")
    deep_scan_n = st.select_slider("深度技術掃描檔數", options=[100,150,200,250,300,400,500], value=250)
    top_n = st.slider("顯示前 N 名", 10, 100, 30, 5)
    min_score = st.slider("最低候選分數", 0, 90, 60, 5)

    run = st.button("🔎 執行全市場掃描", type="primary", use_container_width=True)
    if st.button("♻️ 重新抓取市場資料", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("last_result", None)
        st.rerun()

try:
    snapshot = cached_snapshot()
    val = cached_valuation()
except Exception as exc:
    st.error(f"市場資料讀取失敗：{exc}")
    st.stop()

if not val.empty:
    snapshot = snapshot.merge(val, on=["market","code"], how="left")
else:
    snapshot["pe"] = np.nan
    snapshot["pb"] = np.nan
    snapshot["yield_pct"] = np.nan

date_col = "date_display" if "date_display" in snapshot.columns else "date"
market_dates = "、".join([x for x in sorted(set(snapshot[date_col].astype(str))) if x][:3])

m1,m2,m3,m4 = st.columns(4)
m1.metric("原始股票池", f"{len(snapshot):,} 檔")
m2.metric("上市", f"{(snapshot['market']=='上市').sum():,}")
m3.metric("上櫃", f"{(snapshot['market']=='上櫃').sum():,}")
m4.metric("資料日期", market_dates or "最新交易日")

if price_max < price_min:
    st.error("最高股價必須大於或等於最低股價。")
    st.stop()
if not markets:
    st.warning("請至少選擇一個市場。")
    st.stop()

universe = snapshot[snapshot["market"].isin(markets) & snapshot["close"].between(price_min, price_max)].copy()
universe = universe[universe["amount"].fillna(0) > 0].copy()

st.info(f"目前模式：**{horizon}** ｜ 股價 **{price_min:g}～{price_max:g} 元** ｜ 硬篩選後 **{len(universe):,} 檔** ｜ 市場：{'、'.join(markets)}")

tab1,tab2,tab3,tab4,tab5 = st.tabs(["🏆 候選排行","⚖️ 機會×風險","📊 分數結構","🧠 模型參數","ℹ️ 系統說明"])

result = None
if run and not universe.empty:
    with st.spinner("正在抓取歷史價量並計算 v1.1 候選分數…"):
        result = cached_run(universe.to_json(orient="split"), horizon, deep_scan_n, min_amount_m*1_000_000)
    st.session_state["last_result"] = result
elif "last_result" in st.session_state:
    result = st.session_state["last_result"]

with tab1:
    if universe.empty:
        st.warning("目前條件沒有股票。")
    elif result is None:
        st.subheader("操作方式")
        st.write("設定投資期間與股價範圍後，按「執行全市場掃描」。")
    elif result.empty:
        st.warning("沒有股票通過歷史資料與流動性門檻。")
    else:
        filtered = result[result["候選分數"] >= min_score].head(top_n).copy()
        a,b,c,d,e = st.columns(5)
        a.metric("完成深度評分", f"{len(result)} 檔")
        b.metric("A1 優先研究", f"{(result['研究優先度']=='A1｜優先研究').sum()} 檔")
        c.metric("A2 積極觀察", f"{(result['研究優先度']=='A2｜積極觀察').sum()} 檔")
        d.metric("A-R 高風險", f"{(result['研究優先度']=='A-R｜高風險候選').sum()} 檔")
        e.metric("B級候選", f"{result['研究優先度'].str.startswith('B').sum()} 檔")

        cols = ["排名","市場","代碼","名稱","現價","候選分數","研究優先度","風險等級","候選型態","趨勢","動能","價量","流動性","風險","估值","ROC20%","RSI14","RVOL5/20","ATR14%","60D最大回撤%","估值完整度%","20D日均成交金額(百萬)"]
        st.dataframe(filtered[cols], hide_index=True, use_container_width=True)

        st.download_button("⬇️ 下載完整候選 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"candidate_discovery_v1_1_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")

with tab2:
    st.subheader("Opportunity × Risk 雙軸判讀")
    st.write("Candidate Score 代表機會強度；Risk Level 獨立呈現風險。高分但高風險者不再與穩健型 A 級混為一談。")
    if result is None or result.empty:
        st.info("先執行一次全市場掃描。")
    else:
        risk_order = ["Low","Medium","High","Extreme"]
        rc = result["風險等級"].value_counts().reindex(risk_order, fill_value=0)
        tc = result["候選型態"].value_counts()
        c1,c2 = st.columns(2)
        with c1:
            st.bar_chart(rc)
        with c2:
            st.bar_chart(tc)
        st.dataframe(result.head(top_n)[["排名","代碼","名稱","候選分數","研究優先度","風險等級","候選型態","ATR14%","60D最大回撤%"]],
                     hide_index=True, use_container_width=True)

with tab3:
    st.subheader("Candidate Score 結構")
    weights = HORIZON_CFG[horizon]["weights"]
    name_map = {"trend":"趨勢","momentum":"動能","volume":"價量","liquidity":"流動性","risk":"風險","valuation":"估值"}
    wdf = pd.DataFrame([{"因子":name_map[k],"權重%":v} for k,v in weights.items()])
    st.bar_chart(wdf.set_index("因子"), horizontal=True)
    st.dataframe(wdf, hide_index=True, use_container_width=True)
    st.markdown("A1：80分以上且 Low Risk；A2：80分以上且 Medium Risk；A-R：80分以上且 High/Extreme Risk。")

with tab4:
    st.subheader("v1.1 核心模型與參數")
    st.markdown("""
- **Trend v1.1**：連續評分，使用 Price/MA20、MA5/MA20、MA20/MA60、MA60/MA120、斜率與 ROC。
- **Momentum**：ROC5 / ROC20 / ROC60、RSI14。
- **PVCS-lite 價量**：RVOL5/20 + OBV20 Balance + 20D 漲跌量比。
- **Risk**：ATR14%、60D 最大回撤，另產生 Low / Medium / High / Extreme。
- **Valuation**：PE、PB、殖利率 + Data Completeness。
- **Candidate Type**：穩健趨勢、動能突破、價量轉強、防守型、高風險動能、綜合型。
""")

with tab5:
    st.subheader("資料來源與版本")
    st.markdown("""
- 股票池 / 最新盤後行情：TWSE、TPEx 官方 OpenAPI。
- 歷史 OHLCV：Yahoo Finance。
- 日期顯示：v1.1 已將民國日期轉成西元 YYYY-MM-DD。
- 用途：研究候選排序，不是買進訊號。
- 下一階段：法人籌碼、月營收、EPS Growth、ROE、現金流、產業相對估值。
""")

st.divider()
st.caption("Candidate Discovery Engine Web App v1.1｜Opportunity × Risk｜建議持續以 Walk-Forward / OOS Evidence 校準。")
