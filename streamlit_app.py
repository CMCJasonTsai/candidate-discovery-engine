from __future__ import annotations

from datetime import datetime
import hashlib
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

from src.candidate_engine import HORIZON_CFG, load_market_snapshot, load_valuation, run_engine

st.set_page_config(
    page_title="Candidate Discovery Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.small-note {color:#6b7280;font-size:0.88rem;}
.hero {padding:.3rem 0 1rem 0;}
</style>
""", unsafe_allow_html=True)


def check_access():
    try:
        expected = st.secrets.get("APP_ACCESS_CODE", "")
    except Exception:
        expected = ""
    if not expected:
        return
    if st.session_state.get("authorized"):
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


st.markdown('<div class="hero">', unsafe_allow_html=True)
st.title("📈 全市場 Candidate Discovery Engine")
st.caption("Web App v1.0｜TWSE＋TPEx｜短／中／長期候選股票發掘｜研究排序工具")
st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("候選篩選條件")
    horizon = st.radio("A. 投資期間", list(HORIZON_CFG.keys()), index=1)
    st.caption(HORIZON_CFG[horizon]["desc"])

    st.markdown("**B. 股價範圍（元）**")
    c1, c2 = st.columns(2)
    price_min = c1.number_input("最低", min_value=0.0, max_value=5000.0, value=20.0, step=5.0)
    price_max = c2.number_input("最高", min_value=0.0, max_value=5000.0, value=500.0, step=10.0)

    markets = st.multiselect("市場", ["上市", "上櫃"], default=["上市", "上櫃"])
    min_amount_m = st.selectbox("最低 20 日平均成交金額", [10, 30, 50, 100, 300], index=2,
                                format_func=lambda x: f"{x:,} 百萬元")
    deep_scan_n = st.select_slider("深度技術掃描檔數", options=[100, 150, 200, 250, 300, 400, 500], value=250)
    top_n = st.slider("顯示前 N 名", 10, 100, 30, 5)
    min_score = st.slider("最低候選分數", 0, 90, 60, 5)
    run = st.button("🔎 執行全市場掃描", type="primary", use_container_width=True)
    if st.button("♻️ 重新抓取市場資料", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    snapshot = cached_snapshot()
    val = cached_valuation()
except Exception as exc:
    st.error(f"市場資料讀取失敗：{exc}")
    st.info("可能是交易所 API 暫時無法連線。稍後重新整理即可，不需重建 App。")
    st.stop()

if not val.empty:
    snapshot = snapshot.merge(val, on=["market", "code"], how="left")
else:
    snapshot["pe"] = np.nan; snapshot["pb"] = np.nan; snapshot["yield_pct"] = np.nan

market_dates = "、".join([x for x in sorted(set(snapshot["date"].astype(str))) if x][:3])

m1, m2, m3, m4 = st.columns(4)
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

universe = snapshot[
    snapshot["market"].isin(markets) & snapshot["close"].between(price_min, price_max)
].copy()
universe = universe[universe["amount"].fillna(0) > 0].copy()

st.info(
    f"目前模式：**{horizon}** ｜ 股價 **{price_min:g}～{price_max:g} 元** ｜ "
    f"硬篩選後 **{len(universe):,} 檔** ｜ 市場：{'、'.join(markets)}"
)

tab1, tab2, tab3, tab4 = st.tabs(["🏆 候選排行", "📊 分數結構", "🧠 模型參數", "ℹ️ 系統說明"])

with tab1:
    if not run:
        st.subheader("操作方式")
        st.write("在左側設定投資期間與股價範圍後，按下「執行全市場掃描」。系統會先 Hard Gate，再計算 Candidate Score。")
        st.dataframe(
            universe.sort_values("amount", ascending=False).head(20)[["market","code","name","close","amount"]]
                .rename(columns={"market":"市場","code":"代碼","name":"名稱","close":"現價","amount":"當日成交金額"}),
            hide_index=True, use_container_width=True,
        )
    elif universe.empty:
        st.warning("目前條件沒有股票，請放寬股價或市場條件。")
    else:
        with st.spinner("正在抓取歷史價量並計算候選分數…"):
            result = cached_run(universe.to_json(orient="split"), horizon, deep_scan_n, min_amount_m * 1_000_000)
        if result.empty:
            st.warning("沒有股票通過歷史資料與流動性門檻。可降低成交金額門檻或增加掃描檔數。")
        else:
            filtered = result[result["候選分數"] >= min_score].head(top_n).copy()
            a,b,c,d = st.columns(4)
            a.metric("完成深度評分", f"{len(result)} 檔")
            b.metric("A級高關注", f"{result['候選等級'].str.startswith('A').sum()} 檔")
            c.metric("B級候選", f"{result['候選等級'].str.startswith('B').sum()} 檔")
            d.metric("目前顯示", f"{len(filtered)} 檔")

            cols = ["排名","市場","代碼","名稱","現價","候選分數","候選等級","趨勢","動能","價量","流動性","風險","估值",
                    "ROC20%","RSI14","RVOL20","ATR14%","20D日均成交金額(百萬)"]
            st.dataframe(
                filtered[cols], hide_index=True, use_container_width=True,
                column_config={
                    "候選分數": st.column_config.ProgressColumn("候選分數", min_value=0, max_value=100, format="%.1f"),
                    "趨勢": st.column_config.ProgressColumn("趨勢", min_value=0, max_value=100, format="%.0f"),
                    "動能": st.column_config.ProgressColumn("動能", min_value=0, max_value=100, format="%.0f"),
                    "價量": st.column_config.ProgressColumn("價量", min_value=0, max_value=100, format="%.0f"),
                },
            )
            st.download_button(
                "⬇️ 下載完整候選 CSV", result.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"candidate_discovery_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv"
            )
            with st.expander("查看全部技術／估值欄位"):
                st.dataframe(result.head(top_n), hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Candidate Score 結構")
    weights = HORIZON_CFG[horizon]["weights"]
    name_map = {"trend":"趨勢","momentum":"動能","volume":"價量","liquidity":"流動性","risk":"風險","valuation":"估值"}
    wdf = pd.DataFrame([{"因子":name_map[k], "權重%":v} for k,v in weights.items()])
    st.bar_chart(wdf.set_index("因子"), horizontal=True)
    st.dataframe(wdf, hide_index=True, use_container_width=True)
    st.markdown("**候選分級：** A ≥ 80；B = 70–79.9；C = 60–69.9；D < 60。")

with tab3:
    st.subheader("核心模型與參數")
    st.markdown("""
- **Hard Gate**：市場、股價、成交資料、20 日平均成交金額。
- **Trend**：MA5 / MA20 / MA60 / MA120、ROC20、ROC60。
- **Momentum**：ROC5 / ROC20 / ROC60、RSI14。
- **價量**：RVOL20；主要加分區約 1.2～2.5。
- **Liquidity**：20 日平均成交金額。
- **Risk**：ATR14%、60 日最大回撤。
- **Valuation**：PE、PB、殖利率；中長期提高權重。

v1.0 的長期模式仍屬「候選初篩」，後續應加入月營收、EPS、ROE、現金流、法人籌碼與產業相對估值。
""")

with tab4:
    st.subheader("資料來源與版本")
    st.markdown("""
- **股票池 / 最新盤後行情**：TWSE、TPEx 官方 OpenAPI。
- **歷史 OHLCV**：Yahoo Finance（Web App v1.0 PoC）。
- **用途**：全市場 Candidate Discovery，不是自動下單或投資建議。
- **分享方式**：部署到 Streamlit Community Cloud 後，團隊只需開啟網址。
- **保護方式**：可在 Streamlit Secrets 設定 `APP_ACCESS_CODE`；未設定時則不顯示登入畫面。
""")

st.divider()
st.caption("Candidate Discovery Engine Web App v1.0｜建議以 Walk-Forward / OOS Evidence 持續校準門檻與權重。")
