from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.market_factors import score_chip, score_revenue, score_chip_history, score_revenue_history

TWSE_DAILY = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_VAL = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
TPEX_DAILY = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
TPEX_VAL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis"

HEADERS = {
    "User-Agent": "Mozilla/5.0 CandidateDiscoveryEngine/1.3",
    "Accept": "application/json,text/plain,*/*",
}

HORIZON_CFG = {
    "短期（5～20交易日）": {
        "period": "6mo",
        "weights": {"trend": 20, "momentum": 20, "volume": 18, "chip": 17, "revenue": 5, "liquidity": 8, "risk": 12, "valuation": 0},
        "desc": "偏重趨勢、動能與價量，適合約 1～4 週候選發掘。",
    },
    "中期（20～60交易日）": {
        "period": "1y",
        "weights": {"trend": 24, "momentum": 15, "volume": 12, "chip": 15, "revenue": 12, "liquidity": 7, "risk": 10, "valuation": 5},
        "desc": "提高趨勢持續性與風險控制，適合約 1～3 個月。",
    },
    "長期（60～250交易日）": {
        "period": "2y",
        "weights": {"trend": 20, "momentum": 8, "volume": 7, "chip": 8, "revenue": 22, "liquidity": 5, "risk": 10, "valuation": 20},
        "desc": "v1.2 提高月營收與估值權重，作為長期候選的基本面動能初篩。",
    },
}


def clean_num(v):
    if v is None:
        return np.nan
    s = str(v).strip().replace(",", "").replace("--", "").replace("---", "").replace("+", "")
    if s in ("", "-", "N/A", "nan", "None"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def pick(row, *keys):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def fetch_json(url: str, timeout: int = 30):
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def format_market_date(value: str) -> str:
    """Convert ROC yyyMMdd (e.g. 1151109) or Gregorian yyyymmdd to YYYY-MM-DD."""
    s = str(value or "").strip().replace("/", "").replace("-", "")
    if not s.isdigit():
        return str(value or "")
    try:
        if len(s) == 7:
            year = int(s[:3]) + 1911
            return f"{year:04d}-{int(s[3:5]):02d}-{int(s[5:7]):02d}"
        if len(s) == 8:
            return f"{int(s[:4]):04d}-{int(s[4:6]):02d}-{int(s[6:8]):02d}"
    except Exception:
        pass
    return str(value or "")


def load_market_snapshot() -> pd.DataFrame:
    rows = []
    errors = []

    try:
        for x in fetch_json(TWSE_DAILY):
            code = str(pick(x, "Code", "證券代號") or "").strip()
            if not (code.isdigit() and len(code) == 4 and not code.startswith("0")):
                continue
            rows.append({
                "market": "上市", "code": code,
                "name": str(pick(x, "Name", "證券名稱") or "").strip(),
                "date": str(pick(x, "Date", "日期") or "").strip(),
                "open": clean_num(pick(x, "OpeningPrice", "開盤價")),
                "high": clean_num(pick(x, "HighestPrice", "最高價")),
                "low": clean_num(pick(x, "LowestPrice", "最低價")),
                "close": clean_num(pick(x, "ClosingPrice", "收盤價")),
                "change": clean_num(pick(x, "Change", "漲跌價差")),
                "volume": clean_num(pick(x, "TradeVolume", "成交股數")),
                "amount": clean_num(pick(x, "TradeValue", "成交金額")),
            })
    except Exception as exc:
        errors.append(f"TWSE: {exc}")

    try:
        for x in fetch_json(TPEX_DAILY):
            code = str(pick(x, "SecuritiesCompanyCode", "Code") or "").strip()
            if not (code.isdigit() and len(code) == 4 and not code.startswith("0")):
                continue
            rows.append({
                "market": "上櫃", "code": code,
                "name": str(pick(x, "CompanyName", "SecuritiesCompanyName", "Name") or "").strip(),
                "date": str(pick(x, "Date") or "").strip(),
                "open": clean_num(pick(x, "Open")),
                "high": clean_num(pick(x, "High")),
                "low": clean_num(pick(x, "Low")),
                "close": clean_num(pick(x, "Close")),
                "change": clean_num(pick(x, "Change")),
                "volume": clean_num(pick(x, "TradingShares")),
                "amount": clean_num(pick(x, "TransactionAmount")),
            })
    except Exception as exc:
        errors.append(f"TPEx: {exc}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("市場快照讀取失敗：" + " | ".join(errors))
    df = df.drop_duplicates(["market", "code"]).copy()
    df["date_display"] = df["date"].map(format_market_date)
    return df[df["close"].notna() & (df["close"] > 0)].copy()


def load_valuation() -> pd.DataFrame:
    rows = []
    try:
        for x in fetch_json(TWSE_VAL):
            code = str(pick(x, "Code", "證券代號") or "").strip()
            if not (code.isdigit() and len(code) == 4 and not code.startswith("0")):
                continue
            rows.append({
                "market": "上市", "code": code,
                "pe": clean_num(pick(x, "PEratio", "本益比")),
                "pb": clean_num(pick(x, "PBratio", "股價淨值比")),
                "yield_pct": clean_num(pick(x, "DividendYield", "殖利率(%)")),
            })
    except Exception:
        pass

    try:
        for x in fetch_json(TPEX_VAL):
            code = str(pick(x, "SecuritiesCompanyCode", "Code") or "").strip()
            if not (code.isdigit() and len(code) == 4 and not code.startswith("0")):
                continue
            rows.append({
                "market": "上櫃", "code": code,
                "pe": clean_num(pick(x, "PriceEarningRatio", "PER", "P/E")),
                "pb": clean_num(pick(x, "PriceBookRatio", "PBR", "P/B")),
                "yield_pct": clean_num(pick(x, "YieldRatio", "DividendYield")),
            })
    except Exception:
        pass

    return pd.DataFrame(rows)


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean() / close * 100


def max_drawdown(close: pd.Series, n: int = 60) -> float:
    s = close.tail(n)
    if len(s) < 2:
        return np.nan
    dd = s / s.cummax() - 1
    return float(dd.min() * 100)


def clamp(x, lo=0.0, hi=100.0):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0


def sigmoid_tanh(x, scale):
    if not np.isfinite(x):
        return 50.0
    return clamp(50 + 50 * np.tanh(float(x) / float(scale)))


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    if isinstance(x.columns, pd.MultiIndex) and x.columns.nlevels >= 2:
        level0 = set(map(str, x.columns.get_level_values(0)))
        level1 = set(map(str, x.columns.get_level_values(1)))
        prices = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if prices.intersection(level0):
            x.columns = x.columns.get_level_values(0)
        elif prices.intersection(level1):
            x.columns = x.columns.get_level_values(1)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in x.columns for c in needed):
        return pd.DataFrame()
    return x[needed].copy()


def download_history(meta_records: Iterable[dict], period: str) -> Dict[Tuple[str, str], pd.DataFrame]:
    meta = pd.DataFrame(list(meta_records))
    if meta.empty:
        return {}

    tickers, ticker_map = [], {}
    for _, row in meta.iterrows():
        ticker = f'{row["code"]}.TW' if row["market"] == "上市" else f'{row["code"]}.TWO'
        tickers.append(ticker)
        ticker_map[ticker] = (row["market"], row["code"])

    result = {}
    for start in range(0, len(tickers), 50):
        batch = tickers[start:start+50]
        try:
            raw = yf.download(
                tickers=batch, period=period, interval="1d", auto_adjust=False,
                group_by="ticker", threads=True, progress=False, timeout=25,
            )
        except Exception:
            continue

        if len(batch) == 1:
            result[ticker_map[batch[0]]] = normalize_ohlcv(raw)
            continue

        for ticker in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if ticker in raw.columns.get_level_values(0):
                        frame = raw[ticker]
                    elif ticker in raw.columns.get_level_values(1):
                        frame = raw.xs(ticker, axis=1, level=1)
                    else:
                        continue
                else:
                    continue
                result[ticker_map[ticker]] = normalize_ohlcv(frame)
            except Exception:
                continue
    return result


def calc_features(ohlcv: pd.DataFrame):
    d = normalize_ohlcv(ohlcv).dropna(subset=["Close"]).copy()
    if len(d) < 65:
        return None

    c = pd.to_numeric(d["Close"], errors="coerce")
    h = pd.to_numeric(d["High"], errors="coerce")
    l = pd.to_numeric(d["Low"], errors="coerce")
    v = pd.to_numeric(d["Volume"], errors="coerce").fillna(0)

    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    ma120 = c.rolling(120).mean()
    ma120_last = ma120.iloc[-1] if len(c) >= 120 and np.isfinite(ma120.iloc[-1]) else ma60.iloc[-1]

    mean20 = v.rolling(20).mean().iloc[-1]
    mean5 = v.rolling(5).mean().iloc[-1]
    rvol20 = float(v.iloc[-1] / mean20) if mean20 > 0 else np.nan
    rvol5 = float(mean5 / mean20) if mean20 > 0 else np.nan

    direction = np.sign(c.diff()).fillna(0)
    obv = (direction * v).cumsum()
    obv_balance20 = np.nan
    if len(obv) >= 21 and mean20 > 0:
        obv_balance20 = float((obv.iloc[-1] - obv.iloc[-21]) / (mean20 * 20) * 100)

    up_vol = float(v.where(c.diff() > 0, 0).tail(20).sum())
    down_vol = float(v.where(c.diff() < 0, 0).tail(20).sum())
    up_down_vol_ratio = up_vol / down_vol if down_vol > 0 else (5.0 if up_vol > 0 else np.nan)

    ma20_slope5 = float((ma20.iloc[-1] / ma20.iloc[-6] - 1) * 100) if len(ma20.dropna()) >= 6 else np.nan
    ma60_slope10 = float((ma60.iloc[-1] / ma60.iloc[-11] - 1) * 100) if len(ma60.dropna()) >= 11 else np.nan

    return {
        "close": float(c.iloc[-1]),
        "ma5": float(ma5.iloc[-1]), "ma20": float(ma20.iloc[-1]), "ma60": float(ma60.iloc[-1]),
        "ma120": float(ma120_last),
        "price_ma20_pct": float((c.iloc[-1] / ma20.iloc[-1] - 1) * 100),
        "ma5_ma20_pct": float((ma5.iloc[-1] / ma20.iloc[-1] - 1) * 100),
        "ma20_ma60_pct": float((ma20.iloc[-1] / ma60.iloc[-1] - 1) * 100),
        "ma60_ma120_pct": float((ma60.iloc[-1] / ma120_last - 1) * 100) if ma120_last else np.nan,
        "ma20_slope5_pct": ma20_slope5,
        "ma60_slope10_pct": ma60_slope10,
        "roc5": float((c.iloc[-1] / c.iloc[-6] - 1) * 100) if len(c) >= 6 else np.nan,
        "roc20": float((c.iloc[-1] / c.iloc[-21] - 1) * 100) if len(c) >= 21 else np.nan,
        "roc60": float((c.iloc[-1] / c.iloc[-61] - 1) * 100) if len(c) >= 61 else np.nan,
        "rsi14": float(rsi(c, 14).iloc[-1]),
        "rvol20": rvol20,
        "rvol5": rvol5,
        "obv_balance20": obv_balance20,
        "up_down_vol_ratio20": up_down_vol_ratio,
        "atr14_pct": float(atr_pct(h, l, c, 14).iloc[-1]),
        "avg_amount20": float((c * v).rolling(20).mean().iloc[-1]),
        "maxdd": max_drawdown(c, 60),
        "history_days": int(len(d)),
    }


def score_trend(f, horizon):
    pos = sigmoid_tanh(f["price_ma20_pct"], 8)
    align20_60 = sigmoid_tanh(f["ma20_ma60_pct"], 8)
    slope20 = sigmoid_tanh(f["ma20_slope5_pct"], 2.5)
    slope60 = sigmoid_tanh(f["ma60_slope10_pct"], 3.0)
    roc20 = sigmoid_tanh(f["roc20"], 15)
    roc60 = sigmoid_tanh(f["roc60"], 30)

    if horizon.startswith("短"):
        align5_20 = sigmoid_tanh(f["ma5_ma20_pct"], 4)
        return clamp(0.20*pos + 0.20*align5_20 + 0.20*align20_60 + 0.25*slope20 + 0.15*roc20)
    if horizon.startswith("中"):
        return clamp(0.15*pos + 0.25*align20_60 + 0.25*slope20 + 0.20*slope60 + 0.15*roc60)

    align60_120 = sigmoid_tanh(f["ma60_ma120_pct"], 10)
    return clamp(0.10*pos + 0.20*align20_60 + 0.25*align60_120 + 0.20*slope60 + 0.25*roc60)


def score_momentum(f, horizon):
    r = f["rsi14"]
    if not np.isfinite(r):
        rsi_score = 50
    elif 55 <= r <= 72:
        rsi_score = 100
    elif 48 <= r < 55 or 72 < r <= 78:
        rsi_score = 70
    elif 40 <= r < 48:
        rsi_score = 35
    else:
        rsi_score = 10

    roc5 = f["roc5"] if np.isfinite(f["roc5"]) else 0
    roc20 = f["roc20"] if np.isfinite(f["roc20"]) else 0
    roc60 = f["roc60"] if np.isfinite(f["roc60"]) else 0
    if horizon.startswith("短"):
        roc = 0.65*np.tanh(roc5/6) + 0.35*np.tanh(roc20/12)
    elif horizon.startswith("中"):
        roc = 0.4*np.tanh(roc20/12) + 0.6*np.tanh(roc60/25)
    else:
        roc = 0.25*np.tanh(roc20/12) + 0.75*np.tanh(roc60/30)
    return clamp(0.45*rsi_score + 0.55*(50 + 50*roc))


def score_volume(f):
    """PVCS-lite v1.1: use 5D/20D RVOL + OBV balance + up/down volume ratio."""
    rv = f["rvol5"]
    if not np.isfinite(rv):
        rv_score = 40
    elif 1.10 <= rv <= 2.20:
        rv_score = 100
    elif 0.85 <= rv < 1.10 or 2.20 < rv <= 3.00:
        rv_score = 75
    elif 0.60 <= rv < 0.85 or 3.00 < rv <= 4.00:
        rv_score = 50
    else:
        rv_score = 30

    obv_score = sigmoid_tanh(f["obv_balance20"], 35)

    ratio = f["up_down_vol_ratio20"]
    if np.isfinite(ratio) and ratio > 0:
        ratio_score = clamp(50 + 50*np.tanh(np.log(ratio)*1.4))
    else:
        ratio_score = 50

    return clamp(0.50*rv_score + 0.30*obv_score + 0.20*ratio_score)


def score_liquidity(avg_amount):
    if not np.isfinite(avg_amount) or avg_amount <= 0:
        return 0
    return clamp((np.log10(avg_amount) - 6.5) / (9.0 - 6.5) * 100)


def score_risk(f, horizon):
    atr = f["atr14_pct"] if np.isfinite(f["atr14_pct"]) else 5
    mdd = abs(f["maxdd"]) if np.isfinite(f["maxdd"]) else 20
    if horizon.startswith("短"):
        atr_s = 100 if 1.5 <= atr <= 5.5 else 65 if 0.8 <= atr <= 7.5 else 30
        dd_s = clamp(100 - max(0, mdd - 6) * 3.0)
    elif horizon.startswith("中"):
        atr_s = 100 if 1.2 <= atr <= 4.5 else 65 if 0.7 <= atr <= 6.0 else 30
        dd_s = clamp(100 - max(0, mdd - 10) * 2.5)
    else:
        atr_s = 100 if 0.8 <= atr <= 4.0 else 65 if 0.5 <= atr <= 5.5 else 30
        dd_s = clamp(100 - max(0, mdd - 15) * 1.8)
    return clamp(0.45*atr_s + 0.55*dd_s)


def risk_level(f, horizon):
    atr = f["atr14_pct"] if np.isfinite(f["atr14_pct"]) else 99
    mdd = abs(f["maxdd"]) if np.isfinite(f["maxdd"]) else 99

    if horizon.startswith("短"):
        if atr >= 10 or mdd >= 30: return "Extreme"
        if atr >= 8 or mdd >= 24: return "High"
        if atr >= 6 or mdd >= 18: return "Medium"
        return "Low"

    if horizon.startswith("中"):
        if atr >= 10 or mdd >= 28: return "Extreme"
        if atr >= 8 or mdd >= 20: return "High"
        if atr >= 4.5 or mdd >= 15: return "Medium"
        return "Low"

    if atr >= 8 or mdd >= 30: return "Extreme"
    if atr >= 6 or mdd >= 24: return "High"
    if atr >= 4.5 or mdd >= 18: return "Medium"
    return "Low"


def score_valuation(pe, pb, yld):
    parts = []
    if np.isfinite(pe) and pe > 0:
        parts.append(100 if 8 <= pe <= 25 else 75 if 4 <= pe < 8 or 25 < pe <= 35 else 45 if pe <= 50 else 20)
    if np.isfinite(pb) and pb > 0:
        parts.append(100 if 0.7 <= pb <= 3 else 70 if pb <= 5 else 35)
    if np.isfinite(yld) and yld >= 0:
        parts.append(clamp(yld / 5 * 100))

    completeness = len(parts) / 3.0
    raw = float(np.mean(parts)) if parts else 0.0
    adjusted = raw * completeness
    return adjusted, raw, completeness


def grade(score):
    if score >= 80: return "A｜高關注"
    if score >= 70: return "B｜候選"
    if score >= 60: return "C｜觀察"
    return "D｜暫不列入"


def research_priority(score, risk):
    if score >= 80:
        if risk == "Low": return "A1｜優先研究"
        if risk == "Medium": return "A2｜積極觀察"
        return "A-R｜高風險候選"
    if score >= 70: return "B｜候選"
    if score >= 60: return "C｜觀察"
    return "D｜暫不列入"


def candidate_type(trend, momentum, volume, risk_score, risk):
    if risk in ("High", "Extreme") and momentum >= 75:
        return "高風險動能"
    if trend >= 70 and momentum >= 75 and volume >= 70:
        return "動能突破"
    if trend >= 70 and risk_score >= 75 and volume < 70:
        return "穩健趨勢"
    if volume >= 75 and momentum >= 60:
        return "價量轉強"
    if risk_score >= 85 and momentum < 65:
        return "防守型"
    return "綜合型"


def run_engine(universe: pd.DataFrame, horizon: str, deep_scan_n: int, min_avg_amount: float, history=None):
    cfg = HORIZON_CFG[horizon]
    pre = universe.sort_values("amount", ascending=False).head(deep_scan_n).copy()
    hist = history if history is not None else download_history(
        pre[["market", "code"]].to_dict("records"), cfg["period"]
    )
    rows, weights = [], cfg["weights"]

    for _, row in pre.iterrows():
        key = (row["market"], row["code"])
        if key not in hist:
            continue
        f = calc_features(hist[key])
        if not f or f["avg_amount20"] < min_avg_amount:
            continue

        trend = score_trend(f, horizon)
        momentum = score_momentum(f, horizon)
        volume = score_volume(f)
        liquidity = score_liquidity(f["avg_amount20"])
        risk_score = score_risk(f, horizon)
        risk = risk_level(f, horizon)

        pe = row.get("pe", np.nan)
        pb = row.get("pb", np.nan)
        yld = row.get("yield_pct", np.nan)
        valuation, valuation_raw, completeness = score_valuation(pe, pb, yld)

        chip_current, foreign_ratio, trust_ratio, inst_total_ratio, chip_complete_current = score_chip(
            row.get("foreign_net", np.nan), row.get("trust_net", np.nan),
            row.get("inst_total_net", np.nan), row.get("volume", np.nan)
        )
        revenue_current, revenue_complete_current = score_revenue(
            row.get("revenue_mom_pct", np.nan), row.get("revenue_yoy_pct", np.nan),
            row.get("revenue_cum_yoy_pct", np.nan)
        )

        chip_hist, chip_hist_complete = score_chip_history(row, row.get("volume", np.nan))
        revenue_hist, revenue_hist_complete = score_revenue_history(row)

        if chip_hist_complete >= 0.35:
            chip_score = 0.75 * chip_hist + 0.25 * chip_current
            chip_mode = "5D/20D"
            chip_complete = max(chip_complete_current, chip_hist_complete)
        else:
            chip_score = chip_current
            chip_mode = "單日"
            chip_complete = chip_complete_current

        if revenue_hist_complete >= 0.30:
            revenue_score = 0.80 * revenue_hist + 0.20 * revenue_current
            revenue_mode = "3M/6M"
            revenue_complete = max(revenue_complete_current, revenue_hist_complete)
        else:
            revenue_score = revenue_current
            revenue_mode = "單月"
            revenue_complete = revenue_complete_current

        total = (
            trend*weights["trend"] + momentum*weights["momentum"] + volume*weights["volume"] +
            chip_score*weights["chip"] + revenue_score*weights["revenue"] +
            liquidity*weights["liquidity"] + risk_score*weights["risk"] + valuation*weights["valuation"]
        ) / max(1, sum(weights.values()))

        priority = research_priority(total, risk)
        ctype = candidate_type(trend, momentum, volume, risk_score, risk)

        rows.append({
            "市場": row["market"], "代碼": row["code"], "名稱": row["name"], "現價": row["close"],
            "候選分數": round(total, 1), "候選等級": grade(total),
            "研究優先度": priority, "風險等級": risk, "候選型態": ctype,
            "趨勢": round(trend, 1), "動能": round(momentum, 1), "價量": round(volume, 1),
            "籌碼": round(chip_score, 1), "營收動能": round(revenue_score, 1),
            "籌碼模式": chip_mode, "營收模式": revenue_mode,
            "籌碼歷史分": round(chip_hist, 1), "營收歷史分": round(revenue_hist, 1),
            "流動性": round(liquidity, 1), "風險": round(risk_score, 1), "估值": round(valuation, 1),
            "外資淨買超/成交量%": None if not np.isfinite(foreign_ratio) else round(foreign_ratio, 2),
            "投信淨買超/成交量%": None if not np.isfinite(trust_ratio) else round(trust_ratio, 2),
            "法人合計/成交量%": None if not np.isfinite(inst_total_ratio) else round(inst_total_ratio, 2),
            "籌碼完整度%": round(chip_complete*100, 0),
            "月營收MoM%": None if not np.isfinite(row.get("revenue_mom_pct", np.nan)) else round(float(row.get("revenue_mom_pct")), 2),
            "月營收YoY%": None if not np.isfinite(row.get("revenue_yoy_pct", np.nan)) else round(float(row.get("revenue_yoy_pct")), 2),
            "累計營收YoY%": None if not np.isfinite(row.get("revenue_cum_yoy_pct", np.nan)) else round(float(row.get("revenue_cum_yoy_pct")), 2),
            "營收完整度%": round(revenue_complete*100, 0),
            "法人5D": None if not np.isfinite(row.get("total_5d", np.nan)) else round(float(row.get("total_5d")), 0),
            "法人20D": None if not np.isfinite(row.get("total_20d", np.nan)) else round(float(row.get("total_20d")), 0),
            "法人買超日數5D": None if not np.isfinite(row.get("inst_buy_days_5d", np.nan)) else int(row.get("inst_buy_days_5d")),
            "法人買超日數20D": None if not np.isfinite(row.get("inst_buy_days_20d", np.nan)) else int(row.get("inst_buy_days_20d")),
            "營收YoY 3M平均%": None if not np.isfinite(row.get("revenue_yoy_3m_avg", np.nan)) else round(float(row.get("revenue_yoy_3m_avg")), 2),
            "營收YoY 6M平均%": None if not np.isfinite(row.get("revenue_yoy_6m_avg", np.nan)) else round(float(row.get("revenue_yoy_6m_avg")), 2),
            "營收3M加速度": None if not np.isfinite(row.get("revenue_accel_3m", np.nan)) else round(float(row.get("revenue_accel_3m")), 2),
            "營收正成長月數6M": None if not np.isfinite(row.get("revenue_positive_months_6", np.nan)) else int(row.get("revenue_positive_months_6")),
            "估值原始分": round(valuation_raw, 1), "估值完整度%": round(completeness*100, 0),
            "ROC5%": round(f["roc5"], 2), "ROC20%": round(f["roc20"], 2), "ROC60%": round(f["roc60"], 2),
            "RSI14": round(f["rsi14"], 1), "RVOL5/20": round(f["rvol5"], 2),
            "單日RVOL20": round(f["rvol20"], 2), "OBV20平衡%": round(f["obv_balance20"], 2),
            "漲跌量比20": round(f["up_down_vol_ratio20"], 2), "ATR14%": round(f["atr14_pct"], 2),
            "60D最大回撤%": round(f["maxdd"], 2),
            "MA20斜率5D%": round(f["ma20_slope5_pct"], 2), "MA60斜率10D%": round(f["ma60_slope10_pct"], 2),
            "Price-MA20%": round(f["price_ma20_pct"], 2), "MA20-MA60%": round(f["ma20_ma60_pct"], 2),
            "20D日均成交金額(百萬)": round(f["avg_amount20"]/1e6, 1),
            "PE": None if not np.isfinite(pe) else round(float(pe), 2),
            "PB": None if not np.isfinite(pb) else round(float(pb), 2),
            "殖利率%": None if not np.isfinite(yld) else round(float(yld), 2),
            "歷史樣本日數": f["history_days"],
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["候選分數", "20D日均成交金額(百萬)"], ascending=[False, False]
        ).reset_index(drop=True)
        out.insert(0, "排名", np.arange(1, len(out)+1))
    return out, hist
