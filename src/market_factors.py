from __future__ import annotations

from typing import Dict, Optional
import math

import numpy as np
import pandas as pd
import requests

TWSE_T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_INST = "https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading"
TWSE_REVENUE = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
TPEX_REVENUE = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"

HEADERS = {
    "User-Agent": "Mozilla/5.0 CandidateDiscoveryEngine/1.2",
    "Accept": "application/json,text/plain,*/*",
}

T86_CODE = "證券代號"
T86_FOREIGN = "外陸資買賣超股數(不含外資自營商)"
T86_TRUST = "投信買賣超股數"
T86_DEALER = "自營商買賣超股數"
T86_TOTAL = "三大法人買賣超股數"

TPEX_FOREIGN = (
    "Foreign Investors include Mainland Area Investors "
    "(Foreign Dealers excluded)-Difference"
)
TPEX_TRUST = "SecuritiesInvestmentTrustCompanies-Difference"
TPEX_DEALER = "Dealers-Difference"
TPEX_TOTAL = "TotalDifference"


def _to_float(v):
    if v is None:
        return np.nan
    s = str(v).strip().replace(",", "").replace("+", "")
    if s in ("", "-", "--", "---", "N/A", "nan", "None"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _to_int(v):
    x = _to_float(v)
    return None if not np.isfinite(x) else int(x)


def roc_to_ad_yyyymmdd(value: str) -> Optional[str]:
    s = str(value or "").strip().replace("/", "").replace("-", "")
    if len(s) == 7 and s.isdigit():
        return f"{int(s[:3]) + 1911:04d}{s[3:]}"
    if len(s) == 8 and s.isdigit():
        return s
    return None


def _get_json(url, *, params=None, timeout=30):
    r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def load_institutional_snapshot(latest_market_date: str) -> pd.DataFrame:
    """Latest per-stock institutional flow for TWSE + TPEx.

    Units are shares. Fail-open: any source outage yields partial data instead
    of breaking the candidate engine.
    """
    rows = []
    ad_date = roc_to_ad_yyyymmdd(latest_market_date)

    # TWSE T86
    try:
        if ad_date:
            payload = _get_json(
                TWSE_T86,
                params={"response": "json", "date": ad_date, "selectType": "ALLBUT0999"},
            )
            fields = payload.get("fields", []) if isinstance(payload, dict) else []
            data = payload.get("data", []) if isinstance(payload, dict) else []
            required = [T86_CODE, T86_FOREIGN, T86_TRUST, T86_DEALER, T86_TOTAL]
            if all(x in fields for x in required):
                idx = {x: fields.index(x) for x in required}
                for r in data:
                    try:
                        code = str(r[idx[T86_CODE]]).strip()
                        foreign = _to_int(r[idx[T86_FOREIGN]])
                        trust = _to_int(r[idx[T86_TRUST]])
                        dealer = _to_int(r[idx[T86_DEALER]])
                        total = _to_int(r[idx[T86_TOTAL]])
                        if code and foreign is not None and trust is not None and dealer is not None:
                            rows.append({
                                "market": "上市", "code": code, "inst_date": ad_date,
                                "foreign_net": foreign, "trust_net": trust, "dealer_net": dealer,
                                "inst_total_net": total if total is not None else foreign + trust + dealer,
                                "inst_source": "TWSE-T86",
                            })
                    except Exception:
                        continue
    except Exception:
        pass

    # TPEx latest snapshot
    try:
        payload = _get_json(TPEX_INST)
        if isinstance(payload, list):
            for r in payload:
                try:
                    code = str(r.get("SecuritiesCompanyCode", "")).strip()
                    foreign = _to_int(r.get(TPEX_FOREIGN))
                    trust = _to_int(r.get(TPEX_TRUST))
                    dealer = _to_int(r.get(TPEX_DEALER))
                    total = _to_int(r.get(TPEX_TOTAL))
                    if not code or foreign is None or trust is None or dealer is None:
                        continue
                    rows.append({
                        "market": "上櫃", "code": code,
                        "inst_date": roc_to_ad_yyyymmdd(r.get("Date", "")) or str(r.get("Date", "")),
                        "foreign_net": foreign, "trust_net": trust, "dealer_net": dealer,
                        "inst_total_net": total if total is not None else foreign + trust + dealer,
                        "inst_source": "TPEx-OpenAPI",
                    })
                except Exception:
                    continue
    except Exception:
        pass

    cols = ["market","code","inst_date","foreign_net","trust_net","dealer_net","inst_total_net","inst_source"]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.drop_duplicates(["market","code"], keep="last")
    return out


def _revenue_row(raw: dict, market: str):
    code = str(raw.get("公司代號", "")).strip()
    if not code:
        return None

    current = _to_float(raw.get("營業收入-當月營收"))
    last_month = _to_float(raw.get("營業收入-上月營收"))
    last_year = _to_float(raw.get("營業收入-去年當月營收"))

    mom = _to_float(raw.get("營業收入-上月比較增減(%)"))
    if not np.isfinite(mom):
        mom = _to_float(raw.get("營業收入-上月增減(%)"))
    if not np.isfinite(mom) and np.isfinite(current) and np.isfinite(last_month) and last_month != 0:
        mom = (current / last_month - 1) * 100

    yoy = _to_float(raw.get("營業收入-去年同月增減(%)"))
    if not np.isfinite(yoy) and np.isfinite(current) and np.isfinite(last_year) and last_year != 0:
        yoy = (current / last_year - 1) * 100

    cum_current = _to_float(raw.get("累計營業收入-當月累計營收"))
    cum_last = _to_float(raw.get("累計營業收入-去年累計營收"))
    cum_yoy = _to_float(raw.get("累計營業收入-前期比較增減(%)"))
    if not np.isfinite(cum_yoy):
        cum_yoy = _to_float(raw.get("累計營業收入-去年同期增減(%)"))
    if not np.isfinite(cum_yoy) and np.isfinite(cum_current) and np.isfinite(cum_last) and cum_last != 0:
        cum_yoy = (cum_current / cum_last - 1) * 100

    return {
        "market": market,
        "code": code,
        "revenue_period": str(raw.get("資料年月", "")).strip(),
        "revenue_current_thousand": current,
        "revenue_mom_pct": mom,
        "revenue_yoy_pct": yoy,
        "revenue_cum_yoy_pct": cum_yoy,
        "revenue_source": "TWSE-MOPS" if market == "上市" else "TPEx-MOPS",
    }


def load_revenue_snapshot() -> pd.DataFrame:
    rows = []
    for url, market in ((TWSE_REVENUE, "上市"), (TPEX_REVENUE, "上櫃")):
        try:
            payload = _get_json(url)
            if not isinstance(payload, list):
                continue
            for raw in payload:
                try:
                    row = _revenue_row(raw, market)
                    if row:
                        rows.append(row)
                except Exception:
                    continue
        except Exception:
            pass

    cols = [
        "market","code","revenue_period","revenue_current_thousand",
        "revenue_mom_pct","revenue_yoy_pct","revenue_cum_yoy_pct","revenue_source"
    ]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.drop_duplicates(["market","code"], keep="last")
    return out


def clamp(x, lo=0.0, hi=100.0):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0


def _tanh_score(value, scale):
    if value is None or not np.isfinite(value):
        return 50.0
    return clamp(50 + 50 * np.tanh(float(value) / float(scale)))


def score_chip(foreign_net, trust_net, total_net, daily_volume):
    """Score latest-day institutional flow after normalizing by daily volume."""
    if daily_volume is None or not np.isfinite(daily_volume) or daily_volume <= 0:
        return 50.0, np.nan, np.nan, np.nan, 0.0

    vals = [foreign_net, trust_net, total_net]
    available = sum(1 for x in vals if x is not None and np.isfinite(x))
    if available == 0:
        return 50.0, np.nan, np.nan, np.nan, 0.0

    fr = float(foreign_net) / daily_volume * 100 if foreign_net is not None and np.isfinite(foreign_net) else np.nan
    tr = float(trust_net) / daily_volume * 100 if trust_net is not None and np.isfinite(trust_net) else np.nan
    tor = float(total_net) / daily_volume * 100 if total_net is not None and np.isfinite(total_net) else np.nan

    fs = _tanh_score(fr, 4.0)
    ts = _tanh_score(tr, 1.8)
    tots = _tanh_score(tor, 5.0)
    score = 0.45 * fs + 0.35 * ts + 0.20 * tots
    return clamp(score), fr, tr, tor, available / 3.0


def score_revenue(mom, yoy, cum_yoy):
    vals = [mom, yoy, cum_yoy]
    available = sum(1 for x in vals if x is not None and np.isfinite(x))
    if available == 0:
        return 50.0, 0.0

    ms = _tanh_score(mom, 20.0)
    ys = _tanh_score(yoy, 30.0)
    cs = _tanh_score(cum_yoy, 25.0)
    return clamp(0.20 * ms + 0.50 * ys + 0.30 * cs), available / 3.0
