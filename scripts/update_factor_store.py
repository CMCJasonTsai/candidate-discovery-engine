from __future__ import annotations

import html
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "factor_store"
STORE.mkdir(parents=True, exist_ok=True)

INST_FILE = STORE / "institutional_history.csv"
REV_FILE = STORE / "revenue_history.csv"
FEATURE_FILE = STORE / "factor_features.csv"

UA = "Mozilla/5.0 CandidateDiscoveryEngine-HistoricalStore/1.3"
HEADERS = {"User-Agent": UA, "Accept": "application/json,text/plain,*/*"}

TWSE_T86 = "https://www.twse.com.tw/rwd/zh/fund/T86"
TPEX_HIST = "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade"
MOPS_TWSE = "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{roc_year}_{month}_{variant}.html"
MOPS_TPEX = "https://mopsov.twse.com.tw/nas/t21/otc/t21sc03_{roc_year}_{month}_{variant}.html"

INST_COLUMNS = [
    "date","market","code","foreign_net","trust_net","dealer_net","total_net","source"
]
REV_COLUMNS = [
    "period","market","code","revenue_current_thousand","revenue_mom_pct",
    "revenue_yoy_pct","revenue_cum_yoy_pct","source"
]


def num(v):
    if v is None:
        return np.nan
    s = str(v).strip().replace(",", "").replace("+", "").replace("%", "")
    if s in ("", "-", "--", "---", "N/A", "nan", "None"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path, dtype={"code": str, "date": str, "period": str})
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        return df[columns].copy()
    except Exception:
        return pd.DataFrame(columns=columns)


def request_json(url, *, params=None, timeout=30):
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_twse_t86(payload, ymd: str) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    if payload.get("stat") not in ("OK", "ok", None):
        return []

    fields = payload.get("fields")
    data = payload.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        tables = payload.get("tables") or []
        if tables and isinstance(tables[0], dict):
            fields = tables[0].get("fields")
            data = tables[0].get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        return []

    aliases = {
        "code": ["證券代號"],
        "foreign": ["外陸資買賣超股數(不含外資自營商)", "外資買賣超股數"],
        "trust": ["投信買賣超股數"],
        "dealer": ["自營商買賣超股數"],
        "total": ["三大法人買賣超股數"],
    }

    idx = {}
    for k, names in aliases.items():
        found = next((fields.index(n) for n in names if n in fields), None)
        if found is None:
            return []
        idx[k] = found

    rows = []
    for r in data:
        try:
            code = str(r[idx["code"]]).strip()
            if not re.fullmatch(r"\d{4}", code):
                continue
            vals = {k: num(r[idx[k]]) for k in ("foreign","trust","dealer","total")}
            if not all(np.isfinite(vals[k]) for k in ("foreign","trust","dealer")):
                continue
            rows.append({
                "date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}",
                "market": "上市",
                "code": code,
                "foreign_net": int(vals["foreign"]),
                "trust_net": int(vals["trust"]),
                "dealer_net": int(vals["dealer"]),
                "total_net": int(vals["total"]) if np.isfinite(vals["total"]) else int(vals["foreign"]+vals["trust"]+vals["dealer"]),
                "source": "TWSE-T86-hist",
            })
        except Exception:
            continue
    return rows


def parse_tpex_hist(payload, ymd: str) -> list[dict]:
    """Parse both TPEx historical institutional schemas.

    Legacy schema: tables[1], 16 columns.
    Current schema: tables[0], 24 columns.
    """
    if not isinstance(payload, dict) or payload.get("stat") != "ok":
        return []
    tables = payload.get("tables") or []

    # Legacy schema: table[1], exact positions documented by TPEx historical response.
    legacy_fields = [
        "代號", "名稱", "外資及陸資買股數", "外資及陸資賣股數", "外資及陸資淨買股數",
        "投信買進股數", "投信賣股數", "投信淨買股數", "自營淨買股數",
        "自營商(自行買賣)買股數", "自營商(自行買賣)賣股數", "自營商(自行買賣)淨買股數",
        "自營商(避險)買股數", "自營商(避險)賣股數", "自營商(避險)淨買股數",
        "三大法人買賣超股數",
    ]
    if len(tables) > 1 and isinstance(tables[1], dict):
        t = tables[1]
        fields, data = t.get("fields"), t.get("data")
        if fields == legacy_fields and isinstance(data, list):
            rows = []
            for r in data:
                try:
                    if not isinstance(r, list) or len(r) != 16:
                        continue
                    code = str(r[0]).strip()
                    if not re.fullmatch(r"\d{4}", code):
                        continue
                    foreign, trust, dealer, total = num(r[4]), num(r[7]), num(r[8]), num(r[15])
                    if not all(np.isfinite(x) for x in (foreign, trust, dealer)):
                        continue
                    rows.append({
                        "date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}",
                        "market": "上櫃", "code": code,
                        "foreign_net": int(foreign), "trust_net": int(trust),
                        "dealer_net": int(dealer),
                        "total_net": int(total) if np.isfinite(total) else int(foreign + trust + dealer),
                        "source": "TPEx-dailyTrade-hist",
                    })
                except Exception:
                    continue
            if rows:
                return rows

    # Current schema: tables[0], 24 columns.
    if tables and isinstance(tables[0], dict):
        t = tables[0]
        fields, data = t.get("fields"), t.get("data")
        if isinstance(fields, list) and len(fields) == 24 and fields[23] == "三大法人買賣超股數合計" and isinstance(data, list):
            rows = []
            for r in data:
                try:
                    if not isinstance(r, list) or len(r) != 24:
                        continue
                    code = str(r[0]).strip()
                    if not re.fullmatch(r"\d{4}", code):
                        continue
                    foreign, trust, dealer, total = num(r[10]), num(r[13]), num(r[22]), num(r[23])
                    if not all(np.isfinite(x) for x in (foreign, trust, dealer)):
                        continue
                    rows.append({
                        "date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}",
                        "market": "上櫃", "code": code,
                        "foreign_net": int(foreign), "trust_net": int(trust),
                        "dealer_net": int(dealer),
                        "total_net": int(total) if np.isfinite(total) else int(foreign + trust + dealer),
                        "source": "TPEx-dailyTrade-hist",
                    })
                except Exception:
                    continue
            if rows:
                return rows

    return []


def fetch_institutional_day(d: date) -> list[dict]:
    ymd = d.strftime("%Y%m%d")
    rows = []

    try:
        p = request_json(TWSE_T86, params={"response":"json","date":ymd,"selectType":"ALL"})
        rows.extend(parse_twse_t86(p, ymd))
    except Exception as exc:
        print(f"[warn] TWSE {ymd}: {exc}")
    time.sleep(1.7)

    try:
        p = request_json(
            TPEX_HIST,
            params={"type":"Daily","sect":"EW","date":d.strftime("%Y/%m/%d"),"response":"json"},
        )
        rows.extend(parse_tpex_hist(p, ymd))
    except Exception as exc:
        print(f"[warn] TPEx {ymd}: {exc}")
    time.sleep(0.35)
    return rows


def cell_text(raw: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def decode_mops(content: bytes) -> str:
    for enc in ("big5", "cp950", "utf-8"):
        try:
            text = content.decode(enc)
            if "營業收入" in text or "公司代號" in text:
                return text
        except Exception:
            continue
    return content.decode("cp950", errors="ignore")


def parse_mops_month(content: bytes, period: str, market: str) -> list[dict]:
    text = decode_mops(content)
    if "營業收入" not in text:
        return []
    rows = []
    for m in re.finditer(r"<tr\b[^>]*>([\s\S]*?)</tr>", text, flags=re.I):
        cells = [cell_text(x) for x in re.findall(r"<td\b[^>]*>([\s\S]*?)</td>", m.group(1), flags=re.I)]
        if len(cells) != 11 or not re.fullmatch(r"\d{4}", cells[0] or ""):
            continue
        rows.append({
            "period": period,
            "market": market,
            "code": cells[0],
            "revenue_current_thousand": num(cells[2]),
            "revenue_mom_pct": num(cells[5]),
            "revenue_yoy_pct": num(cells[6]),
            "revenue_cum_yoy_pct": num(cells[9]),
            "source": "MOPS-hist",
        })
    return rows


def fetch_mops_month(year: int, month: int, market: str) -> list[dict]:
    roc_year = year - 1911
    template = MOPS_TWSE if market == "上市" else MOPS_TPEX
    out = []
    for variant in (0, 1):
        url = template.format(roc_year=roc_year, month=month, variant=variant)
        try:
            r = requests.get(
                url,
                headers={**HEADERS, "Referer":"https://mopsov.twse.com.tw/", "Accept":"text/html,*/*"},
                timeout=45,
            )
            if r.ok and len(r.content) > 1000:
                out.extend(parse_mops_month(r.content, f"{year:04d}-{month:02d}", market))
        except Exception as exc:
            print(f"[warn] MOPS {market} {year}-{month:02d} v{variant}: {exc}")
        time.sleep(0.35)
    return out


def month_shift(year: int, month: int, delta: int) -> tuple[int,int]:
    n = year * 12 + (month - 1) + delta
    return n // 12, n % 12 + 1


def update_institutional(now: datetime) -> pd.DataFrame:
    old = load_csv(INST_FILE, INST_COLUMNS)
    markets_present = set(old["market"].dropna().astype(str)) if not old.empty else set()
    if old.empty or not {"上市", "上櫃"}.issubset(markets_present):
        # Any missing market triggers a full 45-day repair backfill.
        start = now.date() - timedelta(days=45)
    else:
        latest = pd.to_datetime(old["date"], errors="coerce").max()
        start = (latest.date() - timedelta(days=7)) if pd.notna(latest) else now.date() - timedelta(days=45)

    rows = []
    d = start
    end = now.date()
    while d <= end:
        if d.weekday() < 5:
            rows.extend(fetch_institutional_day(d))
        d += timedelta(days=1)

    new = pd.DataFrame(rows, columns=INST_COLUMNS)
    all_df = pd.concat([old, new], ignore_index=True)
    if all_df.empty:
        return all_df
    all_df["code"] = all_df["code"].astype(str).str.zfill(4)
    all_df = all_df.drop_duplicates(["date","market","code"], keep="last")
    all_df = all_df.sort_values(["date","market","code"]).reset_index(drop=True)
    all_df.to_csv(INST_FILE, index=False, encoding="utf-8")
    return all_df


def update_revenue(now: datetime) -> pd.DataFrame:
    old = load_csv(REV_FILE, REV_COLUMNS)
    if old.empty:
        months = [month_shift(now.year, now.month, -i) for i in range(0, 8)]
    else:
        months = [month_shift(now.year, now.month, -i) for i in range(0, 3)]

    rows = []
    for year, month in reversed(months):
        for market in ("上市","上櫃"):
            rows.extend(fetch_mops_month(year, month, market))

    new = pd.DataFrame(rows, columns=REV_COLUMNS)
    all_df = pd.concat([old, new], ignore_index=True)
    if all_df.empty:
        return all_df
    all_df["code"] = all_df["code"].astype(str).str.zfill(4)
    all_df = all_df.drop_duplicates(["period","market","code"], keep="last")
    all_df = all_df.sort_values(["period","market","code"]).reset_index(drop=True)
    all_df.to_csv(REV_FILE, index=False, encoding="utf-8")
    return all_df


def safe_mean(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan


def build_features(inst: pd.DataFrame, rev: pd.DataFrame) -> pd.DataFrame:
    keys = set()
    if not inst.empty:
        keys |= set(map(tuple, inst[["market","code"]].drop_duplicates().to_records(index=False)))
    if not rev.empty:
        keys |= set(map(tuple, rev[["market","code"]].drop_duplicates().to_records(index=False)))

    inst_map = {k:g.sort_values("date") for k,g in inst.groupby(["market","code"])} if not inst.empty else {}
    rev_map = {k:g.sort_values("period") for k,g in rev.groupby(["market","code"])} if not rev.empty else {}

    out = []
    for market, code in sorted(keys):
        row = {"market":market, "code":str(code).zfill(4)}
        g = inst_map.get((market, code))
        if g is not None and not g.empty:
            g = g.drop_duplicates("date", keep="last").sort_values("date")
            t5, t20 = g.tail(5), g.tail(20)
            row.update({
                "inst_last_date": str(g.iloc[-1]["date"]),
                "inst_obs_5d": len(t5),
                "inst_obs_20d": len(t20),
                "foreign_5d": pd.to_numeric(t5["foreign_net"], errors="coerce").sum(),
                "trust_5d": pd.to_numeric(t5["trust_net"], errors="coerce").sum(),
                "total_5d": pd.to_numeric(t5["total_net"], errors="coerce").sum(),
                "foreign_20d": pd.to_numeric(t20["foreign_net"], errors="coerce").sum(),
                "trust_20d": pd.to_numeric(t20["trust_net"], errors="coerce").sum(),
                "total_20d": pd.to_numeric(t20["total_net"], errors="coerce").sum(),
                "inst_buy_days_5d": int((pd.to_numeric(t5["total_net"], errors="coerce") > 0).sum()),
                "inst_buy_days_20d": int((pd.to_numeric(t20["total_net"], errors="coerce") > 0).sum()),
                "foreign_buy_days_5d": int((pd.to_numeric(t5["foreign_net"], errors="coerce") > 0).sum()),
                "trust_buy_days_5d": int((pd.to_numeric(t5["trust_net"], errors="coerce") > 0).sum()),
            })

        r = rev_map.get((market, code))
        if r is not None and not r.empty:
            r = r.drop_duplicates("period", keep="last").sort_values("period")
            t3, t6 = r.tail(3), r.tail(6)
            prev3 = r.tail(6).head(max(0, len(r.tail(6)) - len(t3)))
            avg3 = safe_mean(t3["revenue_yoy_pct"])
            avg6 = safe_mean(t6["revenue_yoy_pct"])
            prevavg = safe_mean(prev3["revenue_yoy_pct"]) if len(prev3) else np.nan
            row.update({
                "revenue_last_period": str(r.iloc[-1]["period"]),
                "revenue_obs_3m": len(t3),
                "revenue_obs_6m": len(t6),
                "revenue_yoy_3m_avg": avg3,
                "revenue_yoy_6m_avg": avg6,
                "revenue_yoy_prev3_avg": prevavg,
                "revenue_accel_3m": (avg3 - prevavg) if np.isfinite(avg3) and np.isfinite(prevavg) else np.nan,
                "revenue_mom_3m_avg": safe_mean(t3["revenue_mom_pct"]),
                "revenue_positive_months_6": int((pd.to_numeric(t6["revenue_yoy_pct"], errors="coerce") > 0).sum()),
            })
        out.append(row)

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["market","code"]).reset_index(drop=True)
    df.to_csv(FEATURE_FILE, index=False, encoding="utf-8")
    return df


def main():
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    print(f"[factor-store] start {now.isoformat()}")
    inst = update_institutional(now)
    rev = update_revenue(now)
    feat = build_features(inst, rev)
    print(f"[factor-store] institutional_rows={len(inst)} revenue_rows={len(rev)} features={len(feat)}")
    if not inst.empty:
        print(f"[factor-store] institutional range {inst['date'].min()} .. {inst['date'].max()}")
    if not rev.empty:
        print(f"[factor-store] revenue range {rev['period'].min()} .. {rev['period'].max()}")


if __name__ == "__main__":
    main()
