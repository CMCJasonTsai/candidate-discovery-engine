# Candidate Discovery Engine Web App v1.2

台股上市＋上櫃全市場候選股票發掘 Web App。

## 功能
- 短期（5～20D）／中期（20～60D）／長期（60～250D）
- 使用者自訂股價區間
- 上市／上櫃市場篩選
- 流動性 Hard Gate
- Trend / Momentum / PVCS-lite Volume / Institutional Chip / Monthly Revenue / Liquidity / Risk / Valuation 分數
- Candidate Score + Risk Level + Candidate Type + A1/A2/A-R 分級
- 最新交易日外資／投信／三大法人買賣超
- 最新月營收 MoM / YoY / 累計 YoY
- 候選結果 CSV 下載
- 可選團隊存取碼

## 本機啟動
Windows 可直接雙擊 `run_local.bat`，或在命令列執行：

```bat
py -m pip install -r requirements.txt
py -m streamlit run streamlit_app.py
```

## 部署到 Streamlit Community Cloud
1. 建立 GitHub repository。
2. 將本資料夾所有檔案上傳到 repository 根目錄。
3. 到 https://share.streamlit.io 建立 App。
4. 選擇 repository / branch，Entry point 設為 `streamlit_app.py`。
5. Deploy 後取得固定 `*.streamlit.app` 網址，分享給團隊。

## 選擇性存取碼
若要加團隊共用存取碼，在 Streamlit Cloud 的 App Settings > Secrets 加入：

```toml
APP_ACCESS_CODE = "your-team-code"
```

若不設定 `APP_ACCESS_CODE`，App 將直接開放瀏覽。

## 資料來源
- TWSE / TPEx 官方 OpenAPI：股票池與最新盤後資料
- Yahoo Finance：PoC 歷史 OHLCV

## 後續建議
v1.3：法人 5D/20D 歷史累積 + EPS/ROE/現金流 + 產業相對估值
v1.2：LightGBM Ranking + Market Regime
v2.0：Chronos / TimesFM + Q10/Q50/Q90 + OOS Evidence
