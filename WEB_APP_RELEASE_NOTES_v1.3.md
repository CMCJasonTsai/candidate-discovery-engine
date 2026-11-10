# Candidate Discovery Engine Web App v1.3

## 核心升級
- 建立 Historical Factor Store，保存於 `data/factor_store/`。
- 新增 GitHub Actions 自動更新流程：
  - 週一至週五台灣時間約 18:30 執行。
  - 支援 workflow_dispatch 手動執行。
- 首次回補：
  - 法人：最近約 45 個日曆日，建立近 20 個交易日資料。
  - 月營收：最近 8 個月。
- 新增法人歷史因子：
  - Foreign / Trust / Total 5D
  - Foreign / Trust / Total 20D
  - 法人買超日數 5D / 20D
- 新增營收歷史因子：
  - Revenue YoY 3M Average
  - Revenue YoY 6M Average
  - Revenue 3M Acceleration
  - 6M 正成長月數
- Hybrid Score：
  - Store 完整時以歷史分數為主，最新單日／單月為輔。
  - Store 尚未建立時自動 fallback 到 v1.2，不中斷服務。

## 資料檔
- `data/factor_store/institutional_history.csv`
- `data/factor_store/revenue_history.csv`
- `data/factor_store/factor_features.csv`

## 重要限制
- 歷史法人資料以官方 TWSE T86 / TPEx dailyTrade 回補。
- 月營收歷史來自 MOPS 歷史月檔。
- v1.3 仍屬研究候選排序，不等於買進訊號。
