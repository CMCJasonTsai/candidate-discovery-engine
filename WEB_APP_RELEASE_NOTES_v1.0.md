# Candidate Discovery Engine Web App v1.0 — Release Notes

- 改為雲端可部署入口 `streamlit_app.py`
- 保留短／中／長期與股價範圍核心篩選
- 增加上市／上櫃、20D 平均成交金額、深度掃描與最低分數
- 增加 Candidate Score 分項視覺化
- 增加 CSV 匯出
- 增加資料快取與重新抓取按鈕
- 增加可選團隊存取碼（Streamlit Secrets）
- 增加 GitHub / Streamlit Cloud 部署結構與說明
- 將核心計算拆至 `src/candidate_engine.py`，便於後續整合 PVCS、籌碼與預測模型
