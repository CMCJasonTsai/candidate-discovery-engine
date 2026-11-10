# Candidate Discovery Engine Web App v1.2

## 主要更新
- 新增 TWSE T86 / TPEx 三大法人最新交易日籌碼快照。
- 籌碼以外資、投信、三大法人淨買超占當日成交量比例正規化。
- 新增上市／上櫃最新月營收 MoM、YoY、累計 YoY。
- 短／中／長期分別採不同 Chip / Revenue 權重。
- 保留 v1.1 的 Trend、PVCS-lite、Risk Level、Candidate Type 與 A1/A2/A-R。
- 來源取得失敗時 fail-open，不讓整個 Candidate Engine 中斷。

## 權重方向
- 短期：提高價量與籌碼。
- 中期：趨勢、籌碼、營收動能平衡。
- 長期：提高月營收與估值。

## 限制
- 法人目前為最新交易日快照，尚未建立 5D / 20D 累積。
- 月營收 OpenAPI 只提供最新一期，尚未建立 3M / 6M 歷史趨勢。
- v1.2 仍屬候選研究排序，不代表買進訊號。

## 下一步
建立每日／每月歷史保存層，再加入法人 5D/20D、Revenue 3M Trend、EPS Growth、ROE、營業現金流與產業相對估值。
