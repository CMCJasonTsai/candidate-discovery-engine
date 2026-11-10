# Web App 部署操作指引

## 推薦：GitHub + Streamlit Community Cloud

### A. GitHub
建立一個 repository，例如：
`candidate-discovery-engine`

把下列檔案與資料夾放在 repo 根目錄：
- streamlit_app.py
- src/
- requirements.txt
- .streamlit/config.toml
- README.md

不要上傳 `.streamlit/secrets.toml`。

### B. Streamlit Cloud
1. 登入 Streamlit Community Cloud。
2. Create app。
3. 選擇 GitHub repository。
4. Branch：通常 `main`。
5. Main file path：`streamlit_app.py`。
6. Deploy。

### C. 團隊分享
部署完成後會得到固定網址，例如：
`https://candidate-discovery-engine.streamlit.app`

直接將網址傳給團隊即可。

### D. 若要限制團隊存取
在 App Settings > Secrets 設定：

```toml
APP_ACCESS_CODE = "your-private-team-code"
```

請勿將真實密碼寫入 GitHub。

### E. 更新 App
後續只要把新版程式 commit / push 到同一 GitHub repo，Streamlit Cloud 會重新部署。
