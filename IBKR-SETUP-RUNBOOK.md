# מדריך קצר: הפעלה מקומית עם IBKR (Gateway Paper) + אפליקציה

מסמך זה מסביר בצורה מרוכזת איך להרים את הגשר (.NET), להתחבר ל‑IB Gateway Paper, לבדוק חיבור, ולהריץ את ה‑UI. בסוף יש גם אופציה משלימה ל‑Client Portal (HYBRID) והמלצות לפתרון תקלות.

## מה נדרש
- Windows, PowerShell
- .NET 8 SDK
- Python 3.10+ (מומלץ) + תלותים מ‑`requirements.txt`
- IB Gateway (Paper) מחובר לחשבון Paper, פורט 4002
- Git כדי לשכפל את הריפוזיטורי

## שלבים במחשב חדש (נקי)
1) שכפול קוד וענף העבודה
```powershell
# משוך את הענף עם השינויים
git clone https://github.com/skwick-gif/AI-BOT-TRADE.git "D:\AI BOT TRADE"
cd "D:\AI BOT TRADE"
git checkout backup-before-cleanup
```

2) התקנת תלותים ל‑Python
```powershell
# אם תרצה, צור venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

3) בניית הגשר (.NET)
```powershell
dotnet build "D:\AI BOT TRADE\InterReactSolution\InterReactSolution.sln" -c Release
```

4) הפעלת הגשר על פורט 5080
```powershell
$env:ASPNETCORE_URLS = 'http://localhost:5080'
dotnet run --project "D:\AI BOT TRADE\InterReactSolution\InterReactBridge\InterReactBridge.csproj" -c Release
# בדיקות בריאות (בחלון PowerShell נוסף):
# Invoke-WebRequest -UseBasicParsing http://localhost:5080/health
```

5) הגדרות IB Gateway (Paper 4002)
- ה‑Gateway צריך להיות מחובר לחשבון Paper (פורט 4002).
- Trusted IPs: הוסף 127.0.0.1.
- Read‑Only API: כבה אם מתכננים לבצע פקודות.
- Popup “Allow incoming connection”: בפעם הראשונה צריך לאשר. שים לב לפופ‑אפ שלא מסתתר מאחורי חלון אחר.

6) בדיקת חיבור מהגשר ל‑Gateway
```powershell
# סקריפט עזר שמריץ health → probe → connect → status → diagnostics
powershell -ExecutionPolicy Bypass -File "D:\AI BOT TRADE\scripts\test_bridge_connect.ps1" -ApiHost 127.0.0.1 -Port 4002 -ClientId 101 -BaseUrl http://localhost:5080
```
צפוי לראות connected=true אחרי connect מוצלח. אם מוצג CONNECT_TIMEOUT, לרוב המשמעות: popup/Trusted IP/הרשאות API ב‑Gateway.

7) הרצת ה‑UI
```powershell
# מתוך תיקיית הפרויקט
python "D:\AI BOT TRADE\main.py"
```
- Connection → Connection Wizard…
  - תוכל לבדוק Status/Probe/Retry Connect
- לאחר התחברות: Connection → Sync From Bridge Status
- בדיקת פקודות מתקדמות:
  - Connection → Place Bracket Order… (לדוגמה: AAPL, Qty 1, MKT, TP +2%, SL ‑1%)
  - Connection → Place OCO Orders… (שתי הוראות SELL: LMT ו‑STP). יש CheckBox ל‑Allow Outside RTH.

## תקלות נפוצות (Troubleshooting)
- CONNECT_TIMEOUT (ב‑/diagnostics או /connect/status):
  - פירוש: TCP נגיש אבל ה‑Gateway לא מקבל session מה‑API.
  - פתרון: אשר Popup “Allow incoming connection”, הוסף 127.0.0.1 ל‑Trusted IPs, ודא שה‑Gateway מחובר לחשבון Paper.
- TCP_PREFLIGHT_FAILED (ב‑/probe):
  - פירוש: הפורט לא עונה.
  - פתרון: בדוק שה‑Gateway פעיל על 4002 ושאין Firewall חוסם.
- פורט 5080 תפוס / “Another InterReactBridge instance is running”: 
  - סגור מופעים קיימים של הגשר:
```powershell
Get-Process -Name InterReactBridge -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
```
- אימות מהיר:
```powershell
# בריאות הגשר
Invoke-WebRequest -UseBasicParsing http://localhost:5080/health
# סטטוס חיבור
Invoke-WebRequest -UseBasicParsing http://localhost:5080/connect/status
# דיאגנוסטיקות עם רמז לתיקון
Invoke-WebRequest -UseBasicParsing http://localhost:5080/diagnostics
# בדיקת TCP ל‑Gateway
Invoke-WebRequest -UseBasicParsing "http://localhost:5080/probe?host=127.0.0.1&port=4002&timeoutMs=800"
```

## Outside RTH (מחוץ לשעות מסחר רגילות)
- Bracket: קיימת תמיכה (נעביר OutsideRegularTradingHours=true על כל רגל).
- OCO/Combo: הוסף CheckBox בדיאלוג OCO; ב‑Combo ניתן להעביר outside_rth דרך ה‑Adapter.
- המלצה: מחוץ לשעות – העדף LIMIT, הימנע מ‑MARKET בגלל נזילות דלילה.

## אופציונלי: Client Portal (HYBRID)
אפשר להפעיל מצב משולב בו חשבון/פורטפוליו נשלפים דרך Client Portal, ומסחר/נתונים חיים נשארים דרך TWS/Gateway.

1) הפעל Client Portal Gateway מקומי (ע״י IBKR) על https://localhost:5000.
2) הגדר משתני סביבה לגשר לפני ההרצה:
```powershell
$env:IBKR_PROVIDER = 'HYBRID'   # או 'CP' / 'TWS'
$env:CP_BASE_URL = 'https://localhost:5000'
$env:CP_INSECURE = 'true'       # לפיתוח מקומי עם תעודה עצמית
$env:ASPNETCORE_URLS = 'http://localhost:5080'
dotnet run --project "D:\AI BOT TRADE\InterReactSolution\InterReactBridge\InterReactBridge.csproj" -c Release
```
3) בדוק את מצב ה‑Provider:
```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:5080/provider
```
הערה: CP אינו מחליף מלא ל‑TWS API לכל הפיצ׳רים. בשלב זה משמש בעיקר לשליפת חשבון/פורטפוליו במצב HYBRID.

## REST שימושיים
- GET /health
- GET /diagnostics
- GET /connect/status
- POST /connect?host=127.0.0.1&port=4002&clientId=101
- POST /connect/retry
- GET /probe?host=127.0.0.1&port=4002&timeoutMs=800
- POST /orders/bracket  (JSON)
- POST /orders/oco      (JSON)
- POST /orders/combo    (JSON)
- GET /account
- GET /portfolio
- GET /marketdata?symbol=AAPL&secType=STK&exchange=SMART&durationSeconds=5
- GET /livedata?symbol=AAPL&secType=STK&exchange=SMART (SSE)
- GET /optionschain?underlying=AAPL&exchange=SMART
- GET /scan?scanType=TOP_PERC_GAIN&numberOfRows=10

---
אם משהו לא עובד — פתח `Connection Wizard…` ב‑UI, בדוק Diagnostics ו‑Status, או הרץ את הסקריפט `scripts/test_bridge_connect.ps1` לקבלת תמונת מצב מלאה. אפשר לפנות אליי כדי להריץ connect ובדיקת הזמנה קטנה ב‑Paper לאימות E2E.
