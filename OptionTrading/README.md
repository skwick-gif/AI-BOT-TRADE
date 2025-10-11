# Options Trading — PyQt6 (v4)
- Covered Call added end-to-end.
- About > Learning Center: Markdown viewer for strategy docs (`docs/*.md`).
- Edit > Prompt Templates: JSON editor to manage Perplexity prompts & output schema.
- Perplexity endpoint is set to FINANCE in `config.json`.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env   # set PERPLEXITY_API_KEY
python main.py
```

## Config
- `config.json` → `perplexity.base_url` points to `.../finance/chat/completions`.
- `paths.docs_dir` & `paths.prompt_templates` control locations for learning docs & prompt templates.


## v5 — Pricing & Greeks
- חדש: `services/pricing.py` מחשב מחיר/יוונים לכל leg עם **py_vollib** (ונופל ל-fallback אם הספרייה לא מותקנת).
- ה־Strategy Details מציגים כעת פרמיות חישוביות/יוונים מצטברים על בסיס מחיר נוכחי (spot).

### התקנה (מומלץ)
```bash
pip install py_vollib numpy
```


## v6 — Expiry/Strike Selector (Step 1)
- חדש: `ui/option_chain_selector.py` — דיאלוג בחירת Expiry & Strikes מתוך IBKR SecDef.
- חיווט: בעת בחירת אסטרטגיה, נפתח Selector שמאפשר לבחור פקיעה וסטרייקים (או לקבל הצעות ע"י % OTM).
- נתמך עבור: Vertical (Bull Call), Iron Condor, Covered Call, ולברירת מחדל Strangle.


## v7 — Step 2: Strategy DSL + Advanced Types
- strategies.json — JSON-DSL לתיאור רגלי אסטרטגיה (vertical/condor/covered_call/calendar/diagonal/butterfly).
- services/strategies.py — בנאי כללי שממיר פרמטרים (K1,K2,E1,E_near,...) ל־legs ל-UI ולביצוע.
- option_chain_selector — תומך כעת ב-**שתי פקיעות** (near/far) לאסטרטגיות Calendar/Diagonal.
- trading_window — משתמש ב-DSL כדי לבנות legs בהתאם לבחירת המשתמש.


## v8 — Step 3: Hybrid Market Scanner
- services/market_scanner.py — סורק היברידי שמושך snapshot מ-IBKR ומבצע דירוג פוטנציאל לוקאלי (ניתן לשדרוג לכללים מבוססי IV/ATR).
- ui/scanner_panel — הוסף בחירת מקור (Hybrid/IBKR-only/Local-only) ושדה Max + כפתור Scan.
- trading_window — שיטת `on_scan_bank_request` שמריצה סריקה על בנק ברירת מחדל (או banks.json אם קיים) ומציגה תוצאות.


## v9 — Step 3 (On‑prem Vectorized Scanner)
- services/local_vector_scanner.py — סורק לוקאלי שמחשב:
  - **IV Snapshot** (median על דגימה קטנה של אופציות סביב ATM), 
  - **IV Rank** (דורש היסטוריית IV חיצונית שתספק בקובץ בעתיד), 
  - **Liquidity/Spread** (מרווחי bid/ask מדגמיים), 
  - **POP מקורב** (עפ"י דלתא ממוצעת של אופציות בסביבה).
- trading_window — כעת בחירת מקור "Local-only" בסורק תריץ את הסורק החדש.
- הערה: כדי לשפר דיוק, הוסף מאקרו שמעדכן מטמון היסטוריית IV לפי יום (CSV/JSON) והזנה ל־LocalVectorScanner.


## v10 — Production Hardening (5 items)
1) **Real combo execution** לכל האסטרטגיות ב-DSL: בנאי רגליים → ComboLegs → הזמנה (Limit/Market) עם TIF.2) **Perplexity validation** קשיחה עם `pydantic` — אין parsing "מוק".3) **No mocks**: שליחת הזמנה מה-UI קוראת ל-`ibkr_client.place_combo_order` (PyQt).4) **Pacing/Throttling** עדין לכל קריאה ל-IBKR.5) **Contract/Exchange**: SMART/USD, qualify מלא, conId, ודיווח שגיאות ברור.


## v11 — Stronger Validation & Scanner Upgrade
- **Perplexity**: ולידציה מחמירה (pydantic) עם בקרה על טווחים, מפתחות לא-ריקים, ולוג מינימלי לקובץ `logs/perplexity_analyze.log`.
- **Scanner (Local-only)**: שודרג עם **היסטוריית IV** (`data/iv_history.json`), **משקלים/ספים בקונפיג** (`config.json`), ומדד פוטנציאל שקוף.
- **TradingWindow**: לאחר סריקה מקומית, ההיסטוריה מתעדכנת אוטומטית לפר symbol.
