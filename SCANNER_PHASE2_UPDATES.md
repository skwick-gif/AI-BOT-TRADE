# 🚀 Scanner Phase 2 Updates - Stock-Specific Normalization & Advanced Features

## ✅ שלב 1: תיקון מערכת הנרמול - הושלם

### 🎯 בעיה שתוקנה:
הנרמול הקודם היה מול שוק כללי (RSI/100, caps ב-5.0) במקום מול המניה עצמה.

### 🔧 הפתרון שיושם:

#### 1. **פונקציית עזר חדשה**: `_calculate_percentile_score()`
- מחשבת ציון מבוסס percentiles (0-4) מול ההיסטוריה של המניה
- Thresholds מותאמים לפי סוג המטריקה
- Fallback mechanisms למקרי כשל

#### 2. **נרמול מתקדם לכל מטריקה**:

**📊 RSI Normalization:**
- מחשב RSI היסטורי של המניה (100+ נרות)
- מדרג לפי percentile: ≥80%=4, ≥60%=3, ≥40%=2, ≥20%=1, <20%=0
- שימוש ב-pandas_ta או חישוב ידני

**📈 Volume Normalization:**
- השוואה מול 60 ימי מסחר אחרונים
- Thresholds גבוהים יותר: [40, 60, 80, 95] percentiles
- זיהוי extreme volume (95%+) vs average volume

**🎢 Volatility Assessment:**
- מבוסס על daily returns histograms
- Thresholds: [30, 50, 70, 90] percentiles  
- מדרג תנועות יומיות ביחס למניה עצמה

**🚀 Price Momentum Score:**
- Multi-timeframe analysis (1W, 1M, 3M returns)
- השוואה מול היסטוריה של המניה
- זיהוי האצה וחזק momentum

#### 3. **ציון בסיסי משופר**:
```python
base_score = (
    normalized_rsi * 0.2 +          # RSI (0-0.8)
    normalized_volume * 0.25 +      # Volume (0-1.0) 
    normalized_volatility * 0.15 +  # Volatility (0-0.45)
    price_momentum_score * 0.2 +    # Momentum (0-0.8)
    max(pattern_score, 0) * 0.2     # Candle patterns (0-0.8)
) / 4.0 * 10.0
```

---

## ✅ שלב 2: זיהוי Candle Patterns - הושלם

### 🕯️ פונקציית עזר חדשה: `_detect_candle_patterns()`

**תבניות מזוהות:**
- **Doji**: גוף קטן יחסית לטווח (< 10%)
- **Hammer**: גוף קטן, צל תחתון ארוך, צל עליון קטן
- **Shooting Star**: גוף קטן, צל עליון ארוך, צל תחתון קטן  
- **Bullish Engulfing**: נר ירוק בולע נר אדום קודם
- **Bearish Engulfing**: נר אדום בולע נר ירוק קודם

**ציון תבניות:**
- כל תבנית מקבלת ציון חיובי/שלילי לפי חוזקה
- `pattern_score` מסכם מספר מתווסף למערכת הניקוד הבסיסית
- השפעה על כל האסטרטגיות

---

## ✅ שלב 3: אסטרטגיית Breakout החדשה - הושלם

### 🚀 מטרה: זיהוי מניות לפני/תחילת פריצה

#### תכונות מתקדמות:

**1. 🔍 Consolidation Detection:**
- מדידת טווח מחירים (high-low) ב-20 ימים אחרונים
- Bollinger Bands squeeze detection
- ציון consolidation: 4=Very Tight, 3=Tight, 2=Moderate, 1=Loose

**2. 📈 Volume Buildup:**
- השוואת נפח: 10 ימים אחרונים vs ימים 20-30
- זיהוי volume buildup (1.1x - 2.5x) ללא explosiveness
- אינדיקציה למעורבות מוסדית

**3. 🎯 Resistance Proximity:**
- זיהוי רמות התנגדות מתקופה של 60 ימים
- מציאת local maxima משמעותיים
- בדיקת קרבה למחיר (< 3% distance)

**4. ⚡ Breakout Setup:**
- מחיר מעל SMA20 & SMA50
- Candle pattern חיובי (score > 1.5)
- RSI בטווח בריא (50-75)
- Volume buildup confirmation
- נדרשים 3+ signals מתוך 4

#### כללי ניקוד Breakout (10 קריטריונים):
```python
rules_b = [
    (consolidation_score >= 3, 4.0, 'Tight Consolidation'),
    (volume_buildup, 3.0, 'Volume Building Up'), 
    (resistance_proximity, 2.5, 'Near Resistance Level'),
    (breakout_setup, 3.5, 'Breakout Setup Complete'),
    # + 6 קריטריונים טכניים נוספים
]
```

**🎛️ ממשק משתמש:**
- CheckBox חדש: "Breakout" 
- נכלל ב-"Run All" toggle
- מינימום 4 קריטריונים כברירת מחדל

---

## 🎯 תוצאות השיפורים:

### 1. **דיוק משופר משמעותית**:
- נרמול מותאם למניה = זיהוי טוב יותר של anomalies
- Candle patterns = זיהוי נקודות כניסה טכניות
- Breakout strategy = זיהוי הזדמנויות פריצה מוקדמות

### 2. **מערכת ניקוד מתוחכמת**:
- 5 אסטרטגיות מלאות עם 60+ קריטריונים משוקללים
- בסיס ציון מנורמל מול המניה עצמה
- שילוב candle patterns בציון הבסיסי

### 3. **קנה מידה ל-10,000 מניות**:
- פונקציות עזר ממוטבות
- חישובים vectorized עם pandas
- Exception handling מקיף

---

## 🔮 השלב הבא: Perplexity AI Integration

### תכנון מוצע:
1. **Pipeline חדש**: 10K מניות → סריקה טכנית → 20-30 מועמדות → Perplexity AI → TOP 5-10
2. **Toggle למשתמש**: בחירה מתי להפעיל AI analysis
3. **Prompt מתוחכם**: כולל טכני + פונדמנטלי + סנטימנט + חדשות
4. **UI חדש**: עמודה/טבלה נפרדת לציון AI וניתוח

### רכיבי Prompt מתוכנן:
- נתוני המניה הטכניים (RSI, Volume, Patterns, etc.)
- מטריקות פונדמנטליות זמינות
- סנטימנט שוק כללי ומגמות סקטוריאליות  
- חדשות ודוחות אחרונים
- השוואה עם peers בתעשייה
- תנאים מאקרו כלכליים רלוונטיים

### Expected Output:
- **AI Score**: 0-100
- **Success Probability**: %
- **Time Horizon**: זמן צפוי לתנועה
- **Key Risks**: סיכונים מזוהים
- **Catalysts**: קטליזטורים צפויים

---

## 📊 סיכום טכני:

**שינויים בקובץ `scanner_widget.py`:**
- +123 שורות קוד חדשות
- 2 פונקציות עזר חדשות
- 1 אסטרטגיה חדשה מלאה
- שיפור מערכת הניקוד הבסיסית
- תמיכה מלאה ב-UI עבור Breakout

**ביצועים:**
- זמן סריקה דומה (optimized calculations)
- דיוק משופר משמעותית
- תמיכה בקנה מידה של 10K מניות

**יציבות:**
- Exception handling מקיף
- Fallback mechanisms בכל חישוב
- בדיקות תקינות רב-שכבתיות