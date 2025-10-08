# בדיקת ביצועי לשונית AI Chat

## השיפורים שבוצעו

### 1. מדידת זמני תגובה
- הוספת מדידת זמן ב-`AIWorker.process_message()` 
- מדידת זמן נפרד לפקודות מיוחדות vs קריאות AI
- הוספת מידע על זמן תגובה ל-logs

### 2. שיפורי AI Service
- הוספת timeout של 30 שניות לבקשות API
- מדידת זמן מפורטת לכל קריאת API
- שיפור הודעות שגיאה עם זמני תגובה
- תמיכה במודלים חלופיים במקרה של שגיאה

### 3. שיפורי ממשק משתמש
- הוספת מידע על זמן תגובה למשתמש (אם מעל 3 שניות)
- שיפור מבחן ה-API עם מדידת זמן
- מבחר מודלים ב-dropdown
- סטטוס API בזמן אמת

## איך לבדוק את הביצועים

### בדיקה בסיסית:
1. פתח את לשונית "🤖 AI Agent"
2. לחץ על כפתור "Test API" - תראה את זמן התגובה
3. שלח הודעה פשוטה כמו "Hello" ובדוק את זמן התגובה

### בדיקה מתקדמת:
1. בדוק את ה-logs בטרמינל לזמני תגובה מפורטים
2. השתמש בפקודות מיוחדות כמו "portfolio status" (מהיר יותר)
3. נסה הודעות ארוכות ומורכבות לבדיקת ביצועים

### זמני תגובה צפויים:
- **פקודות מיוחדות**: < 1 שנייה
- **שאלות פשוטות ל-AI**: 2-5 שניות  
- **שאלות מורכבות**: 5-10 שניות
- **זמן חריג**: > 10 שניות (יוצג אזהרה)

## תכונות נוספות

### כפתורי פעולה מהירה:
- "📊 Portfolio Status" - סטטוס תיק השקעות
- "📈 Market Update" - עדכון שוק
- "🎯 Trading Help" - עזרה במסחר
- "⚠️ Risk Analysis" - ניתוח סיכונים

### בחירת מודל AI:
הישום תומך במודלים הבאים:
- `reasoning-pro` (מתקדם, איטי יותר)
- `sonar` (מהיר, טוב לשאלות כלליות)
- `sonar-pro` (איזון בין מהירות לאיכות)
- `sonar-small` (מהיר מאוד, פחות מדויק)

## פתרון בעיות נפוצות

### תגובה איטית:
1. בדוק חיבור לאינטרנט
2. נסה להחליף מודל ל-`sonar` (מהיר יותר)
3. בדוק שהמפתח של Perplexity API תקין

### שגיאות API:
1. ודא שהמפתח ב-.env נכון
2. בדוק שלא חרגת ממכסת ה-API
3. נסה לאתחל את היישום

### לוגים לא מופיעים:
הרץ את היישום מהטרמינל:
```
C:/Users/eranl/Downloads/AI-BOT-TRADE/.venv/Scripts/python.exe main.py
```

## סיכום
השיפורים כוללים מעקב מפורט אחר ביצועים, טיפול משופר בשגיאות, וממשק משתמש מתקדם לבדיקת ביצועי ה-AI Chat.