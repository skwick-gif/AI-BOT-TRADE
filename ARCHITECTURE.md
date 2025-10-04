# AI Trading Bot - PyQt6 Architecture & Structure

## 📋 Project Overview

This document outlines the complete architecture and folder structure for the AI Trading Bot PyQt6 application, including the new ML, Watchlist, and Scanner modules.

## 🏗️ Complete Folder Structure

```
📁 src/
├── 📁 main.py                          # Entry point
├── 📁 ui/
│   ├── 📁 windows/
│   │   ├── main_window.py              # MainWindow class
│   │   └── settings_window.py          # Settings dialog
│   ├── 📁 widgets/
│   │   ├── __init__.py
│   │   ├── chat_widget.py              # AI Chat functionality
│   │   ├── ai_trading_widget.py        # AI Trading control panel (replaces old portfolio widget)
│   │   ├── trading_widget.py           # Trading interface
│   │   ├── dashboard_widget.py         # Dashboard overview
│   │   ├── macro_widget.py             # Economic indicators
│   │   ├── ml_widget.py                # 🤖 ML Training & Models
│   │   ├── watchlist_widget.py         # 👁️ Stock Watchlist Management
│   │   ├── scanner_widget.py           # 🔍 Stock Scanner & Screener
│   │   └── charts_widget.py            # Chart components
│   ├── 📁 components/
│   │   ├── __init__.py
│   │   ├── navigation_bar.py           # Sidebar navigation
│   │   ├── status_bar.py               # Status indicators
│   │   ├── metric_cards.py             # Reusable metric displays
│   │   ├── message_bubble.py           # Chat message components
│   │   ├── table_components.py         # Enhanced table widgets
│   │   ├── ml_components.py            # 🧠 ML-specific UI components
│   │   ├── watchlist_components.py     # 📋 Watchlist items & cards
│   │   └── scanner_components.py       # 🔎 Scanner result components
│   └── 📁 dialogs/
│       ├── __init__.py
│       ├── trade_dialog.py             # Trade execution dialog
│       ├── settings_dialog.py          # App settings
│       ├── ml_training_dialog.py       # 🎯 ML Training configuration
│       ├── watchlist_dialog.py         # 📝 Add/Edit watchlist items
│       ├── scanner_config_dialog.py    # ⚙️ Scanner filter settings
│       └── about_dialog.py             # About dialog
├── 📁 core/
│   ├── __init__.py
│   ├── config.py                       # Configuration management
│   ├── signals.py                      # Custom PyQt signals
│   └── app_state.py                    # Application state management
├── 📁 services/
│   ├── __init__.py
│   ├── ibkr_service.py                 # IBKR API integration
│   ├── ai_service.py                   # AI APIs (OpenAI, Perplexity)
│   ├── market_data_service.py          # Market data fetching
│   ├── portfolio_service.py            # Portfolio calculations
│   ├── macro_data_service.py           # Economic data (FRED)
│   ├── ml_service.py                   # 🧠 ML Training & Prediction service
│   ├── watchlist_service.py            # 📋 Watchlist data management
│   └── scanner_service.py              # 🔍 Stock screening service
├── 📁 ml/                              # 🤖 Machine Learning Module
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── price_predictor.py          # Stock price prediction models
│   │   ├── trend_analyzer.py           # Market trend analysis
│   │   ├── risk_assessor.py            # Portfolio risk assessment
│   │   └── sentiment_analyzer.py       # News/social sentiment analysis
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── base_trainer.py             # Base training class
│   │   ├── lstm_trainer.py             # LSTM model trainer
│   │   ├── rf_trainer.py               # Random Forest trainer
│   │   └── ensemble_trainer.py         # Ensemble methods
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py              # Data loading utilities
│   │   ├── preprocessor.py             # Data preprocessing
│   │   └── feature_engineer.py         # Feature engineering
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                  # ML evaluation metrics
│       └── backtester.py               # Strategy backtesting
├── 📁 scanner/                         # 🔍 Stock Scanner Module
│   ├── __init__.py
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── technical_filters.py        # Technical analysis filters
│   │   ├── fundamental_filters.py      # Fundamental analysis filters
│   │   ├── price_filters.py            # Price/volume filters
│   │   └── custom_filters.py           # User-defined filters
│   ├── screeners/
│   │   ├── __init__.py
│   │   ├── momentum_screener.py        # Momentum stocks
│   │   ├── value_screener.py           # Value stocks
│   │   ├── breakout_screener.py        # Breakout patterns
│   │   └── earnings_screener.py        # Earnings-based screening
│   └── strategies/
│       ├── __init__.py
│       ├── preset_strategies.py        # Pre-built screening strategies
│       └── strategy_builder.py         # Custom strategy creation
├── 📁 watchlist/                       # 👁️ Watchlist Module
│   ├── __init__.py
│   ├── managers/
│   │   ├── __init__.py
│   │   ├── watchlist_manager.py        # Watchlist CRUD operations
│   │   ├── alert_manager.py            # Price/news alerts
│   │   └── sync_manager.py             # Cloud sync for watchlists
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── performance_analyzer.py     # Watchlist performance tracking
│   │   ├── correlation_analyzer.py     # Inter-stock correlations
│   │   └── sector_analyzer.py          # Sector distribution analysis
│   └── exporters/
│       ├── __init__.py
│       ├── csv_exporter.py             # CSV export functionality
│       └── pdf_reporter.py             # PDF report generation
├── 📁 models/
│   ├── __init__.py
│   ├── portfolio.py                    # Portfolio data models
│   ├── trade.py                        # Trade data models
│   ├── market_data.py                  # Market data structures
│   ├── chat_message.py                 # Chat message models
│   ├── ml_model.py                     # 🧠 ML model metadata
│   ├── watchlist_item.py               # 📋 Watchlist item model
│   └── scanner_result.py               # 🔍 Scanner result model
├── 📁 utils/
│   ├── __init__.py
│   ├── formatters.py                   # Data formatting utilities
│   ├── validators.py                   # Input validation
│   ├── calculators.py                  # Financial calculations
│   ├── threading.py                    # Threading utilities
│   ├── ml_utils.py                     # 🤖 ML helper functions
│   ├── scanner_utils.py                # 🔍 Scanner utilities
│   └── watchlist_utils.py              # 📋 Watchlist utilities
└── 📁 data/
    ├── __init__.py
    ├── cache_manager.py                # Data caching
    ├── data_store.py                   # Local data storage
    ├── ml_data/                        # 🧠 ML training data storage
    │   ├── models/                     # Saved ML models
    │   ├── datasets/                   # Training datasets
    │   └── features/                   # Feature data
    ├── watchlists/                     # 📋 Saved watchlists
    │   ├── personal/                   # User watchlists
    │   └── presets/                    # Preset watchlists
    └── scanner/                        # 🔍 Scanner data
        ├── results/                    # Scan results history
        └── strategies/                 # Saved scan strategies

📁 assets/
├── 📁 styles/
│   ├── dark_theme.qss                  # Main stylesheet (from user files)
│   ├── components.qss                  # Component-specific styles
│   ├── ml_theme.qss                    # 🧠 ML-specific styling
│   ├── watchlist_theme.qss             # 📋 Watchlist styling
│   ├── scanner_theme.qss               # 🔍 Scanner styling
│   └── animations.qss                  # Animation definitions
├── 📁 icons/
│   ├── app_icon.png
│   ├── navigation/
│   │   ├── ml_icon.svg                 # 🧠 ML tab icon
│   │   ├── watchlist_icon.svg          # 📋 Watchlist tab icon
│   │   └── scanner_icon.svg            # 🔍 Scanner tab icon
│   └── status/
└── 📁 fonts/
    └── custom_fonts.ttf

📁 tests/
├── 📁 unit/
│   ├── test_services/
│   ├── test_models/
│   ├── test_ml/
│   ├── test_scanner/
│   ├── test_watchlist/
│   └── test_utils/
└── 📁 integration/
    ├── test_ibkr_integration.py
    ├── test_ml_integration.py
    └── test_ui_workflow.py

📁 docs/
├── API.md                              # API documentation
├── SETUP.md                            # Setup instructions
├── ARCHITECTURE.md                     # System architecture
├── ML_GUIDE.md                         # ML module guide
├── SCANNER_GUIDE.md                    # Scanner module guide
└── WATCHLIST_GUIDE.md                  # Watchlist module guide

📁 legacy-streamlit/                    # Previous version (backup)
```

## 🎯 Module Features

### 🤖 ML (Machine Learning) Tab Features:
- **Data Loading**: טעינת נתונים היסטוריים לאימון
- **Model Training**: אימון מודלים שונים (LSTM, Random Forest, Ensemble)
- **Feature Engineering**: יצירת פיצ'רים טכניים ויסודיים
- **Backtesting**: בדיקת ביצועים היסטוריים
- **Real-time Predictions**: תחזיות בזמן אמת
- **Model Comparison**: השוואת מודלים שונים
- **Training Progress**: מעקב התקדמות אימון
- **Model Export/Import**: שמירה וטעינה של מודלים מאומנים

### 👁️ WATCHLIST Tab Features:
- **Multi-Watchlists**: ניהול רשימות מעקב מרובות
- **Real-time Updates**: עדכונים בזמן אמת של מחירים
- **Alerts System**: התראות על מחירים/חדשות
- **Performance Tracking**: מעקב ביצועים
- **Sector Analysis**: ניתוח חלוקה סקטוריאלית
- **Export/Import**: יצוא/יבוא רשימות
- **Notes & Tags**: הערות ותגיות למניות
- **Correlation Analysis**: ניתוח קורלציות בין מניות

### 🔍 SCANNER Tab Features:
- **Technical Screening**: סריקה לפי אינדיקטורים טכניים
- **Fundamental Screening**: סריקה לפי נתונים יסודיים
- **Custom Filters**: מסנני צפיה מותאמים אישית
- **Preset Strategies**: אסטרטגיות סריקה מוכנות מראש
- **Real-time Scanning**: סריקה בזמן אמת
- **Results History**: היסטוריית תוצאות סריקה
- **Filter Builder**: בונה מסננים חזותי
- **Export Results**: יצוא תוצאות סריקה

## 🔧 Technical Architecture

### Layer Separation:
- **UI Layer**: PyQt6 widgets and components
- **Service Layer**: API integrations and data services
- **Business Logic**: Core application logic and calculations
- **Data Layer**: Models, storage, and caching
- **Specialized Modules**: ML, Scanner, Watchlist modules

### Design Patterns:
- **MVC Pattern**: Model-View-Controller separation
- **Observer Pattern**: Signal/slot communication
- **Strategy Pattern**: For ML models and scanning strategies
- **Factory Pattern**: For creating different types of components
- **Singleton Pattern**: For configuration and app state

### Key Technologies:
- **PyQt6**: Modern GUI framework
- **pandas/numpy**: Data manipulation
- **scikit-learn/tensorflow**: Machine learning
- **TA-Lib/pandas-ta**: Technical analysis
- **ib_insync**: Interactive Brokers integration
- **asyncio**: Asynchronous programming

## 🚀 Development Roadmap

### Phase 1: Core Infrastructure
1. Create folder structure
2. Set up base classes and interfaces
3. Implement configuration system
4. Create main window skeleton

### Phase 2: Basic Modules
1. **Dashboard Widget**: Overview and metrics
2. **Portfolio Widget**: Basic portfolio tracking
3. **Chat Widget**: AI assistant functionality

### Phase 3: Trading Features
1. **Trading Widget**: Order execution interface
2. **IBKR Service**: Complete broker integration
3. **Market Data Service**: Real-time data feeds

### Phase 4: Analysis Modules
1. **WATCHLIST**: Stock monitoring and alerts
2. **SCANNER**: Stock screening and filtering
3. **Macro Widget**: Economic indicators

### Phase 5: Advanced Features
1. **ML Module**: Machine learning and predictions
2. **Advanced Charts**: Technical analysis charts
3. **Backtesting**: Strategy testing

### Phase 6: Polish & Optimization
1. Performance optimization
2. UI/UX improvements
3. Testing and bug fixes
4. Documentation completion

## 📚 Dependencies

### Core Dependencies:
```
PyQt6>=6.6.1
pandas>=2.1.4
numpy>=1.24.3
```

### Trading & Data:
```
ib_insync>=0.9.86
yfinance>=0.2.28
pandas-ta>=0.3.14b0
```

### AI & ML:
```
scikit-learn>=1.3.0
tensorflow>=2.13.0
openai>=1.6.1
```

### Styling & UI:
```
qdarkstyle>=3.2.3
qtawesome>=1.3.1
```

## 🎨 UI/UX Principles

### Design System:
- **Dark Theme**: Professional dark interface
- **Color Coding**: Consistent color usage (green=profit, red=loss, blue=info)
- **Typography**: Clear hierarchy and readability
- **Spacing**: Consistent margins and padding
- **Icons**: Meaningful and consistent iconography

### User Experience:
- **Responsive Design**: Adapts to different screen sizes
- **Real-time Updates**: Live data without manual refresh
- **Error Handling**: Graceful error messages and recovery
- **Performance**: Fast loading and smooth interactions
- **Accessibility**: Keyboard shortcuts and screen reader support

## 🔐 Security Considerations

### API Security:
- Environment variable storage for API keys
- Encrypted credential storage
- Secure communication protocols
- API rate limiting

### Data Security:
- Local data encryption
- Secure temporary file handling
- User data privacy protection
- Audit logging for trades

This architecture provides a solid foundation for a professional trading application with room for future enhancements and scalability.