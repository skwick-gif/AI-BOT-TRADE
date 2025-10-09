"""
Configuration Manager for Trading Bot
Handles environment variables and application settings
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class IBKRConfig:
    """IBKR Configuration"""
    host: str = "127.0.0.1"
    port: int = 4001  # IB Gateway (was 7497 for TWS Paper Trading)
    client_id: int = 1
    connect_timeout: int = 10
    request_timeout: int = 30
    account_code: str = ""


@dataclass
class PerplexityConfig:
    """Perplexity API Configuration"""
    api_key: str = ""
    model: str = "reasoning-pro"  # default model; AIService will fallback if invalid
    # Finance-specific overrides: when force_finance=True the app will prefer the
    # finance_model for all Perplexity calls and add finance-focused search filters
    force_finance: bool = True
    finance_model: str = "llama-3.1-sonar-small-128k-online"
    # Comma-separated list of domains to prefer for finance answers (used as search_domain_filter)
    search_domains: str = "finance.yahoo.com,marketwatch.com,bloomberg.com,reuters.com"
    # How recent search results should be (e.g. 'week', 'month', 'day')
    search_recency: str = "week"
    max_tokens: int = 2000


@dataclass
class MLConfig:
    """Machine Learning Configuration"""
    model_save_path: str = "models"
    data_cache_path: str = "cache"
    training_batch_size: int = 32
    validation_split: float = 0.2
    random_seed: int = 42


@dataclass
class UIConfig:
    """UI Configuration"""
    theme: str = "dark"
    update_interval: int = 1000  # milliseconds
    chart_timeframe: str = "1D"
    default_symbols: list = None


class ConfigManager:
    """Central configuration manager"""
    
    def __init__(self, env_file: Optional[str] = None):
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            # Look for .env file in project root
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
            if env_path.exists():
                load_dotenv(env_path)
        
        # Initialize configurations
        self.ibkr = self._load_ibkr_config()
        self.perplexity = self._load_perplexity_config()
        self.ml = self._load_ml_config()
        self.ui = self._load_ui_config()

    # ---- Helpers to safely parse environment values ----
    def _get_int(self, name: str, default: int) -> int:
        val = os.getenv(name)
        if val is None or str(val).strip() == "":
            return default
        try:
            return int(val)
        except Exception:
            return default

    def _get_float(self, name: str, default: float) -> float:
        val = os.getenv(name)
        if val is None or str(val).strip() == "":
            return default
        try:
            return float(val)
        except Exception:
            return default
    
    def _load_ibkr_config(self) -> IBKRConfig:
        """Load IBKR configuration from environment"""
        return IBKRConfig(
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            port=self._get_int("IBKR_PORT", 4001),  # IB Gateway default
            client_id=self._get_int("IBKR_CLIENT_ID", 1),
            connect_timeout=self._get_int("IBKR_CONNECT_TIMEOUT", 10),
            request_timeout=self._get_int("IBKR_REQUEST_TIMEOUT", 30),
            account_code=os.getenv("IBKR_ACCOUNT_CODE", "").strip()
        )
    
    def _load_perplexity_config(self) -> PerplexityConfig:
        """Load Perplexity configuration from environment"""
        return PerplexityConfig(
            api_key=os.getenv("PERPLEXITY_API_KEY", ""),
            model=os.getenv("PERPLEXITY_MODEL", "reasoning-pro"),
            force_finance=(os.getenv("PERPLEXITY_FORCE_FINANCE", "true").lower() in ("1","true","yes")),
            finance_model=os.getenv("PERPLEXITY_FINANCE_MODEL", "sonar"),
            search_domains=os.getenv("PERPLEXITY_SEARCH_DOMAINS", "finance.yahoo.com,marketwatch.com,bloomberg.com,reuters.com"),
            search_recency=os.getenv("PERPLEXITY_SEARCH_RECENCY", "week"),
            max_tokens=self._get_int("PERPLEXITY_MAX_TOKENS", 2000)
        )
    
    def _load_ml_config(self) -> MLConfig:
        """Load ML configuration from environment"""
        return MLConfig(
            model_save_path=os.getenv("ML_MODEL_PATH", "models"),
            data_cache_path=os.getenv("ML_CACHE_PATH", "cache"),
            training_batch_size=self._get_int("ML_BATCH_SIZE", 32),
            validation_split=self._get_float("ML_VALIDATION_SPLIT", 0.2),
            random_seed=self._get_int("ML_RANDOM_SEED", 42)
        )
    
    def _load_ui_config(self) -> UIConfig:
        """Load UI configuration from environment"""
        # By default do not seed the watchlist with hardcoded tickers.
        # Users can set DEFAULT_SYMBOLS in their environment or .env if they
        # want an initial list (comma-separated). If not set, start empty.
        ds_raw = os.getenv("DEFAULT_SYMBOLS", "")
        default_symbols = [s.strip().upper() for s in ds_raw.split(",") if s.strip()] if ds_raw else []
        
        return UIConfig(
            theme=os.getenv("UI_THEME", "dark"),
            update_interval=self._get_int("UI_UPDATE_INTERVAL", 1000),
            chart_timeframe=os.getenv("UI_CHART_TIMEFRAME", "1D"),
            default_symbols=default_symbols
        )
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            "ibkr": self.ibkr.__dict__,
            "perplexity": self.perplexity.__dict__,
            "ml": self.ml.__dict__,
            "ui": self.ui.__dict__
        }
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration and return status"""
        validation = {
            "perplexity_api_key": bool(self.perplexity.api_key),
            "ibkr_config": all([
                self.ibkr.host,
                self.ibkr.port > 0,
                self.ibkr.client_id >= 0
            ])
        }
        return validation