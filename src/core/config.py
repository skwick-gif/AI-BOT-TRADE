"""
Configuration management for the trading application
"""

import os
from dotenv import load_dotenv

def load_configuration():
    """Load configuration from environment variables"""
    # Load .env file
    load_dotenv()
    
    config = {
        # API Keys
        'PERPLEXITY_API_KEY': os.environ.get('PERPLEXITY_API_KEY'),
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
        'QUANTIQ_API_KEY': os.environ.get('QUANTIQ_API'),
        'FRED_API_KEY': os.environ.get('FRED_API_KEY'),
        'FMP_API_KEY': os.environ.get('FMP_API_KEY'),
        'NEWSAPI_KEY': os.environ.get('NEWSAPI_KEY'),
        
        # Reddit API
        'REDDIT_CLIENT_ID': os.environ.get('REDDIT_CLIENT_ID'),
        'REDDIT_CLIENT_SECRET': os.environ.get('REDDIT_CLIENT_SECRET'),
        'REDDIT_USER_AGENT': os.environ.get('REDDIT_USER_AGENT', 'StockAnalysisBot/1.0'),
        
        # IBKR Configuration
        'IBKR_HOST': os.environ.get('IBKR_HOST', '127.0.0.1'),
        'IBKR_PORT': int(os.environ.get('IBKR_PORT', '4001')),  # IB Gateway default
        'IBKR_CLIENT_ID': int(os.environ.get('IBKR_CLIENT_ID', '99')),
        
        # Application Settings
        'DEBUG': os.environ.get('DEBUG', 'False').lower() == 'true',
        'LOG_LEVEL': os.environ.get('LOG_LEVEL', 'INFO'),
    }
    
    return config

def validate_api_keys(config):
    """Validate that required API keys are present"""
    required_keys = ['PERPLEXITY_API_KEY', 'OPENAI_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not config.get(key) or config[key] == f'your_{key.lower()}_here':
            missing_keys.append(key)
    
    return missing_keys

def get_ibkr_config(config):
    """Get IBKR specific configuration"""
    return {
        'host': config['IBKR_HOST'],
        'port': config['IBKR_PORT'],
        'client_id': config['IBKR_CLIENT_ID']
    }