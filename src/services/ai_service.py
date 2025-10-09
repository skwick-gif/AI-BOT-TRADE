"""
AI Service for trading assistant integration
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List
import aiohttp
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter

from core.config_manager import ConfigManager
from utils.logger import get_logger
from services.prompt_builder import build_numeric_score_prompt


class AIService:
    """AI service for trading assistant functionality"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = get_logger("AIService")
        self.session: Optional[aiohttp.ClientSession] = None
        self.http: Optional[requests.Session] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None
        # do not close self.http here; allow reuse across app lifetime

    def get_http_session(self) -> requests.Session:
        """Lazily create and return a pooled HTTP session for sync requests."""
        if self.http is None:
            s = requests.Session()
            adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=0)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            # optional: default headers could be set here if needed
            self.http = s
        return self.http
    
    async def get_ai_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get AI response from Perplexity API
        
        Args:
            message: User message
            context: Optional context information (portfolio, market data, etc.)
            
        Returns:
            AI response string
        """
        try:
            # Prepare the prompt with context
            enhanced_prompt = self._enhance_prompt(message, context)
            
            # Call Perplexity API
            response = await self._call_perplexity_api(enhanced_prompt)
            
            self.logger.debug(f"AI response generated for message: {message[:50]}...")
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting AI response: {e}")
            return f"I apologize, but I encountered an error processing your request: {e}"
    
    def _enhance_prompt(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance user prompt with trading context
        
        Args:
            message: Original user message
            context: Trading context (portfolio, market data, etc.)
            
        Returns:
            Enhanced prompt string
        """
        system_prompt = """You are an expert AI trading assistant with deep knowledge of:
- Financial markets and trading strategies
- Technical and fundamental analysis
- Risk management and portfolio optimization
- Market psychology and sentiment analysis
- Options, futures, and derivatives
- Cryptocurrency and forex markets

You provide professional, actionable trading advice while emphasizing risk management.
Always consider the user's portfolio context when available.
Be concise but thorough in your responses.
Include specific actionable recommendations when appropriate.
Always remind users about risk management and due diligence."""

        enhanced_message = f"{system_prompt}\\n\\nUser Query: {message}"
        
        # Add context if available
        if context:
            context_str = self._format_context(context)
            enhanced_message += f"\\n\\nContext: {context_str}"
        
        return enhanced_message
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for the AI prompt"""
        context_parts = []
        
        # Portfolio information
        if 'portfolio' in context:
            portfolio = context['portfolio']
            context_parts.append(f"Portfolio: {len(portfolio)} positions")
            for position in portfolio[:5]:  # Limit to top 5 positions
                symbol = position.get('symbol', 'Unknown')
                pnl = position.get('unrealized_pnl', 0)
                context_parts.append(f"- {symbol}: ${pnl:+,.2f} P&L")
        
        # Account summary
        if 'account' in context:
            account = context['account']
            net_liq = account.get('NetLiquidation', {}).get('value', 'N/A')
            buying_power = account.get('BuyingPower', {}).get('value', 'N/A')
            context_parts.append(f"Account: Net Liquidation ${net_liq}, Buying Power ${buying_power}")
        
        # Market data
        if 'market_data' in context:
            market = context['market_data']
            context_parts.append(f"Market: {market}")
        
        # Current time
        context_parts.append(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return " | ".join(context_parts)
    
    async def _call_perplexity_api(self, prompt: str) -> str:
        """
        Call Perplexity API
        
        Args:
            prompt: Enhanced prompt string
            
        Returns:
            API response text
        """
        if not self.config.perplexity.api_key:
            return "Perplexity API key not configured. Please add your API key to the .env file."
        
        # Record API call start time
        api_start_time = time.time()
        
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.perplexity.api_key}",
            "Content-Type": "application/json"
        }
        
        # If force_finance is enabled, prefer the finance_model and add search filters
        model_to_use = self.config.perplexity.finance_model if getattr(self.config.perplexity, 'force_finance', False) else self.config.perplexity.model
        payload = {
            "model": model_to_use,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.perplexity.max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }
        # Add finance-specific search filters if requested
        try:
            if getattr(self.config.perplexity, 'force_finance', False):
                domains = getattr(self.config.perplexity, 'search_domains', None)
                if domains:
                    payload['search_domain_filter'] = [d.strip() for d in str(domains).split(',') if d.strip()]
                recency = getattr(self.config.perplexity, 'search_recency', None)
                if recency:
                    payload['search_recency_filter'] = recency
        except Exception:
            pass
        
        self.logger.debug(f"Calling Perplexity API with model: {self.config.perplexity.model}")
        
        # Create a transient session if none is available, and close it after use.
        # This prevents retaining a ClientSession that is bound to an event loop
        # that may be closed by the caller between invocations.
        created_here = False
        if not self.session or getattr(self.session, "closed", False):
            # Set timeout for API calls (30 seconds)
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            created_here = True

        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    api_response_time = time.time() - api_start_time
                    self.logger.info(f"Perplexity API call successful in {api_response_time:.2f}s")
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    # Handle invalid model error by attempting a fallback
                    if response.status == 400 and 'invalid_model' in error_text.lower():
                        self.logger.warning(f"Invalid model {self.config.perplexity.model}, trying fallback...")
                        # Try a fallback model name
                        fallback = 'sonar'
                        if self.config.perplexity.model != fallback:
                            payload_fallback = dict(payload)
                            payload_fallback['model'] = fallback
                            async with self.session.post(url, headers=headers, json=payload_fallback) as resp2:
                                if resp2.status == 200:
                                    data2 = await resp2.json()
                                    api_response_time = time.time() - api_start_time
                                    self.logger.info(f"Perplexity API fallback call successful in {api_response_time:.2f}s")
                                    return data2['choices'][0]['message']['content']
                                else:
                                    error_text2 = await resp2.text()
                                    api_response_time = time.time() - api_start_time
                                    self.logger.error(f"API fallback call failed after {api_response_time:.2f}s")
                                    raise Exception(f"API call failed (fallback) with status {resp2.status}: {error_text2}")
                    api_response_time = time.time() - api_start_time
                    self.logger.error(f"API call failed after {api_response_time:.2f}s with status {response.status}")
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
        finally:
            if created_here and self.session:
                await self.session.close()
                self.session = None
    
    def analyze_portfolio(self, portfolio_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze portfolio data and return insights
        
        Args:
            portfolio_data: List of position dictionaries
            
        Returns:
            Portfolio analysis dictionary
        """
        if not portfolio_data:
            return {"message": "No portfolio data available"}
        
        analysis = {
            "total_positions": len(portfolio_data),
            "total_value": sum(pos.get('market_value', 0) for pos in portfolio_data),
            "total_pnl": sum(pos.get('unrealized_pnl', 0) for pos in portfolio_data),
            "winning_positions": len([p for p in portfolio_data if p.get('unrealized_pnl', 0) > 0]),
            "losing_positions": len([p for p in portfolio_data if p.get('unrealized_pnl', 0) < 0]),
            "largest_position": max(portfolio_data, key=lambda x: abs(x.get('market_value', 0)), default={}),
            "best_performer": max(portfolio_data, key=lambda x: x.get('unrealized_pnl', 0), default={}),
            "worst_performer": min(portfolio_data, key=lambda x: x.get('unrealized_pnl', 0), default={})
        }
        
        # Calculate win rate
        if analysis["total_positions"] > 0:
            analysis["win_rate"] = analysis["winning_positions"] / analysis["total_positions"] * 100
        else:
            analysis["win_rate"] = 0
        
        return analysis
    
    def generate_trading_insights(self, market_data: Dict[str, Any]) -> List[str]:
        """
        Generate trading insights based on market data
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Market sentiment analysis
        if 'vix' in market_data:
            vix = float(market_data['vix'])
            if vix > 25:
                insights.append("🟡 High volatility (VIX > 25) - Consider defensive positions")
            elif vix < 15:
                insights.append("🟢 Low volatility (VIX < 15) - Good environment for growth plays")
        
        # Index performance
        if 'sp500_change' in market_data:
            sp500_change = float(market_data['sp500_change'])
            if sp500_change > 2:
                insights.append("🟢 Strong market rally - Consider momentum strategies")
            elif sp500_change < -2:
                insights.append("🔴 Market decline - Look for oversold opportunities")
        
        # Sector rotation insights
        insights.append("💡 Consider diversification across sectors")
        insights.append("⚠️ Always use proper risk management")
        
        return insights
    
    async def get_market_summary(self) -> str:
        """
        Get AI-generated market summary
        
        Returns:
            Market summary string
        """
        prompt = """Provide a brief market summary including:
        - Major index performance  
        - Key market drivers and news
        - Sector performance highlights
        - Risk factors to watch
        
        Keep it concise and actionable for active traders."""
        
        return await self.get_ai_response(prompt)
    
    async def get_trading_idea(self, market_context: Optional[str] = None) -> str:
        """
        Get AI-generated trading idea
        
        Args:
            market_context: Optional market context
            
        Returns:
            Trading idea string
        """
        prompt = "Suggest a trading strategy based on current market conditions. Include entry strategy, risk management, and rationale."
        
        if market_context:
            prompt += f" Market context: {market_context}"
        
        return await self.get_ai_response(prompt)

    async def score_symbol(self, symbol: str, *, strategy: Optional[Dict[str, Any]] = None, market_context: Optional[Dict[str, Any]] = None, timeout: float = 15.0) -> Dict[str, Any]:
        """Return structured score for a symbol using Perplexity.

        Output schema:
        { "score": number (0-10), "action": "BUY"|"SELL"|"HOLD", "reason": string }

        The action should be consistent with thresholds if provided in `strategy`:
        - BUY if score >= buy_threshold
        - SELL if score <= sell_threshold
        - otherwise HOLD
        """
        if not self.config.perplexity.api_key:
            raise RuntimeError("Perplexity API key not configured")

        strat_txt = ""
        if strategy:
            bt = strategy.get("buy_threshold", 8.0)
            st = strategy.get("sell_threshold", 4.0)
            strat_txt = f"Use thresholds: buy_threshold={bt}, sell_threshold={st}."

        ctx_txt = ""
        if market_context:
            try:
                ctx_txt = f"Context: {json.dumps(market_context)[:1000]}"
            except Exception:
                ctx_txt = ""

        prompt = (
            f"You are an AI financial analyst specialized in short-term equity trading.\n\n"
            f"Analyze stock symbol \"{symbol}\" based on the latest market and sentiment data (provided separately).\n"
            f"Forecast the expected 7-day price movement.\n\n"
            f"Return **only valid JSON** — no text, no markdown, no explanations.\n\n"
            f"Output format:\n"
            f"{{\n"
            f"  \"score\": <integer 0–10>, \n"
            f"  \"price_target\": <float, USD>\n"
            f"}}"
        )

        # call API
        text = await self._call_perplexity_api(prompt)

        # attempt to parse JSON
        obj: Dict[str, Any]
        try:
            obj = json.loads(text)
        except Exception:
            # try to extract JSON block
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                obj = json.loads(m.group(0))
            else:
                # last resort: try to parse a number score from text
                try:
                    import re
                    num = re.search(r"([0-9]+(\.[0-9]+)?)", text)
                    score = float(num.group(1)) if num else 5.0
                except Exception:
                    score = 5.0
                obj = {"score": score, "price_target": None}

        # normalize fields
        score = float(obj.get("score", 5.0))
        price_target = obj.get("price_target", None)

        # Try to convert price_target to float if it exists
        if price_target is not None:
            try:
                price_target = float(price_target)
            except (ValueError, TypeError):
                price_target = None

        return {"score": score, "price_target": price_target}

    def score_symbol_numeric_sync(self, symbol: str, *, timeout: float = 8.0, market_context: Optional[Dict[str, Any]] = None, profile: Optional[str] = None, thresholds: Optional[Dict[str, float]] = None) -> float:
        """Return ONLY a numeric score 0-10 for symbol using a minimal prompt (synchronous HTTP).

        Raises on failure; caller decides what to do (no dummy fallback).
        """
        if not self.config.perplexity.api_key:
            raise RuntimeError("Perplexity API key not configured")

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.perplexity.api_key}",
            "Content-Type": "application/json",
        }
        # compact, profile-aware numeric-only prompt
        prompt = build_numeric_score_prompt(symbol, profile=profile, market_context=market_context, thresholds=thresholds)
        payload = {
            "model": self.config.perplexity.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False,
        }
        session = self.get_http_session()
        resp = session.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        # try parse float from entire text
        try:
            # remove any extraneous characters
            cleaned = text.strip().split()[0].strip('`\"')
            val = float(cleaned)
            # bound 0..10
            if val < 0:
                val = 0.0
            if val > 10:
                val = 10.0
            return val
        except Exception:
            # last attempt: find a number inside
            import re
            m = re.search(r"([0-9]+(\.[0-9]+)?)", text)
            if not m:
                raise ValueError(f"Non-numeric response: {text[:120]}")
            val = float(m.group(1))
            if val < 0:
                val = 0.0
            if val > 10:
                val = 10.0
            return val

    def score_with_custom_prompt_numeric_sync(self, prompt: str, *, timeout: float = 10.0) -> float:
        """Send a raw custom prompt to Perplexity and parse ONLY a numeric 0-10 response.

        This is used for power-users who want to control the exact prompt. Caller is responsible
        to ensure the prompt instructs the model to output only a number.
        """
        if not self.config.perplexity.api_key:
            raise RuntimeError("Perplexity API key not configured")
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.perplexity.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.perplexity.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False,
        }
        session = self.get_http_session()
        resp = session.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        try:
            cleaned = text.strip().split()[0].strip('`"')
            val = float(cleaned)
        except Exception:
            import re
            m = re.search(r"([0-9]+(\.[0-9]+)?)", text)
            if not m:
                raise ValueError(f"Non-numeric response: {text[:120]}")
            val = float(m.group(1))
        if val < 0:
            val = 0.0
        if val > 10:
            val = 10.0
        return val

    def analyze_stock_simple(self, prompt: str, symbol: str, timeout: int = 60) -> Optional[str]:
        """
        Simple stock analysis using Perplexity API - returns raw response
        
        Args:
            prompt: Analysis prompt
            symbol: Stock symbol
            timeout: Request timeout in seconds
            
        Returns:
            Raw AI response string or None if failed
        """
        try:
            self.logger.info(f"Starting AI analysis for {symbol}")
            
            # Get API key
            api_key = self.config.get("perplexity_api_key")
            if not api_key:
                self.logger.error("Perplexity API key not configured")
                return None
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Build payload for simple analysis using config-driven model and filters
            try:
                model_to_use = self.config.perplexity.finance_model if getattr(self.config.perplexity, 'force_finance', False) else self.config.perplexity.model
            except Exception:
                model_to_use = "llama-3.1-sonar-small-128k-online"

            payload = {
                "model": model_to_use,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional stock analyst. Provide concise, numerical assessments."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.9,
                "return_citations": False,
                "top_k": 0,
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 1
            }
            # Apply configured domain and recency filters when forcing finance
            try:
                if getattr(self.config.perplexity, 'force_finance', False):
                    domains = getattr(self.config.perplexity, 'search_domains', None)
                    if domains:
                        payload['search_domain_filter'] = [d.strip() for d in str(domains).split(',') if d.strip()]
                    recency = getattr(self.config.perplexity, 'search_recency', None)
                    if recency:
                        payload['search_recency_filter'] = recency
            except Exception:
                pass
            
            # Make request with timeout
            session = self.get_http_session()
            response = session.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code != 200:
                self.logger.error(f"Perplexity API error {response.status_code}: {response.text}")
                return None
            
            # Parse response
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0].get('message', {}).get('content', '')
                self.logger.info(f"AI analysis for {symbol} complete: {content[:50]}...")
                return content.strip()
            else:
                self.logger.error(f"Unexpected API response format: {data}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error(f"AI analysis timeout for {symbol}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"AI analysis request error for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"AI analysis error for {symbol}: {e}")
            return None