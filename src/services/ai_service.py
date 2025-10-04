"""
AI Service for trading assistant integration
"""

import asyncio
import json
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
        
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.perplexity.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.perplexity.model,
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
        
        # Create a transient session if none is available, and close it after use.
        # This prevents retaining a ClientSession that is bound to an event loop
        # that may be closed by the caller between invocations.
        created_here = False
        if not self.session or getattr(self.session, "closed", False):
            self.session = aiohttp.ClientSession()
            created_here = True

        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    # Handle invalid model error by attempting a fallback
                    if response.status == 400 and 'invalid_model' in error_text.lower():
                        # Try a fallback model name
                        fallback = 'sonar'
                        if self.config.perplexity.model != fallback:
                            payload_fallback = dict(payload)
                            payload_fallback['model'] = fallback
                            async with self.session.post(url, headers=headers, json=payload_fallback) as resp2:
                                if resp2.status == 200:
                                    data2 = await resp2.json()
                                    return data2['choices'][0]['message']['content']
                                else:
                                    error_text2 = await resp2.text()
                                    raise Exception(f"API call failed (fallback) with status {resp2.status}: {error_text2}")
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
            f"You are an AI trader scoring a trading opportunity for symbol '{symbol}'. "
            f"{strat_txt} Return ONLY JSON with keys score (0-10), action (BUY/SELL/HOLD), reason (short). "
            f"Do not include any commentary outside JSON. {ctx_txt}"
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
                obj = {"score": score, "action": "HOLD", "reason": text[:120]}

        # normalize fields
        score = float(obj.get("score", 5.0))
        action = str(obj.get("action", "HOLD")).upper()
        reason = str(obj.get("reason", ""))

        # enforce thresholds consistency if provided
        if strategy:
            bt = float(strategy.get("buy_threshold", 8.0))
            st = float(strategy.get("sell_threshold", 4.0))
            if score >= bt:
                action = "BUY"
            elif score <= st:
                action = "SELL"
            else:
                action = "HOLD"

        return {"score": score, "action": action, "reason": reason}

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