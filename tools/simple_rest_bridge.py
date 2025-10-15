"""
Simple REST Bridge using FastAPI
Provides the same interface as InterReactBridge but uses our IBKRAdapterService
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = FastAPI(title="IBKR REST Bridge")

# Initialize service lazily
ibkr_service = None

def get_ibkr_service():
    global ibkr_service
    if ibkr_service is None:
        from services.ibkr_adapter_service import IBKRAdapterService
        from core.config_manager import ConfigManager
        config = ConfigManager()
        ibkr_service = IBKRAdapterService(config.ibkr)
    return ibkr_service

class ConnectRequest(BaseModel):
    host: str
    port: int
    clientId: int

class OrderRequest(BaseModel):
    symbol: str
    secType: str = "STK"
    exchange: str = "SMART"
    action: str
    quantity: int
    price: float = None
    orderType: str = "LMT"

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5001)