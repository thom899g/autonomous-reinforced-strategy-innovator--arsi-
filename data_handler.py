"""
Market data handler with real-time streaming and historical data fetching
Uses CCXT for exchange data with robust error handling
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ccxt
import logging
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Supported data sources"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    SIMULATION = "simulation"

@dataclass
class MarketData:
    """Structured market data container"""
    symbol: str
    timeframe: str
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    timestamp: np.ndarray
    
class DataHandler:
    """Robust market data handler with error recovery and caching"""
    
    def __init__(self, 
                 exchange_id: str = "binance",
                 enable_rate_limit: bool = True,
                 cache_size: int = 1000):
        """
        Initialize data handler
        
        Args:
            exchange_id: CCXT exchange ID
            enable_rate_limit: Enable CCXT rate limiting
            cache_size: Number of candles to cache per symbol
        """
        self.exchange_id = exchange_id
        self.cache_size = cache_size
        self.exchange = None
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self._initialize_exchange(enable_rate_limit)
        
    def _initialize_exchange(self, enable_rate_limit: bool) -> None:
        """Initialize CCXT exchange with error handling"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': enable_rate_limit,
                'options': {'defaultType': 'spot'}
            })
            
            # Test connection
            self.exchange.load_markets()
            logger.info(f"Successfully initialized {self.exchange_id} exchange")
            
        except AttributeError as e:
            logger.error(f"Exchange {self.exchange_id} not found in CCXT