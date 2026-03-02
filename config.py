"""
Configuration management for ARSI system with environment validation
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Firebase/Firestore configuration"""
    firebase_credentials_path: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
    firestore_collection: str = os.getenv("FIRESTORE_COLLECTION", "trading_strategies")
    realtime_db_url: str = os.getenv("FIREBASE_REALTIME_DB_URL", "")

@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    symbols: List[str] = None
    timeframe: str = "1h"
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # 10% of balance
    transaction_fee: float = 0.001  # 0.1% fee
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC/USDT", "ETH/USDT"]

@dataclass
class RLConfig:
    """Reinforcement Learning configuration"""
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0003
    buffer_size: int = 100000
    batch_size: int = 64
    tau: float = 0.005  # Target network update rate
    gamma: float = 0.99  # Discount factor
    train_freq: int = 4
    gradient_steps: int = 1
    
@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: int = logging.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/arsi_system.log"

class Config:
    """Main configuration singleton"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.db = DatabaseConfig()
            cls._instance.trading = TradingConfig()
            cls._instance.rl = RLConfig()
            cls._instance.logging = LoggingConfig()
            cls._instance._validate()
        return cls._instance
    
    def _validate(self):
        """Validate configuration"""
        if not os.path.exists(self.db.firebase_credentials_path):
            logging.warning(f"Firebase credentials not found at {self.db.firebase_credentials_path}")
        
        if self.trading.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        
        if self.trading.max_position_size <= 0 or self.trading.max_position_size > 1:
            raise ValueError("Max position size must be between 0 and 1")

# Initialize configuration
config = Config()