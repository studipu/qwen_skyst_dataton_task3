"""
FastMCP Blockchain Analysis - Configuration
모든 설정을 통합 관리하는 설정 파일
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

@dataclass
class Config:
    """통합 설정 클래스"""
    
    # =============================================================================
    # API 설정
    # =============================================================================
    ALCHEMY_API_KEY: str = os.getenv('ALCHEMY_API_KEY', '')
    ALCHEMY_BASE_URL: str = "https://eth-mainnet.g.alchemy.com/v2"
    DEFILLAMA_BASE_URL: str = "https://api.llama.fi"
    
    # =============================================================================
    # 모델 설정
    # =============================================================================
    MODEL_NAME: str = "Qwen/Qwen2.5-1.5B-Instruct"
    CACHE_DIR: str = "./data/hf_models_cache"
    HUGGINGFACE_TOKEN: str = os.getenv('HUGGINGFACE_TOKEN', '')
    
    # 생성 파라미터
    MAX_NEW_TOKENS: int = 1024
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    TOP_K: int = 50
    REPETITION_PENALTY: float = 1.1
    
    # =============================================================================
    # 해커톤 데모 설정
    # =============================================================================
    DEMO_WALLET: str = "0x742d35Cc6798C532c5Ae5e4a5b6a4c7c4a24ba"
    
    DEMO_QUERIES: List[str] = [
        "Show me the current DeFi market overview",
        "What is the TVL of Uniswap protocol?",
        f"Analyze the portfolio for wallet {DEMO_WALLET}",
        "Get the transaction history for the same wallet",
        "Compare Uniswap and SushiSwap TVL"
    ]
    
    # =============================================================================
    # 시스템 설정
    # =============================================================================
    DEBUG_MODE: bool = os.getenv('DEBUG_MODE', '').lower() == 'true'
    API_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    CACHE_ENABLED: bool = True
    
    # =============================================================================
    # 알려진 토큰 및 프로토콜
    # =============================================================================
    KNOWN_TOKENS: Dict[str, str] = {
        'USDC': '0xA0b86a33E6417C9C4E1f03a6C0fA8b7b63Ad2C6A',
        'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
        'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
        'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
        'AAVE': '0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9',
        'COMP': '0xc00e94Cb662C3520282E6f5717214004A7f26888',
        'MKR': '0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2'
    }
    
    DEFI_PROTOCOLS: Dict[str, str] = {
        'uniswap': 'Uniswap V3',
        'aave': 'Aave',
        'compound': 'Compound',
        'makerdao': 'MakerDAO', 
        'sushiswap': 'SushiSwap',
        'curve': 'Curve Finance',
        'yearn': 'Yearn Finance',
        'synthetix': 'Synthetix'
    }
    
    # =============================================================================
    # 검증 메서드
    # =============================================================================
    @classmethod
    def validate(cls) -> bool:
        """설정 유효성 검사"""
        config = cls()
        issues = []
        
        # 필수 API 키 검사
        if not config.ALCHEMY_API_KEY:
            issues.append("ALCHEMY_API_KEY is required")
        
        # 모델 설정 검사
        if not config.MODEL_NAME:
            issues.append("MODEL_NAME is required")
        
        # 디렉토리 생성
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        os.makedirs("./data/api_cache", exist_ok=True)
        os.makedirs("./examples", exist_ok=True)
        
        if issues:
            print("❌ Configuration Issues:")
            for issue in issues:
                print(f"  • {issue}")
            return False
        
        print("✅ Configuration validated successfully")
        if config.DEBUG_MODE:
            print("🐛 Debug mode enabled")
        
        return True
    
    @classmethod
    def print_summary(cls):
        """설정 요약 출력"""
        config = cls()
        
        print("\n🔧 System Configuration:")
        print(f"  Model: {config.MODEL_NAME}")
        print(f"  Cache: {config.CACHE_DIR}")
        print(f"  Alchemy API: {'✅ Set' if config.ALCHEMY_API_KEY else '❌ Missing'}")
        print(f"  HuggingFace Token: {'✅ Set' if config.HUGGINGFACE_TOKEN else '⚠️  Optional'}")
        print(f"  Debug Mode: {'✅ On' if config.DEBUG_MODE else '❌ Off'}")
        print(f"  Demo Wallet: {config.DEMO_WALLET}")
        print(f"  Demo Queries: {len(config.DEMO_QUERIES)} prepared")

# 전역 설정 인스턴스
CONFIG = Config()

# 편의를 위한 상수들
ALCHEMY_API_KEY = CONFIG.ALCHEMY_API_KEY
MODEL_NAME = CONFIG.MODEL_NAME
CACHE_DIR = CONFIG.CACHE_DIR
DEBUG_MODE = CONFIG.DEBUG_MODE
DEMO_QUERIES = CONFIG.DEMO_QUERIES
DEMO_WALLET = CONFIG.DEMO_WALLET

if __name__ == "__main__":
    print("🔧 FastMCP Blockchain Analysis - Configuration Check")
    Config.validate()
    Config.print_summary()