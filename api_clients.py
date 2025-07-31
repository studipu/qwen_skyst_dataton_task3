"""
FastMCP Blockchain Analysis - API Clients
Alchemy + DefiLlama API 클라이언트 통합
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from config import CONFIG

# 로깅 설정
logging.basicConfig(level=logging.INFO if CONFIG.DEBUG_MODE else logging.WARNING)
logger = logging.getLogger(__name__)

class BlockchainAPIClient:
    """통합 블록체인 API 클라이언트"""
    
    def __init__(self):
        self.alchemy_url = f"{CONFIG.ALCHEMY_BASE_URL}/{CONFIG.ALCHEMY_API_KEY}"
        self.defillama_url = CONFIG.DEFILLAMA_BASE_URL
        self.timeout = CONFIG.API_TIMEOUT
        self.max_retries = CONFIG.MAX_RETRIES
        
        if CONFIG.DEBUG_MODE:
            logger.info("🔧 BlockchainAPIClient initialized")
    
    async def _make_request(self, session: aiohttp.ClientSession, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """공통 HTTP 요청 처리"""
        for attempt in range(self.max_retries):
            try:
                if CONFIG.DEBUG_MODE:
                    logger.info(f"🌐 {method.upper()} {url} (attempt {attempt + 1})")
                
                async with session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        data = await response.json()
                        if CONFIG.DEBUG_MODE:
                            logger.info(f"✅ Request successful")
                        return data
                    else:
                        error_text = await response.text()
                        logger.warning(f"❌ HTTP {response.status}: {error_text}")
                        
                        if attempt == self.max_retries - 1:
                            return {'error': f'HTTP {response.status}: {error_text}'}
                        
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    return {'error': 'Request timeout'}
                    
            except Exception as e:
                logger.warning(f"⚠️ Exception on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return {'error': f'Request failed: {str(e)}'}
            
            # 재시도 전 대기
            await asyncio.sleep(2 ** attempt)
        
        return {'error': 'Max retries exceeded'}
    
    # =============================================================================
    # Alchemy API Methods
    # =============================================================================
    
    async def get_token_balances(self, wallet_address: str, token_addresses: List[str] = None) -> Dict[str, Any]:
        """지갑의 토큰 잔액 조회 (Alchemy)"""
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenBalances",
            "params": [wallet_address]
        }
        
        if token_addresses:
            payload["params"].append(token_addresses)
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            result = await self._make_request(session, 'POST', self.alchemy_url, json=payload)
            
            if 'error' in result:
                return result
            
            return result.get('result', {})
    
    async def get_token_metadata(self, token_address: str) -> Dict[str, Any]:
        """토큰 메타데이터 조회 (Alchemy)"""
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenMetadata",
            "params": [token_address]
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            result = await self._make_request(session, 'POST', self.alchemy_url, json=payload)
            
            if 'error' in result:
                return result
            
            return result.get('result', {})
    
    async def get_asset_transfers(self, wallet_address: str, category: List[str] = None, max_count: int = 50) -> Dict[str, Any]:
        """자산 전송 기록 조회 (Alchemy)"""
        if category is None:
            category = ["external", "internal", "erc20", "erc721", "erc1155"]
        
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [{
                "fromAddress": wallet_address,
                "category": category,
                "fromBlock": "0x" + format(19000000, 'x'),  # 최근 블록부터
                "toBlock": "latest",
                "maxCount": max_count
            }]
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            result = await self._make_request(session, 'POST', self.alchemy_url, json=payload)
            
            if 'error' in result:
                return result
            
            return result.get('result', {})
    
    # =============================================================================
    # DefiLlama API Methods  
    # =============================================================================
    
    async def get_protocol_tvl(self, protocol: str) -> Dict[str, Any]:
        """프로토콜 TVL 조회 (DefiLlama)"""
        url = f"{self.defillama_url}/protocol/{protocol}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            return await self._make_request(session, 'GET', url)
    
    async def get_chains_tvl(self) -> Dict[str, Any]:
        """모든 체인의 TVL 조회 (DefiLlama)"""
        url = f"{self.defillama_url}/chains"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            result = await self._make_request(session, 'GET', url)
            
            if 'error' in result:
                return result
            
            # 리스트를 딕셔너리로 변환 (체인명을 키로)
            if isinstance(result, list):
                return {"chains": result}
            
            return result
    
    async def get_protocols_list(self) -> Dict[str, Any]:
        """DeFi 프로토콜 목록 조회 (DefiLlama)"""
        url = f"{self.defillama_url}/protocols"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            result = await self._make_request(session, 'GET', url)
            
            if 'error' in result:
                return result
            
            # 리스트를 딕셔너리로 변환
            if isinstance(result, list):
                return {"protocols": result}
            
            return result
    
    async def get_token_prices(self, token_addresses: List[str]) -> Dict[str, Any]:
        """토큰 가격 조회 (DefiLlama)"""
        # ethereum: prefix 추가
        formatted_addresses = [f"ethereum:{addr}" if not addr.startswith("ethereum:") else addr for addr in token_addresses]
        addresses_str = ",".join(formatted_addresses)
        url = f"{self.defillama_url}/prices/current/{addresses_str}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            result = await self._make_request(session, 'GET', url)
            
            if 'error' in result:
                return result
            
            # 가격 데이터를 더 사용하기 쉬운 형태로 변환
            if 'coins' in result:
                return result['coins']
            
            return result
    
    # =============================================================================
    # 고수준 분석 메서드들
    # =============================================================================
    
    async def analyze_wallet_portfolio(self, wallet_address: str) -> Dict[str, Any]:
        """지갑 포트폴리오 종합 분석"""
        try:
            if CONFIG.DEBUG_MODE:
                logger.info(f"🔍 Analyzing portfolio for {wallet_address}")
            
            # 토큰 잔액 조회
            balances_result = await self.get_token_balances(wallet_address)
            
            if 'error' in balances_result:
                return balances_result
            
            portfolio = []
            token_addresses = []
            
            # 토큰 정보 수집
            for balance in balances_result.get('tokenBalances', []):
                if balance['tokenBalance'] and balance['tokenBalance'] != '0x0':
                    contract_address = balance['contractAddress']
                    token_addresses.append(contract_address)
                    
                    # 토큰 메타데이터 조회
                    metadata = await self.get_token_metadata(contract_address)
                    
                    if 'error' not in metadata:
                        token_balance = int(balance['tokenBalance'], 16)
                        decimals = metadata.get('decimals', 18)
                        actual_balance = token_balance / (10 ** decimals)
                        
                        if actual_balance > 0:
                            portfolio.append({
                                'contract_address': contract_address,
                                'name': metadata.get('name', 'Unknown'),
                                'symbol': metadata.get('symbol', 'UNK'),
                                'balance': actual_balance,
                                'decimals': decimals,
                                'price_usd': 0,  # 가격은 나중에 업데이트
                                'value_usd': 0
                            })
            
            # 토큰 가격 조회
            if token_addresses:
                prices = await self.get_token_prices(token_addresses)
                
                if 'error' not in prices:
                    # 포트폴리오에 가격 정보 추가
                    for token in portfolio:
                        price_key = f"ethereum:{token['contract_address']}"
                        if price_key in prices:
                            price_info = prices[price_key]
                            token['price_usd'] = price_info.get('price', 0)
                            token['value_usd'] = token['balance'] * token['price_usd']
            
            # 총 가치 계산
            total_value = sum(token.get('value_usd', 0) for token in portfolio)
            
            return {
                'wallet_address': wallet_address,
                'total_tokens': len(portfolio),
                'total_value_usd': total_value,
                'portfolio': portfolio,
                'timestamp': datetime.now().isoformat(),
                'analysis': {
                    'diversification_score': min(len(portfolio) * 10, 100),
                    'top_holding': max(portfolio, key=lambda x: x.get('value_usd', 0)) if portfolio else None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Portfolio analysis failed: {e}")
            return {'error': f'Portfolio analysis failed: {str(e)}'}
    
    async def get_defi_market_overview(self) -> Dict[str, Any]:
        """DeFi 시장 전체 개요"""
        try:
            if CONFIG.DEBUG_MODE:
                logger.info("📊 Getting DeFi market overview")
            
            # 체인별 TVL과 프로토콜 목록을 병렬로 조회
            chains_task = self.get_chains_tvl()
            protocols_task = self.get_protocols_list()
            
            chains_result, protocols_result = await asyncio.gather(chains_task, protocols_task)
            
            # 결과 처리
            overview = {
                'timestamp': datetime.now().isoformat(),
                'total_tvl_usd': 0,
                'top_chains': [],
                'top_protocols': []
            }
            
            # 체인 데이터 처리
            if 'error' not in chains_result:
                chains = chains_result.get('chains', [])
                if chains:
                    # 상위 10개 체인
                    top_chains = sorted(chains, key=lambda x: x.get('tvl', 0), reverse=True)[:10]
                    overview['top_chains'] = [
                        {
                            'name': chain.get('name'),
                            'tvl_usd': chain.get('tvl', 0),
                            'change_1d': chain.get('change_1d'),
                            'change_7d': chain.get('change_7d')
                        }
                        for chain in top_chains
                    ]
                    
                    # 전체 TVL 계산
                    overview['total_tvl_usd'] = sum(chain.get('tvl', 0) for chain in chains)
            
            # 프로토콜 데이터 처리
            if 'error' not in protocols_result:
                protocols = protocols_result.get('protocols', [])
                if protocols:
                    # 상위 10개 프로토콜
                    top_protocols = sorted(protocols, key=lambda x: x.get('tvl', 0), reverse=True)[:10]
                    overview['top_protocols'] = [
                        {
                            'name': protocol.get('name'),
                            'tvl_usd': protocol.get('tvl', 0),
                            'category': protocol.get('category'),
                            'change_1d': protocol.get('change_1d'),
                            'chains': protocol.get('chains', [])[:3]  # 상위 3개 체인만
                        }
                        for protocol in top_protocols
                    ]
            
            return overview
            
        except Exception as e:
            logger.error(f"❌ Market overview failed: {e}")
            return {'error': f'Market overview failed: {str(e)}'}

# 전역 API 클라이언트 인스턴스
api_client = BlockchainAPIClient()

# 테스트 함수
async def test_api_clients():
    """API 클라이언트 테스트"""
    print("🧪 Testing API Clients...")
    
    # DeFi 시장 개요 테스트
    print("\n📊 Testing DeFi market overview...")
    market_overview = await api_client.get_defi_market_overview()
    
    if 'error' in market_overview:
        print(f"❌ Market overview failed: {market_overview['error']}")
    else:
        print(f"✅ Total TVL: ${market_overview['total_tvl_usd']:,.0f}")
        print(f"✅ Top chains: {len(market_overview['top_chains'])}")
        print(f"✅ Top protocols: {len(market_overview['top_protocols'])}")
    
    # 프로토콜 TVL 테스트
    print("\n🦄 Testing Uniswap TVL...")
    uniswap_tvl = await api_client.get_protocol_tvl('uniswap')
    
    if 'error' in uniswap_tvl:
        print(f"❌ Uniswap TVL failed: {uniswap_tvl['error']}")
    else:
        print(f"✅ Uniswap data retrieved successfully")
    
    print("\n✅ API client tests completed")

if __name__ == "__main__":
    asyncio.run(test_api_clients())