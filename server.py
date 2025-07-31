"""
FastMCP Blockchain Analysis - Server
Qwen2.5-1.5B + FastMCP 서버 구현
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime

# FastMCP 라이브러리
from fastmcp import FastMCP

# 내부 모듈
from config import CONFIG
from api_clients import api_client
from utils import format_usd, format_percentage, truncate_address

# FastMCP 서버 애플리케이션 생성
app = FastMCP("Blockchain Analysis Server")

print("🚀 FastMCP Blockchain Analysis Server")
print(f"📡 Server: {app.name}")
print(f"🔧 Debug: {'On' if CONFIG.DEBUG_MODE else 'Off'}")

# =============================================================================
# 지갑 분석 도구들
# =============================================================================

@app.tool()
async def analyze_wallet_portfolio(wallet_address: str) -> Dict[str, Any]:
    """
    지갑의 토큰 포트폴리오를 분석합니다.
    
    Args:
        wallet_address: 분석할 지갑 주소 (0x로 시작하는 42자리 주소)
        
    Returns:
        포트폴리오 정보 (토큰 목록, 잔액, USD 가치 포함)
    """
    try:
        if CONFIG.DEBUG_MODE:
            print(f"🔍 Analyzing wallet portfolio: {truncate_address(wallet_address)}")
        
        # 지갑 주소 검증
        if not wallet_address.startswith('0x') or len(wallet_address) != 42:
            return {
                'error': 'Invalid wallet address format. Must be 42 characters starting with 0x',
                'wallet_address': wallet_address
            }
        
        # API 클라이언트를 통한 포트폴리오 분석
        result = await api_client.analyze_wallet_portfolio(wallet_address)
        
        if 'error' in result:
            return result
        
        # 결과 후처리 및 포맷팅
        portfolio = result.get('portfolio', [])
        total_value = result.get('total_value_usd', 0)
        
        # 상위 보유 토큰들 (가치 기준)
        top_holdings = sorted(portfolio, key=lambda x: x.get('value_usd', 0), reverse=True)[:5]
        
        # 분석 요약 생성
        analysis_summary = {
            'total_value': format_usd(total_value),
            'token_count': len(portfolio),
            'diversification': 'High' if len(portfolio) > 10 else 'Medium' if len(portfolio) > 5 else 'Low',
            'top_holdings': [
                {
                    'symbol': token['symbol'],
                    'name': token['name'],
                    'value': format_usd(token.get('value_usd', 0)),
                    'percentage': format_percentage(token.get('value_usd', 0) / total_value if total_value > 0 else 0)
                }
                for token in top_holdings[:3]
            ]
        }
        
        return {
            'wallet_address': wallet_address,
            'analysis_summary': analysis_summary,
            'detailed_portfolio': result,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'error': f'Wallet portfolio analysis failed: {str(e)}',
            'wallet_address': wallet_address
        }

@app.tool()
async def get_wallet_transaction_history(wallet_address: str, limit: int = 20) -> Dict[str, Any]:
    """
    지갑의 최근 거래 내역을 조회합니다.
    
    Args:
        wallet_address: 조회할 지갑 주소
        limit: 반환할 거래 수 (기본값: 20, 최대: 100)
        
    Returns:
        거래 내역 및 통계 정보
    """
    try:
        if CONFIG.DEBUG_MODE:
            print(f"📜 Getting transaction history: {truncate_address(wallet_address)}")
        
        # 지갑 주소 검증
        if not wallet_address.startswith('0x') or len(wallet_address) != 42:
            return {
                'error': 'Invalid wallet address format',
                'wallet_address': wallet_address
            }
        
        # 제한값 검증
        limit = min(max(1, limit), 100)
        
        # 자산 전송 기록 조회
        transfers_result = await api_client.get_asset_transfers(wallet_address, max_count=limit)
        
        if 'error' in transfers_result:
            return transfers_result
        
        transfers = transfers_result.get('transfers', [])
        
        # 거래 내역 처리 및 분류
        processed_transactions = []
        stats = {
            'total_transactions': len(transfers),
            'eth_transfers': 0,
            'token_transfers': 0,
            'nft_transfers': 0,
            'unique_counterparts': set()
        }
        
        for transfer in transfers:
            # 거래 분류
            category = transfer.get('category', 'unknown')
            if category in ['external', 'internal']:
                stats['eth_transfers'] += 1
            elif category == 'erc20':
                stats['token_transfers'] += 1
            elif category in ['erc721', 'erc1155']:
                stats['nft_transfers'] += 1
            
            # 상대방 주소 수집
            from_addr = transfer.get('from')
            to_addr = transfer.get('to')
            if from_addr and from_addr != wallet_address:
                stats['unique_counterparts'].add(from_addr)
            if to_addr and to_addr != wallet_address:
                stats['unique_counterparts'].add(to_addr)
            
            # 거래 정보 정리
            processed_transactions.append({
                'hash': transfer.get('hash'),
                'block_number': transfer.get('blockNum'),
                'from_address': truncate_address(transfer.get('from', '')),
                'to_address': truncate_address(transfer.get('to', '')),
                'value': transfer.get('value', 0),
                'asset': transfer.get('asset', 'ETH'),
                'category': category,
                'metadata': transfer.get('metadata', {})
            })
        
        stats['unique_counterparts'] = len(stats['unique_counterparts'])
        
        return {
            'wallet_address': wallet_address,
            'transaction_count': len(processed_transactions),
            'transactions': processed_transactions,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'error': f'Transaction history failed: {str(e)}',
            'wallet_address': wallet_address
        }

# =============================================================================
# DeFi 프로토콜 분석 도구들
# =============================================================================

@app.tool()
async def get_defi_protocol_info(protocol_name: str) -> Dict[str, Any]:
    """
    DeFi 프로토콜의 TVL 및 상세 정보를 조회합니다.
    
    Args:
        protocol_name: 프로토콜 이름 (예: uniswap, aave, compound)
        
    Returns:
        프로토콜 TVL, 체인별 분포, 기본 정보
    """
    try:
        if CONFIG.DEBUG_MODE:
            print(f"🦄 Getting DeFi protocol info: {protocol_name}")
        
        # 프로토콜 이름 정규화
        protocol_name = protocol_name.lower().strip()
        
        # 알려진 프로토콜 이름 매핑
        protocol_mapping = {
            'uni': 'uniswap',
            'uniswapv3': 'uniswap',
            'aave': 'aave',
            'comp': 'compound',
            'maker': 'makerdao',
            'sushi': 'sushiswap',
            'yearn': 'yearn-finance'
        }
        
        mapped_name = protocol_mapping.get(protocol_name, protocol_name)
        
        # 프로토콜 정보 조회
        protocol_data = await api_client.get_protocol_tvl(mapped_name)
        
        if 'error' in protocol_data:
            return {
                'error': f'Protocol "{protocol_name}" not found or API error',
                'available_protocols': list(CONFIG.DEFI_PROTOCOLS.keys()),
                'protocol_name': protocol_name
            }
        
        # TVL 데이터 처리
        current_tvl = 0
        tvl_history = protocol_data.get('tvl', [])
        if tvl_history:
            latest_tvl_entry = tvl_history[-1]
            current_tvl = latest_tvl_entry.get('totalLiquidityUSD', 0)
        
        # 체인별 TVL 분포
        chain_tvls = protocol_data.get('chainTvls', {})
        
        # 상위 체인들만 추출
        top_chains = []
        for chain, tvl_data in chain_tvls.items():
            if isinstance(tvl_data, list) and tvl_data:
                latest_chain_tvl = tvl_data[-1].get('totalLiquidityUSD', 0)
                if latest_chain_tvl > 0:
                    top_chains.append({
                        'chain': chain,
                        'tvl_usd': latest_chain_tvl,
                        'percentage': latest_chain_tvl / current_tvl * 100 if current_tvl > 0 else 0
                    })
        
        # TVL 기준으로 정렬
        top_chains = sorted(top_chains, key=lambda x: x['tvl_usd'], reverse=True)[:5]
        
        return {
            'protocol_name': protocol_data.get('name', protocol_name),
            'slug': mapped_name,
            'category': protocol_data.get('category', 'Unknown'),
            'current_tvl_usd': current_tvl,
            'formatted_tvl': format_usd(current_tvl),
            'chain_distribution': top_chains,
            'description': protocol_data.get('description', ''),
            'website': protocol_data.get('url', ''),
            'twitter': protocol_data.get('twitter', ''),
            'audit_links': protocol_data.get('audit_links', []),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'error': f'Protocol info failed: {str(e)}',
            'protocol_name': protocol_name
        }

@app.tool()
async def get_defi_market_overview() -> Dict[str, Any]:
    """
    DeFi 시장 전체 개요를 조회합니다.
    
    Returns:
        전체 TVL, 주요 체인별 TVL, 상위 프로토콜 정보
    """
    try:
        if CONFIG.DEBUG_MODE:
            print("📊 Getting DeFi market overview")
        
        # 시장 개요 데이터 조회
        overview = await api_client.get_defi_market_overview()
        
        if 'error' in overview:
            return overview
        
        total_tvl = overview.get('total_tvl_usd', 0)
        top_chains = overview.get('top_chains', [])
        top_protocols = overview.get('top_protocols', [])
        
        # 결과 포맷팅
        formatted_overview = {
            'market_summary': {
                'total_tvl': format_usd(total_tvl),
                'total_tvl_raw': total_tvl,
                'top_chains_count': len(top_chains),
                'top_protocols_count': len(top_protocols)
            },
            'top_chains': [
                {
                    'name': chain['name'],
                    'tvl': format_usd(chain.get('tvl_usd', 0)),
                    'tvl_raw': chain.get('tvl_usd', 0),
                    'change_24h': format_percentage(chain.get('change_1d', 0)),
                    'market_share': format_percentage(chain.get('tvl_usd', 0) / total_tvl if total_tvl > 0 else 0)
                }
                for chain in top_chains[:8]
            ],
            'top_protocols': [
                {
                    'name': protocol['name'],
                    'tvl': format_usd(protocol.get('tvl_usd', 0)),
                    'tvl_raw': protocol.get('tvl_usd', 0),
                    'category': protocol.get('category', 'Unknown'),
                    'change_24h': format_percentage(protocol.get('change_1d', 0)),
                    'chains': protocol.get('chains', []),
                    'market_share': format_percentage(protocol.get('tvl_usd', 0) / total_tvl if total_tvl > 0 else 0)
                }
                for protocol in top_protocols[:10]
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return formatted_overview
        
    except Exception as e:
        return {'error': f'Market overview failed: {str(e)}'}

# =============================================================================
# 고급 분석 도구들
# =============================================================================

@app.tool()
async def analyze_wallet_defi_exposure(wallet_address: str) -> Dict[str, Any]:
    """
    지갑의 DeFi 프로토콜 노출도를 분석합니다.
    
    Args:
        wallet_address: 분석할 지갑 주소
        
    Returns:
        DeFi 관련 토큰 보유량, 위험도 평가, 프로토콜별 노출도
    """
    try:
        if CONFIG.DEBUG_MODE:
            print(f"🎯 Analyzing DeFi exposure: {truncate_address(wallet_address)}")
        
        # 먼저 지갑 포트폴리오 조회
        portfolio_result = await analyze_wallet_portfolio(wallet_address)
        
        if 'error' in portfolio_result:
            return portfolio_result
        
        portfolio = portfolio_result.get('detailed_portfolio', {}).get('portfolio', [])
        total_value = portfolio_result.get('detailed_portfolio', {}).get('total_value_usd', 0)
        
        # DeFi 관련 토큰 식별
        defi_tokens = {
            'UNI': {'protocol': 'Uniswap', 'category': 'DEX'},
            'AAVE': {'protocol': 'Aave', 'category': 'Lending'},
            'COMP': {'protocol': 'Compound', 'category': 'Lending'},
            'MKR': {'protocol': 'MakerDAO', 'category': 'CDP'},
            'SNX': {'protocol': 'Synthetix', 'category': 'Derivatives'},
            'YFI': {'protocol': 'Yearn Finance', 'category': 'Yield'},
            'SUSHI': {'protocol': 'SushiSwap', 'category': 'DEX'},
            'CRV': {'protocol': 'Curve', 'category': 'DEX'},
            'BAL': {'protocol': 'Balancer', 'category': 'DEX'},
            'USDC': {'protocol': 'Various', 'category': 'Stablecoin'},
            'DAI': {'protocol': 'MakerDAO', 'category': 'Stablecoin'},
            'USDT': {'protocol': 'Various', 'category': 'Stablecoin'}
        }
        
        defi_positions = []
        total_defi_value = 0
        category_exposure = {}
        
        for token in portfolio:
            symbol = token.get('symbol', '').upper()
            if symbol in defi_tokens:
                defi_info = defi_tokens[symbol]
                token_value = token.get('value_usd', 0)
                
                position = {
                    'token': {
                        'symbol': symbol,
                        'name': token.get('name'),
                        'balance': token.get('balance'),
                        'value_usd': token_value,
                        'formatted_value': format_usd(token_value)
                    },
                    'protocol': defi_info['protocol'],
                    'category': defi_info['category']
                }
                
                defi_positions.append(position)
                total_defi_value += token_value
                
                # 카테고리별 노출도 계산
                category = defi_info['category']
                if category not in category_exposure:
                    category_exposure[category] = {'value': 0, 'tokens': []}
                category_exposure[category]['value'] += token_value
                category_exposure[category]['tokens'].append(symbol)
        
        # 노출도 분석
        defi_exposure_ratio = total_defi_value / total_value if total_value > 0 else 0
        
        # 위험도 평가
        risk_factors = []
        risk_level = 'Low'
        
        if defi_exposure_ratio > 0.7:
            risk_level = 'High'
            risk_factors.append('High DeFi concentration (>70%)')
        elif defi_exposure_ratio > 0.3:
            risk_level = 'Medium'
            risk_factors.append('Moderate DeFi exposure (30-70%)')
        
        if len(category_exposure) < 2 and defi_exposure_ratio > 0.2:
            risk_factors.append('Low diversification across DeFi categories')
        
        # 스테이블코인 비율 확인
        stablecoin_value = category_exposure.get('Stablecoin', {}).get('value', 0)
        stablecoin_ratio = stablecoin_value / total_defi_value if total_defi_value > 0 else 0
        
        if stablecoin_ratio < 0.1 and defi_exposure_ratio > 0.5:
            risk_factors.append('Low stablecoin allocation for high DeFi exposure')
        
        return {
            'wallet_address': wallet_address,
            'exposure_summary': {
                'total_portfolio_value': format_usd(total_value),
                'total_defi_value': format_usd(total_defi_value),
                'defi_exposure_ratio': format_percentage(defi_exposure_ratio),
                'risk_level': risk_level,
                'risk_factors': risk_factors
            },
            'category_breakdown': {
                category: {
                    'value': format_usd(data['value']),
                    'percentage': format_percentage(data['value'] / total_defi_value if total_defi_value > 0 else 0),
                    'tokens': data['tokens']
                }
                for category, data in category_exposure.items()
            },
            'defi_positions': defi_positions,
            'recommendations': _generate_defi_recommendations(defi_exposure_ratio, category_exposure, risk_level),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'error': f'DeFi exposure analysis failed: {str(e)}',
            'wallet_address': wallet_address
        }

def _generate_defi_recommendations(exposure_ratio: float, categories: Dict, risk_level: str) -> List[str]:
    """DeFi 노출도 기반 추천사항 생성"""
    recommendations = []
    
    if risk_level == 'High':
        recommendations.append("Consider reducing DeFi exposure to below 50% of portfolio")
        recommendations.append("Increase allocation to blue-chip assets (ETH, BTC)")
    
    if len(categories) == 1:
        recommendations.append("Diversify across different DeFi categories (Lending, DEX, Yield)")
    
    stablecoin_value = categories.get('Stablecoin', {}).get('value', 0)
    if stablecoin_value == 0 and exposure_ratio > 0.3:
        recommendations.append("Consider adding stablecoins for reduced volatility")
    
    if 'Lending' not in categories and exposure_ratio > 0.2:
        recommendations.append("Consider lending protocols (AAVE, Compound) for yield generation")
    
    return recommendations

# =============================================================================
# 비교 분석 도구들
# =============================================================================

@app.tool()
async def compare_defi_protocols(protocol1: str, protocol2: str) -> Dict[str, Any]:
    """
    두 DeFi 프로토콜을 비교 분석합니다.
    
    Args:
        protocol1: 첫 번째 프로토콜 이름
        protocol2: 두 번째 프로토콜 이름
        
    Returns:
        두 프로토콜의 TVL, 카테고리, 체인 분포 비교
    """
    try:
        if CONFIG.DEBUG_MODE:
            print(f"⚖️ Comparing protocols: {protocol1} vs {protocol2}")
        
        # 두 프로토콜 정보를 병렬로 조회
        protocol1_task = get_defi_protocol_info(protocol1)
        protocol2_task = get_defi_protocol_info(protocol2)
        
        result1, result2 = await asyncio.gather(protocol1_task, protocol2_task)
        
        if 'error' in result1:
            return {'error': f'Failed to get info for {protocol1}: {result1["error"]}'}
        if 'error' in result2:
            return {'error': f'Failed to get info for {protocol2}: {result2["error"]}'}
        
        # 비교 분석
        tvl1 = result1.get('current_tvl_usd', 0)
        tvl2 = result2.get('current_tvl_usd', 0)
        
        comparison = {
            'protocols': {
                'protocol1': {
                    'name': result1.get('protocol_name'),
                    'tvl': format_usd(tvl1),
                    'category': result1.get('category'),
                    'chains': len(result1.get('chain_distribution', []))
                },
                'protocol2': {
                    'name': result2.get('protocol_name'),
                    'tvl': format_usd(tvl2),
                    'category': result2.get('category'),
                    'chains': len(result2.get('chain_distribution', []))
                }
            },
            'comparison': {
                'tvl_leader': result1.get('protocol_name') if tvl1 > tvl2 else result2.get('protocol_name'),
                'tvl_difference': format_usd(abs(tvl1 - tvl2)),
                'tvl_ratio': f"{max(tvl1, tvl2) / min(tvl1, tvl2):.2f}x" if min(tvl1, tvl2) > 0 else "N/A",
                'same_category': result1.get('category') == result2.get('category'),
                'chain_diversity_leader': result1.get('protocol_name') if len(result1.get('chain_distribution', [])) > len(result2.get('chain_distribution', [])) else result2.get('protocol_name')
            },
            'detailed_comparison': {
                'protocol1_details': result1,
                'protocol2_details': result2
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return comparison
        
    except Exception as e:
        return {'error': f'Protocol comparison failed: {str(e)}'}

# =============================================================================
# 서버 실행 설정
# =============================================================================

@app.tool()
async def get_server_status() -> Dict[str, Any]:
    """
    서버 상태 및 설정 정보를 반환합니다.
    
    Returns:
        서버 상태, 사용 가능한 도구, 설정 정보
    """
    return {
        'server_name': app.name,
        'status': 'healthy',
        'available_tools': [
            'analyze_wallet_portfolio',
            'get_wallet_transaction_history', 
            'get_defi_protocol_info',
            'get_defi_market_overview',
            'analyze_wallet_defi_exposure',
            'compare_defi_protocols'
        ],
        'configuration': {
            'debug_mode': CONFIG.DEBUG_MODE,
            'alchemy_api': 'configured' if CONFIG.ALCHEMY_API_KEY else 'missing',
            'model_name': CONFIG.MODEL_NAME,
            'cache_dir': CONFIG.CACHE_DIR
        },
        'demo_wallet': CONFIG.DEMO_WALLET,
        'timestamp': datetime.now().isoformat()
    }

# 서버 시작 시 설정 검증
if __name__ == "__main__":
    print("\n🔧 Validating configuration...")
    if CONFIG.validate():
        print("✅ Configuration valid, starting server...")
        app.run()
    else:
        print("❌ Configuration invalid, please check your settings")
        exit(1)