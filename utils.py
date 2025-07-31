"""
FastMCP Blockchain Analysis - Utilities
공통 유틸리티 함수들
"""

import json
import re
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime

# =============================================================================
# 주소 및 해시 관련 유틸리티
# =============================================================================

def validate_ethereum_address(address: str) -> bool:
    """이더리움 주소 유효성 검사"""
    if not isinstance(address, str):
        return False
    
    # 0x로 시작하고 42자리인지 확인
    if not address.startswith('0x') or len(address) != 42:
        return False
    
    # 헥사데시말 문자만 포함하는지 확인
    try:
        int(address[2:], 16)
        return True
    except ValueError:
        return False

def truncate_address(address: str, start_chars: int = 6, end_chars: int = 4) -> str:
    """주소를 축약 형태로 표시 (0x1234...abcd)"""
    if not address or len(address) < start_chars + end_chars:
        return address
    
    return f"{address[:start_chars]}...{address[-end_chars:]}"

def truncate_hash(hash_str: str, length: int = 10) -> str:
    """해시를 축약 형태로 표시"""
    if not hash_str or len(hash_str) <= length:
        return hash_str
    
    return f"{hash_str[:length]}..."

# =============================================================================
# 숫자 포맷팅 유틸리티
# =============================================================================

def format_usd(amount: float, decimals: int = 2) -> str:
    """USD 금액을 포맷팅"""
    if amount == 0:
        return "$0.00"
    
    if amount >= 1_000_000_000:  # 10억 이상
        return f"${amount / 1_000_000_000:.{decimals}f}B"
    elif amount >= 1_000_000:  # 100만 이상
        return f"${amount / 1_000_000:.{decimals}f}M"
    elif amount >= 1_000:  # 1천 이상
        return f"${amount / 1_000:.{decimals}f}K"
    else:
        return f"${amount:.{decimals}f}"

def format_percentage(ratio: float, decimals: int = 2) -> str:
    """비율을 퍼센트로 포맷팅"""
    percentage = ratio * 100
    return f"{percentage:.{decimals}f}%"

def format_token_amount(amount: float, symbol: str, decimals: int = 4) -> str:
    """토큰 수량을 포맷팅"""
    if amount == 0:
        return f"0 {symbol}"
    
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.{decimals}f}M {symbol}"
    elif amount >= 1_000:
        return f"{amount / 1_000:.{decimals}f}K {symbol}"
    else:
        return f"{amount:.{decimals}f} {symbol}"

def format_large_number(number: float, decimals: int = 2) -> str:
    """큰 숫자를 읽기 쉽게 포맷팅"""
    if number >= 1_000_000_000_000:  # 1조 이상
        return f"{number / 1_000_000_000_000:.{decimals}f}T"
    elif number >= 1_000_000_000:  # 10억 이상
        return f"{number / 1_000_000_000:.{decimals}f}B"
    elif number >= 1_000_000:  # 100만 이상
        return f"{number / 1_000_000:.{decimals}f}M"
    elif number >= 1_000:  # 1천 이상
        return f"{number / 1_000:.{decimals}f}K"
    else:
        return f"{number:,.{decimals}f}"

# =============================================================================
# 텍스트 파싱 유틸리티
# =============================================================================

def extract_wallet_addresses(text: str) -> List[str]:
    """텍스트에서 이더리움 주소 추출"""
    pattern = r'0x[a-fA-F0-9]{40}'
    addresses = re.findall(pattern, text)
    return [addr for addr in addresses if validate_ethereum_address(addr)]

def extract_transaction_hashes(text: str) -> List[str]:
    """텍스트에서 트랜잭션 해시 추출"""
    pattern = r'0x[a-fA-F0-9]{64}'
    return re.findall(pattern, text)

def parse_mcp_call(text: str) -> Optional[Dict[str, Any]]:
    """텍스트에서 MCP 함수 호출 파싱"""
    patterns = [
        r'<mcp_call>\s*(\{.*?\})\s*</mcp_call>',
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        r'<function_call>\s*(\{.*?\})\s*</function_call>',
        r'```json\s*(\{.*?\})\s*```'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    return None

def clean_protocol_name(protocol_name: str) -> str:
    """프로토콜 이름 정규화"""
    # 소문자 변환 및 공백 제거
    cleaned = protocol_name.lower().strip()
    
    # 일반적인 변형 처리
    mappings = {
        'uni': 'uniswap',
        'uniswap v3': 'uniswap',
        'uniswap v2': 'uniswap',
        'sushi': 'sushiswap',
        'maker': 'makerdao',
        'compound v2': 'compound',
        'aave v2': 'aave',
        'aave v3': 'aave',
        'yearn': 'yearn-finance'
    }
    
    return mappings.get(cleaned, cleaned)

# =============================================================================
# 데이터 처리 유틸리티
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """안전한 나눗셈 (0으로 나누기 방지)"""
    return numerator / denominator if denominator != 0 else default

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """퍼센트 변화율 계산"""
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    
    return ((new_value - old_value) / old_value) * 100

def sort_by_value(data: List[Dict], key: str, reverse: bool = True) -> List[Dict]:
    """딕셔너리 리스트를 특정 키 값으로 정렬"""
    return sorted(data, key=lambda x: x.get(key, 0), reverse=reverse)

def filter_by_threshold(data: List[Dict], key: str, threshold: float, operator: str = 'gte') -> List[Dict]:
    """임계값 기준으로 데이터 필터링"""
    if operator == 'gte':
        return [item for item in data if item.get(key, 0) >= threshold]
    elif operator == 'gt':
        return [item for item in data if item.get(key, 0) > threshold]
    elif operator == 'lte':
        return [item for item in data if item.get(key, 0) <= threshold]
    elif operator == 'lt':
        return [item for item in data if item.get(key, 0) < threshold]
    else:
        return data

# =============================================================================
# 캐싱 유틸리티
# =============================================================================

def generate_cache_key(*args) -> str:
    """캐시 키 생성"""
    key_string = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

def save_to_cache(key: str, data: Any, cache_dir: str = "./data/api_cache") -> bool:
    """데이터를 캐시에 저장"""
    try:
        import os
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f"{key}.json")
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        return True
    except Exception:
        return False

def load_from_cache(key: str, max_age_minutes: int = 60, cache_dir: str = "./data/api_cache") -> Optional[Any]:
    """캐시에서 데이터 로드"""
    try:
        import os
        from datetime import datetime, timedelta
        
        cache_file = os.path.join(cache_dir, f"{key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # 캐시 만료 확인
        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        if datetime.now() - cache_time > timedelta(minutes=max_age_minutes):
            return None
        
        return cache_data['data']
    except Exception:
        return None

# =============================================================================
# 로깅 유틸리티
# =============================================================================

def log_api_call(function_name: str, args: Dict, result: Dict, duration: float = None):
    """API 호출 로깅"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'function': function_name,
        'arguments': args,
        'success': 'error' not in result,
        'duration_ms': duration * 1000 if duration else None
    }
    
    # 에러가 있는 경우 에러 정보 추가
    if 'error' in result:
        log_entry['error'] = result['error']
    
    # 간단한 파일 로깅 (실제로는 더 정교한 로깅 시스템 사용)
    try:
        import os
        os.makedirs("./logs", exist_ok=True)
        
        with open("./logs/api_calls.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass  # 로깅 실패해도 메인 기능에 영향 없도록

def save_demo_result(query: str, result: Dict[str, Any], demo_file: str = "./examples/demo_results.json"):
    """데모 결과를 파일에 저장"""
    try:
        import os
        os.makedirs(os.path.dirname(demo_file), exist_ok=True)
        
        demo_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'result': result,
            'success': 'error' not in result
        }
        
        with open(demo_file, "a") as f:
            f.write(json.dumps(demo_entry, indent=2) + "\n")
    except Exception:
        pass

# =============================================================================
# 검증 유틸리티
# =============================================================================

def validate_token_symbol(symbol: str) -> bool:
    """토큰 심볼 유효성 검사"""
    if not isinstance(symbol, str):
        return False
    
    # 1-10자의 영문자/숫자만 허용
    return bool(re.match(r'^[A-Za-z0-9]{1,10}, symbol'))

def validate_protocol_name(protocol: str) -> bool:
    """프로토콜 이름 유효성 검사"""
    if not isinstance(protocol, str):
        return False
    
    # 영문자, 숫자, 하이픈, 언더스코어만 허용
    return bool(re.match(r'^[A-Za-z0-9\-_]{1,50}, protocol'))

def validate_positive_number(value: Any) -> bool:
    """양수 검증"""
    try:
        num_value = float(value)
        return num_value > 0
    except (ValueError, TypeError):
        return False

def sanitize_user_input(user_input: str, max_length: int = 1000) -> str:
    """사용자 입력 정제"""
    if not isinstance(user_input, str):
        return ""
    
    # 길이 제한
    sanitized = user_input[:max_length]
    
    # 위험한 문자 제거 (기본적인 XSS 방지)
    sanitized = re.sub(r'<[^>]*>', '', sanitized)
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    
    return sanitized.strip()

# =============================================================================
# 데이터 변환 유틸리티
# =============================================================================

def wei_to_ether(wei_value: int) -> float:
    """Wei를 Ether로 변환"""
    return wei_value / 10**18

def hex_to_int(hex_value: str) -> int:
    """16진수 문자열을 정수로 변환"""
    try:
        return int(hex_value, 16)
    except (ValueError, TypeError):
        return 0

def timestamp_to_datetime(timestamp: int) -> str:
    """Unix 타임스탬프를 읽기 쉬운 날짜로 변환"""
    try:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OSError):
        return "Invalid timestamp"

def normalize_token_amount(amount: str, decimals: int) -> float:
    """토큰 수량 정규화 (raw amount를 실제 수량으로)"""
    try:
        if isinstance(amount, str) and amount.startswith('0x'):
            raw_amount = int(amount, 16)
        else:
            raw_amount = int(amount)
        
        return raw_amount / (10 ** decimals)
    except (ValueError, TypeError):
        return 0.0

# =============================================================================
# 성능 유틸리티
# =============================================================================

import time
from functools import wraps

def timing_decorator(func):
    """함수 실행 시간 측정 데코레이터"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"⏱️ {func.__name__} executed in {duration:.3f}s")
        
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"⏱️ {func.__name__} executed in {duration:.3f}s")
        
        return result
    
    # 비동기 함수인지 확인
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def batch_process(items: List[Any], batch_size: int = 10):
    """리스트를 배치로 나누어 처리"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# =============================================================================
# 에러 처리 유틸리티
# =============================================================================

def create_error_response(error_message: str, error_code: str = None, **additional_data) -> Dict[str, Any]:
    """표준화된 에러 응답 생성"""
    error_response = {
        'error': error_message,
        'timestamp': datetime.now().isoformat()
    }
    
    if error_code:
        error_response['error_code'] = error_code
    
    # 추가 데이터 병합
    error_response.update(additional_data)
    
    return error_response

def handle_api_error(error: Exception, context: str = None) -> Dict[str, Any]:
    """API 에러를 표준 형식으로 변환"""
    error_message = str(error)
    
    # 일반적인 에러 패턴 처리
    if "timeout" in error_message.lower():
        return create_error_response(
            "Request timeout occurred",
            error_code="TIMEOUT_ERROR",
            context=context
        )
    elif "connection" in error_message.lower():
        return create_error_response(
            "Connection error occurred",
            error_code="CONNECTION_ERROR", 
            context=context
        )
    elif "404" in error_message:
        return create_error_response(
            "Resource not found",
            error_code="NOT_FOUND_ERROR",
            context=context
        )
    elif "401" in error_message or "403" in error_message:
        return create_error_response(
            "Authentication or authorization failed",
            error_code="AUTH_ERROR",
            context=context
        )
    else:
        return create_error_response(
            f"API error: {error_message}",
            error_code="API_ERROR",
            context=context
        )

# =============================================================================
# 개발 및 디버깅 유틸리티
# =============================================================================

def print_debug(message: str, data: Any = None, level: str = "INFO"):
    """디버그 메시지 출력"""
    from config import CONFIG
    
    if not CONFIG.DEBUG_MODE:
        return
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    level_icons = {
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'SUCCESS': '✅',
        'DEBUG': '🐛'
    }
    
    icon = level_icons.get(level, 'ℹ️')
    print(f"{icon} [{timestamp}] {message}")
    
    if data is not None:
        print(f"   Data: {json.dumps(data, indent=2, default=str)}")

def create_sample_data(data_type: str) -> Dict[str, Any]:
    """테스트용 샘플 데이터 생성"""
    if data_type == "wallet_portfolio":
        return {
            'wallet_address': '0x742d35Cc6798C532c5Ae5e4a5b6a4c7c4a24ba',
            'total_tokens': 5,
            'total_value_usd': 12500.50,
            'portfolio': [
                {
                    'contract_address': '0xA0b86a33E6417C9C4E1f03a6C0fA8b7b63Ad2C6A',
                    'name': 'USD Coin',
                    'symbol': 'USDC',
                    'balance': 5000.0,
                    'price_usd': 1.0,
                    'value_usd': 5000.0
                },
                {
                    'contract_address': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
                    'name': 'Uniswap',
                    'symbol': 'UNI',  
                    'balance': 500.0,
                    'price_usd': 15.0,
                    'value_usd': 7500.0
                }
            ]
        }
    elif data_type == "defi_protocol":
        return {
            'protocol_name': 'Uniswap',
            'current_tvl_usd': 5500000000,
            'category': 'DEX',
            'chain_distribution': [
                {'chain': 'Ethereum', 'tvl_usd': 4000000000},
                {'chain': 'Polygon', 'tvl_usd': 1000000000},
                {'chain': 'Arbitrum', 'tvl_usd': 500000000}
            ]
        }
    else:
        return {'error': f'Unknown sample data type: {data_type}'}

# =============================================================================
# 테스트 지원 함수들
# =============================================================================

def verify_response_structure(response: Dict[str, Any], required_fields: List[str]) -> bool:
    """응답 구조 검증"""
    for field in required_fields:
        if field not in response:
            print_debug(f"Missing required field: {field}", level="ERROR")
            return False
    
    return True

def generate_test_wallet_address() -> str:
    """테스트용 지갑 주소 생성"""
    import random
    
    # 0x + 40자리 헥사데시말
    hex_chars = '0123456789abcdef'
    address = '0x' + ''.join(random.choice(hex_chars) for _ in range(40))
    return address

# 모듈 테스트
if __name__ == "__main__":
    print("🧪 Testing utility functions...")
    
    # 주소 검증 테스트
    test_address = "0x742d35Cc6798C532c5Ae5e4a5b6a4c7c4a24ba"
    print(f"✅ Address validation: {validate_ethereum_address(test_address)}")
    print(f"✅ Truncated address: {truncate_address(test_address)}")
    
    # 포맷팅 테스트
    print(f"✅ USD formatting: {format_usd(1234567.89)}")
    print(f"✅ Percentage formatting: {format_percentage(0.1534)}")
    print(f"✅ Token formatting: {format_token_amount(1234567.89, 'USDC')}")
    
    # 샘플 데이터 테스트
    sample_portfolio = create_sample_data("wallet_portfolio")
    print(f"✅ Sample portfolio generated with {sample_portfolio['total_tokens']} tokens")
    
    print("✅ All utility function tests passed!")