"""
FastMCP Blockchain Analysis - Qwen Client
Qwen2.5-1.5B 모델과 MCP 서버를 연결하는 클라이언트
"""

import asyncio
import json
import re
import torch
import subprocess
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Transformers 라이브러리
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from huggingface_hub import login

# 내부 모듈
from config import CONFIG
from utils import print_debug, parse_mcp_call, save_demo_result, timing_decorator

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class QwenModelManager:
    """Qwen2.5-1.5B 모델 관리자"""
    
    def __init__(self, model_name: str = None, cache_dir: str = None):
        self.model_name = model_name or CONFIG.MODEL_NAME
        self.cache_dir = cache_dir or CONFIG.CACHE_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.tokenizer = None
        
        print_debug(f"Initializing Qwen Model Manager", {
            'model': self.model_name,
            'device': self.device,
            'cache_dir': self.cache_dir
        })
    
    async def load_model(self):
        """모델 로드 (비동기)"""
        print("🤖 Loading Qwen2.5-1.5B model...")
        
        try:
            # HuggingFace 토큰 설정
            if CONFIG.HUGGINGFACE_TOKEN:
                login(token=CONFIG.HUGGINGFACE_TOKEN)
                print_debug("HuggingFace token configured")
            
            # 토크나이저 로드
            print("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                token=CONFIG.HUGGINGFACE_TOKEN if CONFIG.HUGGINGFACE_TOKEN else None
            )
            
            # 모델 로드 설정
            print("🧠 Loading model...")
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if CONFIG.HUGGINGFACE_TOKEN:
                model_kwargs["token"] = CONFIG.HUGGINGFACE_TOKEN
            
            # GPU 설정
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["low_cpu_mem_usage"] = True
                print("🚀 Using GPU acceleration")
            else:
                print("🐌 Using CPU (consider GPU for better performance)")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # CPU로 명시적 이동 (필요한 경우)
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Qwen2.5-1.5B model loaded successfully!")
            print(f"📊 Model size: ~1.5B parameters")
            print(f"💾 Memory usage: ~{self._estimate_memory_usage():.1f}GB")
            
        except Exception as e:
            print(f"❌ Error loading Qwen model: {e}")
            raise
    
    def _estimate_memory_usage(self) -> float:
        """메모리 사용량 추정"""
        if self.model is None:
            return 0.0
        
        param_count = sum(p.numel() for p in self.model.parameters())
        bytes_per_param = 2 if self.device == "cuda" else 4  # fp16 vs fp32
        return (param_count * bytes_per_param) / (1024**3)  # GB
    
    def build_system_prompt(self, available_tools: List[Dict]) -> str:
        """시스템 프롬프트 생성"""
        tools_desc = "\n".join([
            f"- {tool['name']}: {tool.get('description', 'No description available')}"
            for tool in available_tools
        ])
        
        return f"""You are an AI assistant specialized in blockchain and DeFi analysis. You have access to powerful tools via MCP (Model Context Protocol).

Available MCP Tools:
{tools_desc}

IMPORTANT: When you need to use a tool, respond with this EXACT format:
<mcp_call>
{{"tool": "tool_name", "arguments": {{"param": "value"}}}}
</mcp_call>

Guidelines:
1. Use tools when you need real-time blockchain data or DeFi information
2. Provide helpful analysis and insights based on the tool results
3. If you don't need tools, respond normally with your knowledge
4. Always be accurate with wallet addresses (42 characters starting with 0x)
5. Explain complex DeFi concepts in simple terms
6. Format large numbers in readable format (e.g., $1.5M instead of $1500000)

Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    @timing_decorator
    async def generate_response(self, prompt: str) -> str:
        """응답 생성"""
        try:
            if not self.model or not self.tokenizer:
                return "Error: Model not loaded"
            
            print_debug("Generating response", {"prompt_length": len(prompt)})
            
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,  # 1.5B 모델에서 더 긴 컨텍스트 지원
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 생성 설정
            generation_config = GenerationConfig(
                max_new_tokens=CONFIG.MAX_NEW_TOKENS,
                temperature=CONFIG.TEMPERATURE,
                top_p=CONFIG.TOP_P,
                top_k=CONFIG.TOP_K,
                do_sample=True,
                repetition_penalty=CONFIG.REPETITION_PENALTY,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3  # 반복 방지
            )
            
            # 텍스트 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    use_cache=True
                )
            
            # 디코딩 (입력 제외하고 생성된 부분만)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            print_debug("Response generated", {"response_length": len(generated_text)})
            
            return generated_text
            
        except Exception as e:
            print_debug(f"Generation error: {e}", level="ERROR")
            return f"Error generating response: {str(e)}"
    
    def clear_cache(self):
        """메모리 캐시 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_debug("Memory cache cleared")

class MCPConnector:
    """MCP 서버 연결 관리자"""
    
    def __init__(self, server_command: str = "python server.py"):
        self.server_command = server_command
        self.server_process = None
        self.available_tools = []
        self.is_connected = False
        
    async def start_mcp_server(self):
        """MCP 서버 시작"""
        try:
            print("🔧 Starting MCP server...")
            
            # 서버 프로세스 시작
            self.server_process = subprocess.Popen(
                self.server_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 서버 시작 대기
            await asyncio.sleep(3)
            
            # 서버 상태 확인
            if self.server_process.poll() is None:
                print("✅ MCP server started successfully")
                self.is_connected = True
                
                # 사용 가능한 도구 목록 설정
                self.available_tools = [
                    {
                        'name': 'analyze_wallet_portfolio',
                        'description': 'Analyze wallet token portfolio with USD values'
                    },
                    {
                        'name': 'get_wallet_transaction_history', 
                        'description': 'Get recent transaction history for a wallet'
                    },
                    {
                        'name': 'get_defi_protocol_info',
                        'description': 'Get DeFi protocol TVL and information'
                    },
                    {
                        'name': 'get_defi_market_overview',
                        'description': 'Get overall DeFi market overview and statistics'
                    },
                    {
                        'name': 'analyze_wallet_defi_exposure',
                        'description': 'Analyze wallet DeFi protocol exposure and risk'
                    },
                    {
                        'name': 'compare_defi_protocols',
                        'description': 'Compare two DeFi protocols by TVL and features'
                    }
                ]
                
                print(f"🔧 Available tools: {len(self.available_tools)}")
                for tool in self.available_tools:
                    print(f"  • {tool['name']}")
                
            else:
                raise Exception("Server failed to start")
                
        except Exception as e:
            print(f"❌ Failed to start MCP server: {e}")
            raise
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """MCP 도구 호출 (시뮬레이션)"""
        if not self.is_connected:
            return {'error': 'MCP server not connected'}
        
        print_debug(f"Calling MCP tool: {tool_name}", arguments)
        
        # 실제 구현에서는 MCP 프로토콜을 통해 서버와 통신
        # 여기서는 시뮬레이션을 위해 직접 import
        try:
            from server import (
                analyze_wallet_portfolio,
                get_wallet_transaction_history,
                get_defi_protocol_info,
                get_defi_market_overview,
                analyze_wallet_defi_exposure,
                compare_defi_protocols
            )
            
            # 함수 매핑
            tool_functions = {
                'analyze_wallet_portfolio': analyze_wallet_portfolio,
                'get_wallet_transaction_history': get_wallet_transaction_history,
                'get_defi_protocol_info': get_defi_protocol_info,
                'get_defi_market_overview': get_defi_market_overview,
                'analyze_wallet_defi_exposure': analyze_wallet_defi_exposure,
                'compare_defi_protocols': compare_defi_protocols
            }
            
            if tool_name in tool_functions:
                result = await tool_functions[tool_name](**arguments)
                print_debug("Tool execution successful", {"tool": tool_name})
                return result
            else:
                return {'error': f'Unknown tool: {tool_name}'}
                
        except Exception as e:
            print_debug(f"Tool execution failed: {e}", level="ERROR")
            return {'error': f'Tool execution failed: {str(e)}'}
    
    def stop_server(self):
        """MCP 서버 중지"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("🛑 MCP server stopped")

class QwenMcpClient:
    """Qwen MCP 통합 클라이언트"""
    
    def __init__(self):
        self.model_manager = QwenModelManager()
        self.mcp_connector = MCPConnector()
        self.conversation_history = []
        
        print("🚀 Qwen MCP Client initialized")
    
    async def initialize(self):
        """클라이언트 초기화"""
        print("🔧 Initializing Qwen MCP Client...")
        
        # 모델 로드와 MCP 서버 시작을 병렬로 실행
        await asyncio.gather(
            self.model_manager.load_model(),
            self.mcp_connector.start_mcp_server()
        )
        
        print("✅ Qwen MCP Client ready!")
    
    def _build_conversation_prompt(self) -> str:
        """대화 프롬프트 구성"""
        system_prompt = self.model_manager.build_system_prompt(self.mcp_connector.available_tools)
        prompt = system_prompt + "\n\nConversation:\n"
        
        for msg in self.conversation_history:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                prompt += f"Human: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "tool_result":
                prompt += f"Tool Result: {content}\n"
        
        prompt += "Assistant: "
        return prompt
    
    async def chat(self, user_message: str, max_iterations: int = 3) -> str:
        """사용자와 채팅"""
        
        # 사용자 메시지를 대화 기록에 추가
        self.conversation_history.append({"role": "user", "content": user_message})
        print_debug(f"User message: {user_message}")
        
        iteration = 0
        
        while iteration < max_iterations:
            # 현재 대화 기록으로 프롬프트 구성
            prompt = self._build_conversation_prompt()
            
            # Qwen 모델에서 응답 생성
            response = await self.model_manager.generate_response(prompt)
            
            print_debug(f"Qwen response (iteration {iteration + 1})", {
                "response_preview": response[:100] + "..." if len(response) > 100 else response
            })
            
            # MCP 도구 호출 파싱
            tool_call = parse_mcp_call(response)
            
            if tool_call and 'tool' in tool_call:
                # MCP 도구 실행
                tool_name = tool_call['tool']
                arguments = tool_call.get('arguments', {})
                
                print(f"🔧 Executing tool: {tool_name}")
                tool_result = await self.mcp_connector.call_tool(tool_name, arguments)
                
                # 도구 결과를 대화 기록에 추가
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": f"I'll use the {tool_name} tool to get this information."
                })
                self.conversation_history.append({
                    "role": "tool_result", 
                    "content": json.dumps(tool_result, indent=2)
                })
                
                iteration += 1
            else:
                # 최종 응답
                self.conversation_history.append({"role": "assistant", "content": response})
                return response
        
        return "I've reached the maximum number of tool calls. Please try a simpler query."
    
    def clear_conversation(self):
        """대화 기록 초기화"""
        self.conversation_history = []
        print_debug("Conversation history cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """클라이언트 통계 정보"""
        return {
            "conversation_messages": len(self.conversation_history),
            "tool_calls_made": len([msg for msg in self.conversation_history if msg["role"] == "tool_result"]),
            "model_info": {
                "name": self.model_manager.model_name,
                "device": self.model_manager.device,
                "memory_usage_gb": self.model_manager._estimate_memory_usage()
            },
            "mcp_status": {
                "connected": self.mcp_connector.is_connected,
                "available_tools": len(self.mcp_connector.available_tools)
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        print("🧹 Cleaning up resources...")
        
        # MCP 서버 종료
        self.mcp_connector.stop_server()
        
        # 메모리 정리
        self.model_manager.clear_cache()
        
        print("✅ Cleanup completed")

class ChatInterface:
    """채팅 인터페이스"""
    
    def __init__(self, client: QwenMcpClient):
        self.client = client
    
    async def interactive_mode(self):
        """대화형 모드"""
        print("\n" + "="*70)
        print("🤖 Qwen2.5-1.5B Blockchain Analysis Chat")
        print("="*70)
        print("Commands:")
        print("  • quit/exit - 종료")
        print("  • clear - 대화 기록 초기화")
        print("  • stats - 시스템 상태 확인")
        print("  • demo - 자동 데모 실행")
        print("\n💡 Example queries:")
        print("  • 'Analyze wallet 0x742d...' - 지갑 포트폴리오 분석")
        print("  • 'What is the TVL of Uniswap?' - DeFi 프로토콜 정보")
        print("  • 'Show me DeFi market overview' - 전체 시장 현황")
        print("  • 'Compare Uniswap and SushiSwap' - 프로토콜 비교")
        print("="*70)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'clear':
                    self.client.clear_conversation()
                    print("🧹 Conversation cleared")
                    continue
                
                elif user_input.lower() == 'stats':
                    stats = self.client.get_statistics()
                    print("📊 System Statistics:")
                    print(json.dumps(stats, indent=2))
                    continue
                
                elif user_input.lower() == 'demo':
                    await self.run_demo()
                    continue
                
                elif not user_input:
                    continue
                
                # 사용자 질문 처리
                print("🤔 Processing...")
                start_time = time.time()
                
                response = await self.client.chat(user_input)
                
                end_time = time.time()
                print(f"\n🤖 Assistant: {response}")
                
                if CONFIG.DEBUG_MODE:
                    print(f"\n⏱️ Response time: {end_time - start_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                if CONFIG.DEBUG_MODE:
                    import traceback
                    traceback.print_exc()
    
    async def run_demo(self):
        """자동 데모 실행"""
        print("\n🎭 Running automatic demo...")
        
        for i, query in enumerate(CONFIG.DEMO_QUERIES, 1):
            print(f"\n{'='*50}")
            print(f"🎯 Demo {i}/{len(CONFIG.DEMO_QUERIES)}: {query}")
            print('='*50)
            
            try:
                start_time = time.time()
                response = await self.client.chat(query)
                end_time = time.time()
                
                print(f"📝 Response: {response}")
                print(f"⏱️ Time: {end_time - start_time:.2f}s")
                
                # 데모 결과 저장
                save_demo_result(query, {
                    'response': response,
                    'time_taken': end_time - start_time,
                    'success': True
                })
                
                # 데모 간 잠시 대기
                if i < len(CONFIG.DEMO_QUERIES):
                    print("\n⏳ Waiting 3 seconds...")
                    await asyncio.sleep(3)
                
            except Exception as e:
                print(f"❌ Demo {i} failed: {e}")
                save_demo_result(query, {
                    'error': str(e),
                    'success': False
                })
        
        print(f"\n✅ Demo completed! Results saved to examples/demo_results.json")
    
    async def batch_mode(self, queries: List[str]):
        """배치 모드 - 여러 질문을 순차 처리"""
        print(f"\n📋 Batch processing {len(queries)} queries...")
        
        results = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n🔄 Processing query {i}/{len(queries)}: {query}")
            
            try:
                start_time = time.time()
                response = await self.client.chat(query)
                end_time = time.time()
                
                result = {
                    "query": query,
                    "response": response,
                    "time_taken": end_time - start_time,
                    "success": True
                }
                
                results.append(result)
                print(f"✅ Query {i} completed in {end_time - start_time:.2f}s")
                
            except Exception as e:
                result = {
                    "query": query,
                    "error": str(e),
                    "success": False
                }
                results.append(result)
                print(f"❌ Query {i} failed: {e}")
            
            # 메모리 정리
            self.client.model_manager.clear_cache()
        
        return results

# 메인 실행 함수
async def main():
    """메인 함수"""
    import sys
    
    try:
        print("""
    ██████╗ ██╗    ██╗███████╗███╗   ██╗    ███╗   ███╗ ██████╗██████╗ 
    ██╔═══██╗██║    ██║██╔════╝████╗  ██║    ████╗ ████║██╔════╝██╔══██╗
    ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║    ██╔████╔██║██║     ██████╔╝
    ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║    ██║╚██╔╝██║██║     ██╔═══╝ 
    ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║    ██║ ╚═╝ ██║╚██████╗██║     
     ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝    ╚═╝     ╚═╝ ╚═════╝╚═╝     
    
    Qwen2.5-1.5B + FastMCP Blockchain Analysis System
        """)
        
        # 설정 검증
        if not CONFIG.validate():
            print("❌ Configuration validation failed")
            return
        
        # Qwen MCP 클라이언트 초기화
        client = QwenMcpClient()
        
        print("🔧 Initializing system...")
        await client.initialize()
        
        # 채팅 인터페이스 생성
        chat_interface = ChatInterface(client)
        
        # 실행 모드 결정
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
            
            if mode in ['interactive', 'i']:
                await chat_interface.interactive_mode()
            
            elif mode == 'demo':
                await chat_interface.run_demo()
            
            elif mode == 'batch' and len(sys.argv) > 2:
                # 배치 모드 - 명령행에서 질문들 입력
                queries = sys.argv[2:]
                results = await chat_interface.batch_mode(queries)
                
                print("\n📊 Batch Results Summary:")
                successful = sum(1 for r in results if r.get('success', False))
                print(f"✅ Successful: {successful}/{len(results)}")
                
                if CONFIG.DEBUG_MODE:
                    for result in results:
                        status = "✅" if result.get('success', False) else "❌" 
                        print(f"{status} {result['query']}")
            
            elif mode == 'quick' and len(sys.argv) > 2:
                # 빠른 질문 모드
                query = " ".join(sys.argv[2:])
                print(f"❓ Quick query: {query}")
                
                response = await client.chat(query)
                print(f"🤖 Response: {response}")
            
            else:
                print("Usage: python client.py [interactive|demo|batch|quick] [queries...]")
                print("\nExamples:")
                print("  python client.py interactive")
                print("  python client.py demo")
                print('  python client.py quick "What is Uniswap TVL?"')
                print('  python client.py batch "Show DeFi overview" "Analyze wallet 0x123..."')
        
        else:
            # 기본값: 대화형 모드
            await chat_interface.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n👋 System interrupted")
    except Exception as e:
        print(f"❌ System error: {e}")
        if CONFIG.DEBUG_MODE:
            import traceback
            traceback.print_exc()
    finally:
        if 'client' in locals():
            await client.cleanup()

# 개별 함수 테스트
async def test_qwen_model():
    """Qwen 모델 개별 테스트"""
    print("🧪 Testing Qwen model independently...")
    
    model_manager = QwenModelManager()
    await model_manager.load_model()
    
    test_prompt = """You are a blockchain analyst. Answer this question: What is DeFi?

Answer: """
    
    response = await model_manager.generate_response(test_prompt)
    print(f"✅ Model test response: {response}")
    
    model_manager.clear_cache()

async def test_mcp_connection():
    """MCP 연결 개별 테스트"""
    print("🧪 Testing MCP connection independently...")
    
    mcp_connector = MCPConnector()
    await mcp_connector.start_mcp_server()
    
    # 테스트 도구 호출
    test_result = await mcp_connector.call_tool('get_defi_market_overview', {})
    print(f"✅ MCP test result: {test_result}")
    
    mcp_connector.stop_server()

if __name__ == "__main__":
    import sys
    
    # 테스트 모드 지원
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        if len(sys.argv) > 2:
            if sys.argv[2] == 'model':
                asyncio.run(test_qwen_model())
            elif sys.argv[2] == 'mcp':
                asyncio.run(test_mcp_connection())
            else:
                print("Usage: python client.py test [model|mcp]")
        else:
            print("Running all tests...")
            asyncio.run(test_qwen_model())
            asyncio.run(test_mcp_connection())
    else:
        # 일반 실행
        asyncio.run(main())