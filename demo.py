"""
FastMCP Blockchain Analysis - Demo Script
해커톤 데모 실행 스크립트
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any

from config import CONFIG
from client import QwenMcpClient, ChatInterface
from utils import print_debug, save_demo_result

class DemoRunner:
    """데모 실행 관리자"""
    
    def __init__(self):
        self.client = None
        self.results = []
        self.start_time = None
        
    async def initialize(self):
        """데모 시스템 초기화"""
        print("🎭 Initializing Demo System...")
        self.start_time = time.time()
        
        # 클라이언트 초기화
        self.client = QwenMcpClient()
        await self.client.initialize()
        
        print("✅ Demo system ready!")
    
    async def run_full_demo(self):
        """전체 데모 실행"""
        print("\n" + "="*70)
        print("🚀 FastMCP Blockchain Analysis - LIVE DEMO")
        print("="*70)
        print(f"🕐 Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤖 Model: {CONFIG.MODEL_NAME}")
        print(f"🔧 Debug mode: {'ON' if CONFIG.DEBUG_MODE else 'OFF'}")
        print("="*70)
        
        demo_scenarios = [
            {
                'title': '📊 DeFi Market Overview',
                'query': 'Show me the current DeFi market overview with top protocols and chains',
                'explanation': 'Getting real-time DeFi market data from DefiLlama API'
            },
            {
                'title': '🦄 Protocol Analysis',
                'query': 'What is the TVL of Uniswap protocol? Include chain distribution.',
                'explanation': 'Analyzing specific DeFi protocol with detailed breakdown'
            },
            {
                'title': '💼 Wallet Portfolio Analysis',
                'query': f'Analyze the portfolio for wallet {CONFIG.DEMO_WALLET}',
                'explanation': 'Real-time wallet analysis using Alchemy blockchain API'
            },
            {
                'title': '📈 Transaction History',
                'query': f'Get the recent transaction history for wallet {CONFIG.DEMO_WALLET}',
                'explanation': 'Blockchain transaction tracking and analysis'
            },
            {
                'title': '⚖️ Protocol Comparison',
                'query': 'Compare Uniswap and SushiSwap protocols by TVL and features',
                'explanation': 'Comparative analysis of competing DeFi protocols'
            }
        ]
        
        for i, scenario in enumerate(demo_scenarios, 1):
            await self._run_scenario(i, scenario, len(demo_scenarios))
        
        await self._show_demo_summary()
    
    async def _run_scenario(self, scenario_num: int, scenario: Dict, total: int):
        """개별 시나리오 실행"""
        print(f"\n{'='*60}")
        print(f"🎯 Demo {scenario_num}/{total}: {scenario['title']}")
        print(f"💡 {scenario['explanation']}")
        print('='*60)
        print(f"❓ Query: {scenario['query']}")
        print("\n🤔 Processing...")
        
        start_time = time.time()
        
        try:
            # 실제 질의 실행
            response = await self.client.chat(scenario['query'])
            end_time = time.time()
            
            duration = end_time - start_time
            
            print(f"\n🤖 Assistant Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            print(f"⏱️ Response time: {duration:.2f} seconds")
            
            # 결과 저장
            result = {
                'scenario_num': scenario_num,
                'title': scenario['title'],
                'query': scenario['query'],
                'response': response,
                'duration': duration,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            save_demo_result(scenario['query'], result)
            
            print("✅ Scenario completed successfully")
            
            # 다음 시나리오 전 잠시 대기
            if scenario_num < total:
                print("\n⏳ Preparing next scenario...")
                await asyncio.sleep(2)
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n❌ Scenario failed: {e}")
            print(f"⏱️ Time until failure: {duration:.2f} seconds")
            
            # 실패 결과 저장
            result = {
                'scenario_num': scenario_num,
                'title': scenario['title'],
                'query': scenario['query'],
                'error': str(e),
                'duration': duration,
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            save_demo_result(scenario['query'], result)
    
    async def _show_demo_summary(self):
        """데모 요약 출력"""
        total_time = time.time() - self.start_time
        successful_scenarios = sum(1 for r in self.results if r['success'])
        total_scenarios = len(self.results)
        
        print(f"\n{'='*70}")
        print("📊 DEMO SUMMARY")
        print('='*70)
        print(f"🕐 Total demo time: {total_time:.1f} seconds")
        print(f"✅ Successful scenarios: {successful_scenarios}/{total_scenarios}")
        print(f"⏱️ Average response time: {sum(r['duration'] for r in self.results if r['success']) / max(successful_scenarios, 1):.2f}s")
        
        if successful_scenarios == total_scenarios:
            print("🎉 ALL SCENARIOS COMPLETED SUCCESSFULLY!")
        else:
            print(f"⚠️ {total_scenarios - successful_scenarios} scenarios failed")
        
        # 개별 시나리오 결과
        print(f"\n📋 Scenario Results:")
        for result in self.results:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {result['title']} ({result['duration']:.2f}s)")
        
        # 시스템 통계
        if self.client:
            stats = self.client.get_statistics()
            print(f"\n💻 System Statistics:")
            print(f"  • Model: {stats['model_info']['name']}")
            print(f"  • Device: {stats['model_info']['device']}")
            print(f"  • Memory: {stats['model_info']['memory_usage_gb']:.1f}GB")
            print(f"  • Total messages: {stats['conversation_messages']}")
            print(f"  • Tool calls: {stats['tool_calls_made']}")
        
        print("="*70)
        print("🎬 Demo completed! Thank you for watching.")
        print("📁 Detailed results saved to: examples/demo_results.json")
        print("="*70)

class InteractiveDemo:
    """인터랙티브 데모 모드"""
    
    def __init__(self, demo_runner: DemoRunner):
        self.demo_runner = demo_runner
    
    async def run(self):
        """인터랙티브 데모 실행"""
        print("\n🎮 Interactive Demo Mode")
        print("Choose your demo scenarios:")
        
        scenarios = [
            "📊 DeFi Market Overview",
            "🦄 Uniswap Protocol Analysis", 
            "💼 Wallet Portfolio Analysis",
            "📈 Transaction History",
            "⚖️ Protocol Comparison",
            "🎯 Custom Query",
            "🚀 Run Full Auto Demo"
        ]
        
        while True:
            print(f"\n{'='*50}")
            print("Available options:")
            for i, scenario in enumerate(scenarios, 1):
                print(f"  {i}. {scenario}")
            print("  0. Exit")
            print('='*50)
            
            try:
                choice = input("👤 Enter your choice (0-7): ").strip()
                
                if choice == '0':
                    print("👋 Exiting interactive demo")
                    break
                
                elif choice == '7':
                    await self.demo_runner.run_full_demo()
                
                elif choice == '6':
                    custom_query = input("💬 Enter your custom query: ").strip()
                    if custom_query:
                        await self._run_custom_query(custom_query)
                
                elif choice in ['1', '2', '3', '4', '5']:
                    await self._run_selected_scenario(int(choice))
                
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def _run_selected_scenario(self, choice: int):
        """선택된 시나리오 실행"""
        scenario_map = {
            1: {
                'title': '📊 DeFi Market Overview',
                'query': 'Show me the current DeFi market overview',
                'explanation': 'Real-time DeFi market analysis'
            },
            2: {
                'title': '🦄 Uniswap Analysis',
                'query': 'What is the TVL of Uniswap protocol?',
                'explanation': 'Detailed protocol analysis'
            },
            3: {
                'title': '💼 Wallet Analysis',
                'query': f'Analyze portfolio for {CONFIG.DEMO_WALLET}',
                'explanation': 'Comprehensive wallet breakdown'
            },
            4: {
                'title': '📈 Transaction History',
                'query': f'Get transaction history for {CONFIG.DEMO_WALLET}',
                'explanation': 'Recent blockchain activity'
            },
            5: {
                'title': '⚖️ Protocol Comparison',
                'query': 'Compare Uniswap and SushiSwap',
                'explanation': 'Side-by-side protocol analysis'
            }
        }
        
        scenario = scenario_map.get(choice)
        if scenario:
            await self.demo_runner._run_scenario(1, scenario, 1)
    
    async def _run_custom_query(self, query: str):
        """커스텀 쿼리 실행"""
        print(f"\n🎯 Custom Query: {query}")
        print("🤔 Processing...")
        
        start_time = time.time()
        
        try:
            response = await self.demo_runner.client.chat(query)
            end_time = time.time()
            
            print(f"\n🤖 Response:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            print(f"⏱️ Time: {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")

async def main():
    """메인 함수"""
    import sys
    
    try:
        # 데모 러너 초기화
        demo_runner = DemoRunner()
        await demo_runner.initialize()
        
        # 실행 모드 결정
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
            
            if mode == 'auto':
                # 자동 전체 데모
                await demo_runner.run_full_demo()
            
            elif mode == 'interactive':
                # 인터랙티브 모드
                interactive_demo = InteractiveDemo(demo_runner)
                await interactive_demo.run()
            
            elif mode == 'quick':
                # 빠른 단일 쿼리
                if len(sys.argv) > 2:
                    query = " ".join(sys.argv[2:])
                    scenario = {
                        'title': '🚀 Quick Query',
                        'query': query,
                        'explanation': 'User-provided query'
                    }
                    await demo_runner._run_scenario(1, scenario, 1)
                else:
                    print("Usage: python demo.py quick 'your question here'")
            
            else:
                print("Usage: python demo.py [auto|interactive|quick] [query]")
                print("\nModes:")
                print("  auto        - Run full automatic demo")
                print("  interactive - Interactive demo selection")  
                print("  quick       - Single query demo")
        
        else:
            # 기본값: 자동 전체 데모
            await demo_runner.run_full_demo()
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        if CONFIG.DEBUG_MODE:
            import traceback
            traceback.print_exc()
    finally:
        if 'demo_runner' in locals() and demo_runner.client:
            await demo_runner.client.cleanup()

if __name__ == "__main__":
    print("""
    🎭 FastMCP Blockchain Analysis Demo
    ===================================
    
    🚀 Showcasing Qwen2.5-1.5B + FastMCP + Real Blockchain APIs
    """)
    
    asyncio.run(main())