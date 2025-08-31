#!/usr/bin/env python3

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any

# Import our system components
# from glassdoor_trading_system import TradingSignalsSystem
# from database_setup import setup_environment, TradingSignalStorage


class TradingSignalsRunner:
    """Main runner for the Glassdoor trading signals system"""
    
    def __init__(self):
        self.system = None
        self.signal_storage = None
        self.setup_system()
    
    def setup_system(self):
        """Initialize the system components"""
        print("🚀 Initializing Glassdoor Trading Signals System...")
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY environment variable not set!")
            print("Please set your OpenAI API key:")
            print("export OPENAI_API_KEY='your-api-key-here'")
            sys.exit(1)
        
        try:
            # Initialize database and sample data if needed
            if not os.path.exists("glassdoor_reviews.db"):
                print("📊 Setting up database with sample data...")
                from database_setup import setup_environment
                setup_environment()
            
            # Initialize the main system
            from glassdoor_trading_system import TradingSignalsSystem
            from database_setup import TradingSignalStorage
            
            self.system = TradingSignalsSystem()
            self.signal_storage = TradingSignalStorage()
            
            print("✅ System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing system: {str(e)}")
            sys.exit(1)
    
    def analyze_single_company(self, ticker: str) -> Dict[str, Any]:
        """Analyze a single company and return results"""
        print(f"\n🔍 Analyzing {ticker}...")
        
        try:
            start_time = time.time()
            result = self.system.analyze_company(ticker, days_back=30)
            end_time = time.time()
            
            print(f"⏱️  Analysis completed in {end_time - start_time:.1f} seconds")
            
            # Extract key information from the conversation
            signal_summary = self.extract_signal_summary(result)
            
            # Store the signal if valid
            if signal_summary.get('signal_type'):
                signal_summary['ticker'] = ticker
                self.signal_storage.store_signal(signal_summary)
                print(f"💾 Signal stored for {ticker}")
            
            return result
            
        except Exception as e:
            error_result = {
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            print(f"❌ Error analyzing {ticker}: {str(e)}")
            return error_result
    
    def extract_signal_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trading signal summary from agent conversation"""
        # This is a simplified extraction - in practice, you'd parse the conversation
        # more sophisticatedly to extract structured signals
        
        if result.get('error'):
            return {}
        
        # Default signal structure
        signal = {
            'signal_date': datetime.now(),
            'signal_type': 'HOLD',
            'signal_strength': 'MODERATE',
            'confidence_score': 5.0,
            'reasoning': 'Analysis completed but specific signal extraction needs implementation',
            'data_period_start': datetime.now().date(),
            'data_period_end': datetime.now().date(),
            'review_count': 0,
            'avg_rating': 3.0,
            'sentiment_score': 0.0,
            'risk_assessment': 'Medium risk - requires further validation'
        }
        
        return signal
    
    def run_full_scan(self):
        """Run analysis on all active companies"""
        print("\n🔄 Running full market scan...")
        
        try:
            results = self.system.scan_all_companies(min_reviews=10)
            
            print(f"\n📈 Scan Results ({len(results)} companies analyzed):")
            print("=" * 80)
            
            for i, result in enumerate(results, 1):
                ticker = result.get('ticker', 'Unknown')
                status = 'Error' if result.get('error') else 'Completed'
                
                print(f"{i:2d}. {ticker:6s} - {status}")
                
                if result.get('error'):
                    print(f"    ❌ {result['error']}")
                else:
                    signal = self.extract_signal_summary(result)
                    if signal.get('signal_type'):
                        print(f"    📊 Signal: {signal['signal_type']} ({signal['confidence_score']:.1f}/10)")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during full scan: {str(e)}")
            return []
    
    def display_recent_signals(self, days_back: int = 7):
        """Display recent trading signals"""
        print(f"\n📋 Recent Trading Signals (last {days_back} days):")
        print("=" * 80)
        
        try:
            signals_df = self.signal_storage.get_recent_signals(days_back)
            
            if signals_df.empty:
                print("No recent signals found.")
                return
            
            for _, signal in signals_df.iterrows():
                print(f"🎯 {signal['ticker']} - {signal['signal_type']} ({signal['signal_strength']})")
                print(f"   📅 Date: {signal['signal_date']}")
                print(f"   🎯 Confidence: {signal['confidence_score']:.1f}/10")
                print(f"   📝 Reasoning: {signal['reasoning'][:100]}...")
                print(f"   ⚠️  Risk: {signal['risk_assessment'][:50]}...")
                print("-" * 40)
                
        except Exception as e:
            print(f"❌ Error retrieving signals: {str(e)}")
    
    def interactive_mode(self):
        """Run the system in interactive mode"""
        print("\n🎮 Interactive Mode - Glassdoor Trading Signals")
        print("=" * 60)
        
        while True:
            print("\nAvailable commands:")
            print("1. analyze [TICKER] - Analyze a specific company")
            print("2. scan - Run full market scan")
            print("3. signals - Show recent signals")
            print("4. history [TICKER] - Show signal history for ticker")
            print("5. quit - Exit the system")
            
            try:
                command = input("\n💻 Enter command: ").strip().lower()
                
                if command == 'quit' or command == 'q':
                    print("👋 Goodbye!")
                    break
                
                elif command.startswith('analyze'):
                    parts = command.split()
                    if len(parts) > 1:
                        ticker = parts[1].upper()
                        result = self.analyze_single_company(ticker)
                        self.display_analysis_result(result)
                    else:
                        print("❌ Please specify a ticker: analyze AAPL")
                
                elif command == 'scan':
                    self.run_full_scan()
                
                elif command == 'signals':
                    self.display_recent_signals()
                
                elif command.startswith('history'):
                    parts = command.split()
                    if len(parts) > 1:
                        ticker = parts[1].upper()
                        self.display_signal_history(ticker)
                    else:
                        print("❌ Please specify a ticker: history AAPL")
                
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def display_analysis_result(self, result: Dict[str, Any]):
        """Display detailed analysis result"""
        print("\n📊 Analysis Result:")
        print("=" * 50)
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
            return
        
        # Display conversation summary
        conversation = result.get('conversation_history', [])
        if conversation:
            print(f"💬 Agent Conversation ({len(conversation)} messages)")
            
            # Show key messages from each agent
            for msg in conversation[-5:]:  # Show last 5 messages
                sender = msg.get('name', 'Unknown')
                content = msg.get('content', '')[:200]
                print(f"   🤖 {sender}: {content}...")
        
        print(f"✅ Analysis completed at {result.get('timestamp', 'Unknown time')}")
    
    def display_signal_history(self, ticker: str):
        """Display signal history for a ticker"""
        print(f"\n📈 Signal History for {ticker}:")
        print("=" * 50)
        
        try:
            history_df = self.signal_storage.get_signal_history(ticker)
            
            if history_df.empty:
                print(f"No signal history found for {ticker}")
                return
            
            for _, signal in history_df.iterrows():
                print(f"📅 {signal['signal_date']}: {signal['signal_type']} "
                      f"({signal['signal_strength']}) - {signal['confidence_score']:.1f}/10")
                
        except Exception as e:
            print(f"❌ Error retrieving history: {str(e)}")
    
    def run_demo(self):
        """Run a quick demonstration of the system"""
        print("\n🎬 Running System Demo...")
        print("=" * 50)
        
        # Demo companies to analyze
        demo_tickers = ['AAPL', 'TSLA', 'MSFT']
        
        for ticker in demo_tickers:
            print(f"\n🎯 Demo Analysis: {ticker}")
            result = self.analyze_single_company(ticker)
            
            # Brief summary
            if not result.get('error'):
                print(f"✅ {ticker} analysis completed successfully")
            else:
                print(f"❌ {ticker} analysis failed: {result['error']}")
            
            time.sleep(2)  # Brief pause between analyses
        
        # Show results
        print("\n📊 Demo Results Summary:")
        self.display_recent_signals(1)


def main():
    """Main entry point"""
    print("🌟 Glassdoor Trading Signals System")
    print("=" * 60)
    
    # Initialize the runner
    runner = TradingSignalsRunner()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'demo':
            runner.run_demo()
        
        elif command == 'scan':
            runner.run_full_scan()
        
        elif command == 'interactive' or command == 'i':
            runner.interactive_mode()
        
        elif command.startswith('analyze') and len(sys.argv) > 2:
            ticker = sys.argv[2].upper()
            result = runner.analyze_single_company(ticker)
            runner.display_analysis_result(result)
        
        else:
            print("❌ Unknown command or missing arguments")
            print("\nUsage:")
            print("  python system_runner.py demo          - Run demonstration")
            print("  python system_runner.py scan          - Run full market scan")
            print("  python system_runner.py interactive   - Interactive mode")
            print("  python system_runner.py analyze AAPL  - Analyze specific ticker")
    
    else:
        # Default to interactive mode
        runner.interactive_mode()


if __name__ == "__main__":
    main()


# Additional utility functions for system monitoring and management

class SystemMonitor:
    """Monitor system performance and health"""
    
    def __init__(self, db_path: str = "glassdoor_reviews.db"):
        self.db_path = db_path
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Review statistics
        cursor = conn.execute("SELECT COUNT(*) FROM reviews")
        stats['total_reviews'] = cursor.fetchone()[0]
        
        cursor = conn.execute("SELECT COUNT(DISTINCT ticker) FROM reviews")
        stats['total_companies'] = cursor.fetchone()[0]
        
        cursor = conn.execute("""
            SELECT COUNT(*) FROM reviews 
            WHERE review_date >= date('now', '-30 days')
        """)
        stats['recent_reviews'] = cursor.fetchone()[0]
        
        # Signal statistics
        cursor = conn.execute("SELECT COUNT(*) FROM trading_signals")
        stats['total_signals'] = cursor.fetchone()[0]
        
        cursor = conn.execute("""
            SELECT COUNT(*) FROM trading_signals 
            WHERE signal_date >= datetime('now', '-7 days')
        """)
        stats['recent_signals'] = cursor.fetchone()[0]
        
        # Performance metrics
        cursor = conn.execute("""
            SELECT AVG(confidence_score) FROM trading_signals 
            WHERE signal_date >= datetime('now', '-30 days')
        """)
        avg_confidence = cursor.fetchone()[0]
        stats['avg_confidence'] = round(avg_confidence, 2) if avg_confidence else 0
        
        conn.close()
        return stats
    
    def print_system_status(self):
        """Print comprehensive system status"""
        stats = self.get_system_stats()
        
        print("\n📊 System Health Status")
        print("=" * 40)
        print(f"📋 Total Reviews: {stats['total_reviews']:,}")
        print(f"🏢 Companies Tracked: {stats['total_companies']}")
        print(f"📅 Recent Reviews (30d): {stats['recent_reviews']:,}")
        print(f"🎯 Total Signals Generated: {stats['total_signals']}")
        print(f"⚡ Recent Signals (7d): {stats['recent_signals']}")
        print(f"🎯 Avg Confidence Score: {stats['avg_confidence']}/10")
        
        # Health indicators
        print("\n🔍 Health Indicators:")
        if stats['recent_reviews'] > 50:
            print("✅ Data freshness: Good")
        else:
            print("⚠️  Data freshness: Low recent activity")
        
        if stats['avg_confidence'] >= 6:
            print("✅ Signal quality: Good")
        elif stats['avg_confidence'] >= 4:
            print("⚠️  Signal quality: Moderate")
        else:
            print("❌ Signal quality: Needs improvement")


# Configuration management
class ConfigManager:
    """Manage system configuration"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.default_config = {
            "database_path": "glassdoor_reviews.db",
            "min_reviews_for_analysis": 10,
            "analysis_period_days": 30,
            "confidence_threshold": 6.0,
            "max_companies_per_scan": 20,
            "openai_model": "gpt-4",
            "openai_temperature": 0.1
        }
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = self.default_config.copy()
                self.save_config()
        except Exception as e:
            print(f"⚠️  Error loading config, using defaults: {e}")
            self.config = self.default_config.copy()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value and save"""
        self.config[key] = value
        self.save_config()
    
    def print_config(self):
        """Print current configuration"""
        print("\n⚙️  System Configuration:")
        print("=" * 30)
        for key, value in self.config.items():
            print(f"{key}: {value}")


# Export key classes for external use
__all__ = [
    'TradingSignalsRunner',
    'SystemMonitor',
    'ConfigManager'
]
