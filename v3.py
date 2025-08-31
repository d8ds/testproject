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
