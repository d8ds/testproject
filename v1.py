import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Any
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager


class GlassdoorDatabase:
    """Database interface for Glassdoor reviews data"""
    
    def __init__(self, db_path: str = "glassdoor_reviews.db"):
        self.db_path = db_path
    
    def get_recent_reviews(self, company_ticker: str, days_back: int = 30) -> pd.DataFrame:
        """Get recent reviews for a company"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT company_name, ticker, review_date, overall_rating, 
               work_life_balance, culture_values, career_opportunities,
               comp_benefits, senior_management, recommend_to_friend,
               ceo_approval, business_outlook, review_text, job_title,
               employment_status, review_length, helpful_count
        FROM reviews 
        WHERE ticker = ? AND review_date >= date('now', '-{} days')
        ORDER BY review_date DESC
        """.format(days_back)
        
        df = pd.read_sql_query(query, conn, params=(company_ticker,))
        conn.close()
        return df
    
    def get_companies_with_recent_activity(self, min_reviews: int = 10) -> List[str]:
        """Get companies with sufficient recent review activity"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT ticker, COUNT(*) as review_count
        FROM reviews 
        WHERE review_date >= date('now', '-30 days')
        GROUP BY ticker
        HAVING COUNT(*) >= ?
        ORDER BY review_count DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(min_reviews,))
        conn.close()
        return df['ticker'].tolist()


class TradingSignalsSystem:
    """Main system orchestrating the AutoGen agents"""
    
    def __init__(self, db_path: str = "glassdoor_reviews.db"):
        self.db = GlassdoorDatabase(db_path)
        self.setup_agents()
        self.setup_group_chat()
    
    def setup_agents(self):
        """Initialize all AutoGen agents with their specific roles"""
        
        # Configuration for all agents
        config_list = [
            {
                "model": "gpt-4",
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        ]
        
        llm_config = {
            "config_list": config_list,
            "temperature": 0.1,
        }
        
        # Data Analyst Agent
        self.data_analyst = ConversableAgent(
            name="DataAnalyst",
            system_message="""You are a data analyst specializing in employee review analysis.
            Your role is to:
            1. Process Glassdoor review data for companies
            2. Calculate key metrics and trends
            3. Identify significant changes in employee sentiment
            4. Provide statistical summaries and insights
            
            Focus on quantitative analysis and present findings clearly with numbers and trends.
            Always specify the time period and sample size for your analysis.""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Sentiment Analyst Agent
        self.sentiment_analyst = ConversableAgent(
            name="SentimentAnalyst",
            system_message="""You are a sentiment analysis expert focused on workplace reviews.
            Your role is to:
            1. Analyze sentiment trends in employee reviews
            2. Identify key themes and concerns
            3. Detect sentiment shifts that might indicate business changes
            4. Classify reviews into positive, negative, and neutral categories
            
            Pay special attention to:
            - Management quality indicators
            - Company direction and outlook mentions
            - Layoff or hiring signals
            - Cultural changes""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Signal Generator Agent
        self.signal_generator = ConversableAgent(
            name="SignalGenerator",
            system_message="""You are a quantitative analyst who converts employee sentiment data into trading signals.
            Your role is to:
            1. Synthesize data from analysts into actionable trading signals
            2. Assign signal strength (Strong Buy/Buy/Hold/Sell/Strong Sell)
            3. Provide confidence levels and reasoning
            4. Consider signal timing and duration
            
            Signal criteria:
            - Strong positive sentiment trends → Buy signals
            - Management rating improvements → Positive signals
            - Layoff indicators or negative outlook → Sell signals
            - Mixed signals → Hold recommendations
            
            Always provide specific reasoning and confidence levels (1-10).""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Risk Manager Agent
        self.risk_manager = ConversableAgent(
            name="RiskManager",
            system_message="""You are a risk management specialist who validates trading signals.
            Your role is to:
            1. Assess the reliability of generated signals
            2. Identify potential risks and limitations
            3. Suggest position sizing and risk controls
            4. Flag signals that need additional validation
            
            Consider:
            - Sample size adequacy
            - Data quality issues
            - Market conditions
            - Signal reliability based on historical patterns
            
            You can approve, modify, or reject signals with clear reasoning.""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
        
        # Orchestrator Agent
        self.orchestrator = ConversableAgent(
            name="Orchestrator",
            system_message="""You are the system orchestrator managing the analysis workflow.
            Your role is to:
            1. Coordinate analysis across all agents
            2. Ensure all required analysis is completed
            3. Synthesize final recommendations
            4. Present clear, actionable outputs
            
            Process flow:
            1. Data Analyst processes raw data
            2. Sentiment Analyst provides sentiment insights
            3. Signal Generator creates trading signals
            4. Risk Manager validates and approves
            5. You synthesize the final recommendation
            
            Always ensure each step is completed before moving to the next.""",
            llm_config=llm_config,
            human_input_mode="NEVER"
        )
    
    def setup_group_chat(self):
        """Setup the group chat for agent collaboration"""
        self.group_chat = GroupChat(
            agents=[
                self.orchestrator,
                self.data_analyst,
                self.sentiment_analyst,
                self.signal_generator,
                self.risk_manager
            ],
            messages=[],
            max_round=20,
            speaker_selection_method="round_robin"
        )
        
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={
                "config_list": [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}],
                "temperature": 0.1,
            }
        )
    
    def analyze_company(self, ticker: str, days_back: int = 30) -> Dict[str, Any]:
        """Analyze a specific company and generate trading signals"""
        
        # Get the data
        reviews_df = self.db.get_recent_reviews(ticker, days_back)
        
        if reviews_df.empty:
            return {"error": f"No recent data found for {ticker}"}
        
        # Convert DataFrame to summary for agents
        data_summary = self._create_data_summary(reviews_df, ticker)
        
        # Start the analysis workflow
        analysis_prompt = f"""
        Please analyze the following Glassdoor review data for {ticker} and generate trading signals:

        {data_summary}

        Follow this workflow:
        1. DataAnalyst: Provide statistical analysis of the review data
        2. SentimentAnalyst: Analyze sentiment trends and key themes
        3. SignalGenerator: Generate trading signal based on the analysis
        4. RiskManager: Validate the signal and assess risks
        5. Orchestrator: Synthesize final recommendation

        Each agent should complete their analysis before the next agent proceeds.
        """
        
        # Execute the group chat
        self.orchestrator.initiate_chat(
            self.group_chat_manager,
            message=analysis_prompt
        )
        
        # Extract results from the conversation
        return self._extract_final_recommendation()
    
    def _create_data_summary(self, df: pd.DataFrame, ticker: str) -> str:
        """Create a concise summary of the review data for agents"""
        
        summary = f"""
        Company: {ticker}
        Analysis Period: {df['review_date'].min()} to {df['review_date'].max()}
        Total Reviews: {len(df)}
        
        Rating Averages:
        - Overall Rating: {df['overall_rating'].mean():.2f}
        - Work-Life Balance: {df['work_life_balance'].mean():.2f}
        - Culture & Values: {df['culture_values'].mean():.2f}
        - Career Opportunities: {df['career_opportunities'].mean():.2f}
        - Compensation & Benefits: {df['comp_benefits'].mean():.2f}
        - Senior Management: {df['senior_management'].mean():.2f}
        
        Recommendation Metrics:
        - Recommend to Friend: {df['recommend_to_friend'].mean():.1%}
        - CEO Approval: {df['ceo_approval'].mean():.1%}
        - Positive Business Outlook: {df['business_outlook'].mean():.1%}
        
        Recent Trend (last 7 days vs previous period):
        - Overall Rating Trend: {self._calculate_trend(df, 'overall_rating')}
        - Management Rating Trend: {self._calculate_trend(df, 'senior_management')}
        - Outlook Trend: {self._calculate_trend(df, 'business_outlook')}
        
        Review Volume Trend: {self._calculate_volume_trend(df)}
        """
        
        return summary
    
    def _calculate_trend(self, df: pd.DataFrame, column: str) -> str:
        """Calculate trend for a specific metric"""
        df['review_date'] = pd.to_datetime(df['review_date'])
        recent = df[df['review_date'] >= (df['review_date'].max() - timedelta(days=7))]
        older = df[df['review_date'] < (df['review_date'].max() - timedelta(days=7))]
        
        if len(recent) == 0 or len(older) == 0:
            return "Insufficient data"
        
        recent_avg = recent[column].mean()
        older_avg = older[column].mean()
        change = recent_avg - older_avg
        
        if change > 0.1:
            return f"Improving (+{change:.2f})"
        elif change < -0.1:
            return f"Declining ({change:.2f})"
        else:
            return f"Stable ({change:+.2f})"
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> str:
        """Calculate review volume trend"""
        df['review_date'] = pd.to_datetime(df['review_date'])
        recent_count = len(df[df['review_date'] >= (df['review_date'].max() - timedelta(days=7))])
        older_count = len(df[df['review_date'] < (df['review_date'].max() - timedelta(days=7))])
        
        if older_count == 0:
            return "New activity"
        
        change_ratio = recent_count / (older_count / 3)  # Normalize for 7 vs ~23 day periods
        
        if change_ratio > 1.5:
            return f"Increasing ({change_ratio:.1f}x normal)"
        elif change_ratio < 0.5:
            return f"Decreasing ({change_ratio:.1f}x normal)"
        else:
            return "Normal"
    
    def _extract_final_recommendation(self) -> Dict[str, Any]:
        """Extract the final recommendation from the agent conversation"""
        # This would parse the conversation history to extract structured results
        # For now, return a placeholder structure
        return {
            "timestamp": datetime.now().isoformat(),
            "conversation_history": [msg for msg in self.group_chat.messages],
            "status": "completed"
        }
    
    def scan_all_companies(self, min_reviews: int = 10) -> List[Dict[str, Any]]:
        """Scan all companies with sufficient data and generate signals"""
        active_tickers = self.db.get_companies_with_recent_activity(min_reviews)
        results = []
        
        for ticker in active_tickers[:5]:  # Limit to first 5 for demo
            try:
                result = self.analyze_company(ticker)
                result['ticker'] = ticker
                results.append(result)
            except Exception as e:
                results.append({
                    'ticker': ticker,
                    'error': str(e)
                })
        
        return results


# Usage Example
if __name__ == "__main__":
    # Initialize the system
    system = TradingSignalsSystem("glassdoor_reviews.db")
    
    # Analyze a specific company
    print("Analyzing AAPL...")
    result = system.analyze_company("AAPL")
    print(json.dumps(result, indent=2, default=str))
    
    # Scan all active companies
    print("\nScanning all active companies...")
    all_results = system.scan_all_companies()
    for result in all_results:
        print(f"{result.get('ticker', 'Unknown')}: {result.get('status', 'Error')}")
