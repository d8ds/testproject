import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import json


def create_database_schema(db_path: str = "glassdoor_reviews.db"):
    """Create the database schema for Glassdoor reviews"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create reviews table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company_name TEXT NOT NULL,
        ticker TEXT NOT NULL,
        review_date DATE NOT NULL,
        overall_rating REAL,
        work_life_balance REAL,
        culture_values REAL,
        career_opportunities REAL,
        comp_benefits REAL,
        senior_management REAL,
        recommend_to_friend INTEGER,  -- 0 or 1
        ceo_approval INTEGER,         -- 0 or 1
        business_outlook INTEGER,     -- 0 or 1
        review_text TEXT,
        job_title TEXT,
        employment_status TEXT,
        review_length INTEGER,
        helpful_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Create indexes for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON reviews(ticker, review_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_date ON reviews(review_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON reviews(ticker)")
    
    # Create signals table to store generated trading signals
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trading_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        signal_date TIMESTAMP NOT NULL,
        signal_type TEXT NOT NULL,  -- 'BUY', 'SELL', 'HOLD'
        signal_strength TEXT,       -- 'STRONG', 'MODERATE', 'WEAK'
        confidence_score REAL,      -- 1-10
        reasoning TEXT,
        data_period_start DATE,
        data_period_end DATE,
        review_count INTEGER,
        avg_rating REAL,
        sentiment_score REAL,
        risk_assessment TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON trading_signals(ticker, signal_date)")
    
    conn.commit()
    conn.close()
    print(f"Database schema created at {db_path}")


def generate_sample_data(db_path: str = "glassdoor_reviews.db", num_companies: int = 10, reviews_per_company: int = 50):
    """Generate sample Glassdoor review data for testing"""
    
    # Sample companies with tickers
    companies = [
        ("Apple Inc.", "AAPL"),
        ("Microsoft Corporation", "MSFT"),
        ("Amazon.com Inc.", "AMZN"),
        ("Alphabet Inc.", "GOOGL"),
        ("Tesla Inc.", "TSLA"),
        ("Meta Platforms Inc.", "META"),
        ("Netflix Inc.", "NFLX"),
        ("NVIDIA Corporation", "NVDA"),
        ("Salesforce Inc.", "CRM"),
        ("Adobe Inc.", "ADBE")
    ]
    
    job_titles = [
        "Software Engineer", "Product Manager", "Data Scientist", "Sales Representative",
        "Marketing Manager", "Customer Success Manager", "DevOps Engineer", "UX Designer",
        "Business Analyst", "Account Executive", "HR Manager", "Finance Analyst"
    ]
    
    employment_statuses = ["Current Employee", "Former Employee", "Contractor"]
    
    sample_reviews = [
        "Great company culture and excellent benefits. Management is supportive and the work is challenging.",
        "Good work-life balance but compensation could be better. Team collaboration is excellent.",
        "Fast-paced environment with lots of learning opportunities. Sometimes stressful but rewarding.",
        "Management needs improvement. Good technical challenges but poor communication from leadership.",
        "Excellent career growth opportunities. The company is going in the right direction.",
        "Work-life balance is poor. Too much pressure and unrealistic deadlines.",
        "Great compensation and benefits. Company culture is improving under new leadership.",
        "Innovative projects but management is disorganized. Lots of potential if they fix leadership issues.",
        "Best company I've worked for. Amazing colleagues and interesting work.",
        "Company is struggling with direction. Lots of layoffs and uncertainty about the future."
    ]
    
    conn = sqlite3.connect(db_path)
    
    for company_name, ticker in companies[:num_companies]:
        for _ in range(reviews_per_company):
            # Generate random review data
            review_date = datetime.now() - timedelta(days=random.randint(1, 90))
            
            # Create correlated ratings (companies with trends)
            base_rating = random.uniform(2.5, 4.5)
            trend_factor = random.uniform(-0.3, 0.3)
            
            overall_rating = max(1, min(5, base_rating + trend_factor))
            work_life_balance = max(1, min(5, base_rating + random.uniform(-0.5, 0.5)))
            culture_values = max(1, min(5, base_rating + random.uniform(-0.4, 0.4)))
            career_opportunities = max(1, min(5, base_rating + random.uniform(-0.6, 0.6)))
            comp_benefits = max(1, min(5, base_rating + random.uniform(-0.3, 0.7)))
            senior_management = max(1, min(5, base_rating + trend_factor + random.uniform(-0.5, 0.5)))
            
            recommend_to_friend = 1 if overall_rating > 3.5 else random.choice([0, 1])
            ceo_approval = 1 if senior_management > 3.5 else random.choice([0, 1])
            business_outlook = 1 if overall_rating > 3.2 and random.random() > 0.3 else 0
            
            review_text = random.choice(sample_reviews)
            job_title = random.choice(job_titles)
            employment_status = random.choice(employment_statuses)
            review_length = len(review_text)
            helpful_count = random.randint(0, 15)
            
            # Insert the review
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO reviews (
                company_name, ticker, review_date, overall_rating,
                work_life_balance, culture_values, career_opportunities,
                comp_benefits, senior_management, recommend_to_friend,
                ceo_approval, business_outlook, review_text, job_title,
                employment_status, review_length, helpful_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                company_name, ticker, review_date.date(), overall_rating,
                work_life_balance, culture_values, career_opportunities,
                comp_benefits, senior_management, recommend_to_friend,
                ceo_approval, business_outlook, review_text, job_title,
                employment_status, review_length, helpful_count
            ))
    
    conn.commit()
    conn.close()
    print(f"Generated sample data for {num_companies} companies with {reviews_per_company} reviews each")


def view_sample_data(db_path: str = "glassdoor_reviews.db", ticker: str = "AAPL", limit: int = 10):
    """View sample data from the database"""
    
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT ticker, company_name, review_date, overall_rating, 
           senior_management, business_outlook, recommend_to_friend,
           LEFT(review_text, 100) as review_preview
    FROM reviews 
    WHERE ticker = ?
    ORDER BY review_date DESC 
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(ticker, limit))
    conn.close()
    
    print(f"\nSample data for {ticker}:")
    print("=" * 80)
    for _, row in df.iterrows():
        print(f"Date: {row['review_date']}")
        print(f"Rating: {row['overall_rating']:.1f} | Management: {row['senior_management']:.1f}")
        print(f"Outlook: {'Positive' if row['business_outlook'] else 'Negative'} | "
              f"Recommend: {'Yes' if row['recommend_to_friend'] else 'No'}")
        print(f"Preview: {row['review_preview']}...")
        print("-" * 40)


class TradingSignalStorage:
    """Helper class to store and retrieve trading signals"""
    
    def __init__(self, db_path: str = "glassdoor_reviews.db"):
        self.db_path = db_path
    
    def store_signal(self, signal_data: dict):
        """Store a trading signal in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO trading_signals (
            ticker, signal_date, signal_type, signal_strength, 
            confidence_score, reasoning, data_period_start, 
            data_period_end, review_count, avg_rating, 
            sentiment_score, risk_assessment
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_data.get('ticker'),
            signal_data.get('signal_date', datetime.now()),
            signal_data.get('signal_type'),
            signal_data.get('signal_strength'),
            signal_data.get('confidence_score'),
            signal_data.get('reasoning'),
            signal_data.get('data_period_start'),
            signal_data.get('data_period_end'),
            signal_data.get('review_count'),
            signal_data.get('avg_rating'),
            signal_data.get('sentiment_score'),
            signal_data.get('risk_assessment')
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_signals(self, days_back: int = 7) -> pd.DataFrame:
        """Get recent trading signals"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT * FROM trading_signals 
        WHERE signal_date >= datetime('now', '-{} days')
        ORDER BY signal_date DESC
        """.format(days_back)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_signal_history(self, ticker: str) -> pd.DataFrame:
        """Get signal history for a specific ticker"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT * FROM trading_signals 
        WHERE ticker = ?
        ORDER BY signal_date DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(ticker,))
        conn.close()
        return df


# Configuration and setup functions
def setup_environment():
    """Setup the complete environment"""
    print("Setting up Glassdoor Trading Signals System...")
    
    # Create database schema
    create_database_schema()
    
    # Generate sample data
    generate_sample_data(num_companies=10, reviews_per_company=50)
    
    # Display sample data
    view_sample_data("AAPL", 5)
    view_sample_data("TSLA", 5)
    
    print("\n" + "="*80)
    print("Setup complete! You can now run the trading signals system.")
    print("Make sure to set your OPENAI_API_KEY environment variable.")
    print("="*80)


if __name__ == "__main__":
    setup_environment()
