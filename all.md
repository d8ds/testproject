# Glassdoor Trading Signals System

An AutoGen-based multi-agent system that analyzes Glassdoor employee reviews to generate trading signals.

## Requirements

### Python Dependencies

```txt
# Core AutoGen and AI
pyautogen>=0.2.0
openai>=1.0.0

# Data processing
pandas>=1.5.0
numpy>=1.24.0
sqlite3  # Built into Python

# NLP and ML (optional for advanced features)
scikit-learn>=1.3.0
transformers>=4.30.0
torch>=2.0.0

# Web scraping (if implementing live data collection)
requests>=2.31.0
beautifulsoup4>=4.12.0
selenium>=4.10.0

# Market data (optional for backtesting)
yfinance>=0.2.0
alpha-vantage>=2.3.0

# Utilities
python-dotenv>=1.0.0
```

### Environment Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up OpenAI API key:**
```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-openai-api-key-here"

# Option 2: .env file
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

3. **Initialize the system:**
```bash
# This will create the database and sample data
python database_setup.py
```

## Quick Start

### 1. Run Demo
```bash
python system_runner.py demo
```

### 2. Interactive Mode
```bash
python system_runner.py interactive
```

### 3. Analyze Specific Company
```bash
python system_runner.py analyze AAPL
```

### 4. Run Full Market Scan
```bash
python system_runner.py scan
```

## System Architecture

### Agent Roles

1. **DataAnalyst**: Processes raw review data and calculates statistical metrics
2. **SentimentAnalyst**: Analyzes sentiment trends and identifies key themes
3. **SignalGenerator**: Converts insights into actionable trading signals
4. **RiskManager**: Validates signals and assesses risks
5. **Orchestrator**: Coordinates the workflow and synthesizes results

### Database Schema

**Reviews Table:**
- Company and ticker information
- Review ratings across multiple dimensions
- Sentiment indicators (recommend, CEO approval, outlook)
- Review text and metadata

**Trading Signals Table:**
- Generated signals with confidence scores
- Reasoning and risk assessments
- Historical tracking for performance analysis

## Configuration

### Default Settings
```json
{
  "database_path": "glassdoor_reviews.db",
  "min_reviews_for_analysis": 10,
  "analysis_period_days": 30,
  "confidence_threshold": 6.0,
  "max_companies_per_scan": 20,
  "openai_model": "gpt-4",
  "openai_temperature": 0.1
}
```

### Customization
```python
from system_runner import ConfigManager

config = ConfigManager()
config.set("min_reviews_for_analysis", 20)
config.set("confidence_threshold", 7.0)
```

## Usage Examples

### Basic Analysis
```python
from glassdoor_trading_system import TradingSignalsSystem

system = TradingSignalsSystem()
result = system.analyze_company("AAPL", days_back=30)
print(result)
```

### Batch Processing
```python
# Analyze all active companies
results = system.scan_all_companies(min_reviews=15)
for result in results:
    print(f"{result['ticker']}: {result.get('status', 'Error')}")
```

### Signal Storage and Retrieval
```python
from database_setup import TradingSignalStorage

storage = TradingSignalStorage()
recent_signals = storage.get_recent_signals(days_back=7)
print(recent_signals)
```

## System Monitoring

```python
from system_runner import SystemMonitor

monitor = SystemMonitor()
monitor.print_system_status()
```

## Advanced Features

### Custom Agent Behavior
Modify agent system messages in `TradingSignalsSystem.setup_agents()` to customize analysis focus:

```python
self.sentiment_analyst = ConversableAgent(
    name="SentimentAnalyst",
    system_message="Your custom instructions here...",
    llm_config=llm_config
)
```

### Signal Validation
Implement custom validation logic in the RiskManager agent:

```python
# Add to RiskManager system message
"Additional validation criteria:
- Minimum sample size of 50 reviews
- Cross-reference with industry trends
- Consider seasonal factors"
```

### Data Integration
Replace sample data with real Glassdoor data:

```python
def load_real_data():
    # Your data loading logic here
    # Ensure data matches the expected schema
    pass
```

## Best Practices

### 1. Data Quality
- Validate review authenticity
- Filter spam and fake reviews
- Handle missing or incomplete data

### 2. Signal Reliability
- Use confidence thresholds
- Implement backtesting
- Monitor signal performance

### 3. Risk Management
- Set position size limits
- Implement stop-loss rules
- Consider market conditions

### 4. Compliance
- Respect Glassdoor's Terms of Service
- Ensure data usage compliance
- Implement appropriate rate limiting

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Check API key validity
   - Monitor rate limits
   - Handle API failures gracefully

2. **Database Issues**
   - Ensure SQLite permissions
   - Check disk space
   - Backup data regularly

3. **Agent Communication**
   - Monitor conversation flows
   - Handle agent failures
   - Implement fallback mechanisms

### Debug Mode
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor agent conversations
system.group_chat.messages  # View conversation history
```

## Performance Optimization

### 1. Database Optimization
- Create appropriate indexes
- Optimize queries
- Regular maintenance

### 2. API Usage
- Batch similar requests
- Cache responses where appropriate
- Implement retry logic

### 3. Analysis Speed
- Parallel processing for multiple companies
- Incremental analysis for large datasets
- Optimize agent interactions

## Extensions and Enhancements

### 1. Real-time Data
- Implement live data feeds
- WebSocket connections for updates
- Real-time signal generation

### 2. Advanced Analytics
- Machine learning models
- Predictive analytics
- Time series analysis

### 3. Integration
- Trading platform APIs
- Portfolio management systems
- Risk management tools

### 4. Visualization
- Signal dashboards
- Performance tracking
- Market correlation analysis

## Security Considerations

1. **API Key Management**
   - Use environment variables
   - Rotate keys regularly
   - Monitor usage

2. **Data Protection**
   - Encrypt sensitive data
   - Secure database access
   - Regular backups

3. **System Security**
   - Input validation
   - SQL injection prevention
   - Access controls

## Support and Contributing

For issues, enhancements, or questions:
1. Check the troubleshooting guide
2. Review agent conversation logs
3. Test with sample data first
4. Monitor system health metrics

## License and Disclaimer

This system is for educational and research purposes. Trading signals generated should be validated independently before making investment decisions. Past performance does not guarantee future results.
