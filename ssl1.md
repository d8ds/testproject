SLIDE 1: Investment Thesis
Exploiting Textual Information for Chinese Index ETF Trading
Market Opportunity:

Chinese equity markets exhibit lower informational efficiency vs. developed markets
Text-based signals underexploited in A-share market
Index ETFs provide liquid, scalable trading vehicles

Data Sources:

Broker research reports (券商研报): Sell-side attention & sentiment
Earnings call transcripts: Management tone & forward guidance
Coverage: CSI 300, CSI 500, sector ETFs

Unique Angle:

Target: Index-level signals, not stock picking
Approach: Behavioral patterns from aggregated text features
Edge: Multi-source divergence (broker vs. management perspectives)
===========
SLIDE 2: Aggregation Methodology
Stock Data: 300 stocks × sentiment scores
↓
Aggregate: Index_sentiment = Σ(sentiment_i × weight_i)
          Dispersion = σ(sentiment across stocks)
          Momentum = sentiment_today - sentiment_30d_avg
↓
Trading Signal: Combine multiple aggregation views
============
SLIDE 4: Implementation Roadmap
Phase 1: Foundation (Month 1-2)
Quick Win Signals:

Numerical Precision (easiest to implement)
Coverage Intensity (broker report count dynamics)
Sentiment Dispersion (cross-sectional volatility)

Infrastructure:

Chinese NLP pipeline (jieba segmentation, sentiment lexicon)
Data ingestion (broker reports + transcripts)
Feature calculation engine


Phase 2: China-Specific (Month 3-4)
Market-Aware Signals:

Policy Keyword Density
Broker-Management Gap
Coverage Concentration (HHI)

Enhancements:

Event studies (policy announcements, earnings seasons)
Regime detection (bull/bear/sideways markets)
Cross-index validation (CSI 300 vs. sector ETFs)


Phase 3: Advanced (Month 5-6)
Sophisticated Approaches:

Multi-signal combination (ensemble methods)
Dynamic weighting by market regime
Factor decomposition (PCA on linguistic features)

===========
Key Risks
Data Quality:

Earnings call transcript availability (less common than US)
Broker report standardization issues
Chinese NLP complexity (segmentation, sentiment accuracy)

Market Structure:

Retail-dominated A-shares → High noise-to-signal
Policy intervention risk → Regime breaks
Short-selling constraints → Long-bias required

Signal Decay:

Crowding if approach becomes popular
Adaptive management behavior
Regulatory changes in disclosure requirements

==========

