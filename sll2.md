SLIDE 1: Investment Thesis
Exploiting Text Signals Across Index and Sector ETFs in Chinese Equity Markets
Market Opportunity:

Chinese equity markets exhibit information inefficiencies at multiple aggregation levels
Text-based signals underexploited relative to fundamental/technical factors
Liquid ETF vehicles across broad indices AND sectors enable scalable implementation

Data Sources:

Broker research reports (券商研报): 10,000+ reports/month across sectors
Earnings call transcripts: Management tone and forward guidance
Aggregation levels: Stock → Sector → Index

Trading Vehicles:

Broad indices: CSI 300, CSI 500 (market timing)
Sector ETFs: Technology, Finance, Consumer, Healthcare, Real Estate (rotation alpha)

Core Hypothesis:
Textual information aggregates differently across sectors → opportunities for both directional (index timing) and relative value (sector rotation) strategies

==============
SLIDE 2: Multi-Tier Aggregation Framework
Three-Level Hierarchy: Stock → Sector → Index

Individual Stocks (N ≈ 3,000)
    ↓ Tier 1 Aggregation
Sector Portfolios (N ≈ 10-12 sectors)
    ↓ Tier 2 Aggregation
Market Index (N = 1)



==============
Direct Sector ETF Signals
For sector-specific trading, bypass Tier 2:
# Trade sector ETF directly based on sector aggregate
Tech_ETF_Signal = f(tech_sector_features)
Consumer_ETF_Signal = f(consumer_sector_features)
```

---

#### **Signal Flow Examples**

**Example 1: Market Timing (Index-level)**
```
300 stocks × sentiment → 10 sectors × avg(sentiment) → 
→ Index sentiment = Σ(sector_sent × weight) → CSI 300 long/short
```

**Example 2: Sector Rotation**
```
300 stocks × policy_keywords → 10 sectors × policy_density →
→ Rank sectors by policy support → Long Tech ETF / Short Real Estate ETF
```

**Example 3: Within-Sector Signal**
```
30 tech stocks × numerical_precision → Tech sector precision = 0.72 →
→ High precision = confident guidance → Long Tech Sector ETF
```

---

### **SLIDE 3: Signal Catalog by Aggregation Level**

#### **Level 1: Index-Level Signals** (Market Timing)

For CSI 300/500 ETFs - captures broad market regime

| Signal | Construction | Logic | Trade |
|--------|--------------|-------|-------|
| **Market Sentiment** | Value-weighted avg of all stock sentiments | Overall market mood | Positive → Long CSI 300 |
| **Market Dispersion** | `σ(sector_sentiments)` across sectors | Low = consensus, High = divergence | High dispersion → Reduce beta |
| **Aggregate Info Flow** | `Σ(report_count all stocks)` vs. history | Attention intensity | Spike → Contrarian fade |
| **Temporal Orientation** | `Σ(future_tense) / Σ(total_words)` | Forward-looking market | Rising future focus → Risk-on |

**Strategy**: Directional index exposure (long/short/neutral CSI 300)

---

#### **Level 2: Cross-Sector Signals** (Sector Rotation)

Relative value across sectors - which to overweight/underweight

| Signal | Construction | Logic | Trade |
|--------|--------------|-------|-------|
| **Sentiment Spread** | `max(sector_sent) - min(sector_sent)` | Wide spread = rotation opportunity | Long best / Short worst |
| **Policy Divergence** | Rank sectors by policy keyword density | Government support concentration | Long policy-favored sectors |
| **Coverage Ratio** | `Tech_coverage / Finance_coverage` | Relative attention shifts | Follow coverage momentum |
| **Precision Ranking** | Rank sectors by numerical density | Guidance quality hierarchy | Long high-precision sectors |

**Strategy**: Tactical sector allocation (±5% from benchmark weights)

---

#### **Level 3: Sector-Specific Signals** (Direct Sector ETF Trading)

Signals that work best within specific sectors

| Sector | Primary Signals | Sector Characteristics | Expected IC |
|--------|----------------|------------------------|-------------|
| **Technology** | Policy keywords, Future orientation, Coverage intensity | High volatility, Policy-sensitive, Innovation-focused | 0.04-0.06 |
| **Finance** | Numerical precision, Regulatory keywords, Mgmt-broker gap | Stable, Guidance-heavy, Regulatory | 0.03-0.05 |
| **Consumer** | Sentiment dispersion, Call participation, Complexity | Retail-driven, Sentiment-sensitive | 0.03-0.05 |
| **Real Estate** | Policy keywords ONLY | Policy-dominated, Fundamentals secondary | 0.02-0.04 |
| **Healthcare** | Regulatory keywords, Precision, Innovation terms | R&D-heavy, Regulatory risk | 0.02-0.04 |

**Strategy**: Long/short sector pairs or single-sector directional

---

#### **Level 4: Hybrid Signals** (Combined Approach)

Leverage information across multiple levels

| Signal | Multi-Level Construction | Logic | Trade |
|--------|-------------------------|-------|-------|
| **Sector Momentum Divergence** | `sector_signal_momentum - index_signal_momentum` | Sector outperforming market tone | Long divergent sectors |
| **Within vs. Cross Dispersion** | `avg(intra-sector σ) vs. cross-sector σ` | Cohesion structure | High within-sector σ → Stock-picking mode |
| **Breadth-Weighted Index** | `Σ(sector_signal × sector_breadth)` | Quality-weighted market view | Better index timing |

---

### **SLIDE 4: Sector-Specific Signal Profiles**

#### **China Sector Landscape**

| Sector | CSI 300 Weight | ETF Liquidity | Broker Coverage | Transcript Availability | Policy Sensitivity |
|--------|----------------|---------------|-----------------|------------------------|-------------------|
| **Technology** | 25-30% | High | Excellent (40+ reports/stock/yr) | Good | Very High (★★★★★) |
| **Finance** | 20-25% | Very High | Excellent (30+ reports/stock/yr) | Moderate | High (★★★★) |
| **Consumer** | 15-20% | High | Good (25+ reports/stock/yr) | Good | Medium (★★★) |
| **Healthcare** | 8-12% | Medium | Good (20+ reports/stock/yr) | Fair | High (★★★★) |
| **Industrials** | 10-15% | Medium | Fair (15+ reports/stock/yr) | Fair | Medium (★★★) |
| **Real Estate** | 5-8% | Medium | Good (policy-driven) | Poor | Very High (★★★★★) |

---

#### **Sector Signal Mapping**

**Technology Sector** (芯片, 半导体, 5G, AI)
- **Best signals**: Policy keyword density, Future tense ratio, Coverage intensity
- **Logic**: Innovation-driven, government support critical, forward-looking
- **Challenges**: High volatility, rapid narrative shifts
- **Trade**: Long when policy keywords spike + future orientation high

**Finance Sector** (银行, 保险, 券商)
- **Best signals**: Numerical precision, Regulatory keywords, Broker-management gap
- **Logic**: Guidance-heavy, regulatory constraints, conservative language
- **Challenges**: Low volatility, slow-moving fundamentals
- **Trade**: Short when precision declines (vague guidance = caution)

**Consumer Sector** (消费, 零售, 食品饮料)
- **Best signals**: Sentiment dispersion, Earnings call engagement, Sentiment momentum
- **Logic**: Retail investor-driven, confidence-sensitive
- **Challenges**: Earnings seasonality, brand-specific noise
- **Trade**: Long when dispersion low + sentiment accelerating (consensus forming)

**Real Estate Sector** (房地产, 建筑)
- **Best signals**: Policy keywords ONLY (政策, 调控, 支持)
- **Logic**: Fundamentals irrelevant, entirely policy-driven
- **Challenges**: Policy unpredictability, distressed state
- **Trade**: Short-term tactical only around policy announcements

---

### **SLIDE 5: Implementation Strategy**

#### **Three-Pronged Approach**
```
Strategy 1: Index Timing (50% of risk budget)
    ↓
Strategy 2: Sector Rotation (40% of risk budget)
    ↓
Strategy 3: Sector Pairs (10% of risk budget)

=========

