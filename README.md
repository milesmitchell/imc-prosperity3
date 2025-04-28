# Imperial Isle Round 5 Trading Bot 🏝️

Final Round Code Submission for the **IMC Prosperity 3 Global Trading Challenge**  
Our team consisted of **Miles Mitchell**, **Miji Trenkel**, **Rohan Chadha**, and **Shane Reilly**

We are all studying MSc Financial Technology at Imperial College London

---

## 📚 About the Competition, our Experimentation, and Results

This repository contains our code that helped us rank 29th Globally and 3rd in the UK in the Final Round of the competition, out of over 12,600 teams. The final round was a combination of all those prior and and hence the most heavily weighted round of IMC's Global Trading Competition. This was our first year competing, and now we are familiar with the competition structure we look forward to coming back and taking the Top25 spot next year.

Our trading bot implements a **modular multi-product strategy suite**. Throughout the competition we experimented with:
- Machine learning-driven trading signals by training models and adding decision tree logic manually due to library restrictions
- Statistical arbitrage for ETPs (exchange-traded products)
- Option pricing based on Black-Scholes + volatility skew adjustments
- Dynamic Bollinger Bands reversion/momentum trading
- Counterparty behavior tracking (e.g. Olivia informed trader detection)
- Discretionary-inspired sunlight index trading

In our first year competing, we placed **top 2% globally overall** and **top 0.25% globally** in the final round out of 12,620 teams.

## Final Round Strategy Highlights

Our final-round trading bot is modular and manages several independent but coordinated strategies:

| Asset | Strategy | Description |
|:---|:---|:---|
| **KELP** | Bollinger Band Mean Reversion | Dynamic volatility-adaptive entry/exit thresholds |
| **VOLCANIC_ROCK** | Mean Reversion | Bollinger strategy trained each round using RollingWindowCV, with Bayesian Optimisation for parameter selection |
| **VOLCANIC_ROCK_VOUCHERS** | Options Mean Reversion | Copying signals of the underlying for ITM options due to higher correlation. Special handling for deep out-of-the-money 10500 strike calls |
| **SQUID_INK** | Extreme Move Detection | Migrated from bollinger bands initially, to taking aggressive entry/exits on sharp deviations |
| **RAINFOREST_RESIN** | Market Making & Taking | Developed by teammates; Edge-optimised quoting around mid-wall prices |
| **PICNIC_BASKETS** | ETF Arbitrage | Spread trading between baskets and constituents. Exploits mispricings based on reconstructed fair value indices |
| **DJEMBES** | Mean reversion and dynamic spread adjustment | Simple momentum strategy |
| **CROISSANTS** | Informed Trader Tracking | Developed tool to track effiency and sharpe, noting most/least efficient and highest/lowest sharpe for each instrument. Ultimately followed "Olivia"'s counterparty patterns entirely as they beat our previous results |
| **MAGNIFICENT_MACARONS** | Sunlight-Driven Trading | Critical sunlight index found with dynamic take profit based on gradient magnitude. Momentum trading based on sunlight trends. Also incorporates opportunistic arbitrage between domestic and international prices |

We also incorporated:
- **State persistence** with `jsonpickle` for informed trader tracking.
- **Dynamic position sizing** based on volatility and inventory risk.
- **Lightweight real-time logging** for monitoring and debugging.

---

## Key Features

- Modular strategy structure — easy to toggle, debug, and extend
- Adaptive thresholds based on recent volatility conditions
- Persistent memory between rounds (traderData) for behavioral learning
- Fast, efficient code for live competition environments
- Integration-ready with IMC’s provided backtester framework

---

## Results Summary

| Metric | Result |
|:---|:---|
| Global Ranking | 🌍 238th / 12,620 teams |
| Final Round Score | 🚀 Top 0.25% Globally |
| UK National Ranking (Final Round) | 🥉 3rd |

---

## Lessons Learned

- **Simple > Overfitted**: Strategies must be explainable and have sound economic logic.
- **Execution Matters**: Must consider live trading issues that don't appear in backtesting; we lost 100k in R4 due to one variable not persisting through the JSON pickle, which only caused issues on AWS not in the backtester.
- **Discretionary Elements Help**: Even in algorithmic trading, recognizing behavioral patterns (like "Olivia" trades) added serious edge.

---

## Future Goals

We plan to return for Prosperity 4 aiming for a **Global top 25 finish** — armed with deeper knowledge of competition dynamics, faster development pipelines, and enhanced backtesting tools to hit the ground running.

---

## How to Run

This bot is designed for the **IMC Prosperity 3 simulation framework**.

To use:
1. Clone/download this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt

---

## 📜 License

This repository uses the **MIT License** — feel free to fork, modify, or reuse with attribution!  
*(LICENSE file included.)*

---

## 🤝 Connect

If you liked our project, found this helpful, or are interested in discussing further, feel free to connect with me on LinkedIn (https://www.linkedin.com/in/miles2/)

---
