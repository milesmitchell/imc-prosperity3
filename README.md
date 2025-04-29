# Imperial Isle Round 5 Trading Bot + Reversion Strategy Bayesian Optimisation Notebook 🏝️

Final Round Code Submission for the **IMC Prosperity 3 Global Trading Challenge**  
Our team consisted of **Miles Mitchell**, **Miji Trenkel**, **Rohan Chadha**, and **Shane Reilly**

We are all currently studying MSc Financial Technology at Imperial College London

---

## 📚 About the Competition, our Experimentation, and Results

This repository contains our submitted code that helped us rank **top 2% globally overall** but more specifically rank 29th Globally (top 0.25%) and 3rd in the UK in the Final Round of the competition, out of over 12,600 teams. The repository also includes some of our most important optimisation code to explore the hyperparameter space effectively for mean reversion strategies. This helped us tune our parameters through the rounds for our z-score strategies, which made the bulk of our algorithmic profits. The final round was a combination of all those prior and hence the most heavily weighted round of IMC's Global Trading Competition. We thoroughly enjoyed the competition and would like to extend our thanks to IMC for both organising and running it this year. This was our first year competing, and now we are familiar with the competition structure we look forward to coming back and taking the overall Top25 spot next year.

### Our trading bot:
This implements a **modular multi-product strategy suite**. Throughout the competition we experimented with:
- Machine Learning-driven strategies, training sparse random forest models and then adding decision tree logic to live file manually due to library restrictions. We abandoned these methods for more effective momentum/reversion strategies after round2
- Statistical arbitrage for ETPs (exchange-traded products)
- Option pricing based on Black-Scholes + volatility skew adjustments, and reversion strategies tracking the underlying for highly correlated ITM call options
- Dynamic Bollinger Bands reversion/momentum trading
- Counterparty behavior tracking (e.g. Olivia informed trader detection)
- Discretionary-elements, for example dynamic TP levels for macarons scaled by sunlight index gradient

### My optimisation notebook: 
I also developed a separate intuitively customisable parameter optimisation notebook to better understand how our Bollinger-style Z-Score Threshold strategy performed under different parameter configurations, and to visualise trends as a polynomial surface in the hyperparameter space.

Notably, this notebook:

1, Uses Bayesian Optimisation with Optuna to tune the rolling window length, z-threshold, and max volume per tick.

2, Evaluates performance using either mean/median Sharpe ratio or cumulative return across multiple trading days.

3, Visualises the optimisation surface in 3D from multiple angles, and fits a polynomial regression surface to analyse parameter momentum effects.

4, Shows how performance varies with hyperparameters rather than attempting to generalise — prioritising exploration over prediction.

**Important Note:** The purpose of this notebook is to explore the behaviour of the parameterised trading strategy across a limited but dense set of intraday data. **The purpose of this notebook is not to aim for robust out-of-sample generalisation**, rather the focus is on understanding how such a strategy responds to different parameter configurations and whether there are any trends in the hyperparamter space. The polynomial fit and corresponding R² help visualise and summarise optimisation trends across the space of hyperparameters, giving insight into momentum and mean-reversion characteristics rather than producing a predictive model.

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
| **CROISSANTS** | Informed Trader Tracking | Developed tool to track efficiency and sharpe of each trader, noting most/least efficient and highest/lowest sharpe for each instrument. Ultimately followed "Olivia"'s counterparty patterns entirely as they beat our previous results |
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
| Global Ranking Overall | 🌍 238th / 12,620 teams |
| Final Round Score | 🚀 Top 0.25% Globally |
| Final Round UK National Ranking | 🥉 3rd |

---

## Lessons Learned

- **Simple > Overfitted**: Strategies must be explainable and have sound economic logic.
- **Execution Matters**: Must consider live trading issues that don't appear in backtesting; we lost 100k in R4 due to one variable not persisting through the JSON pickle, which only caused issues on AWS not in the backtester.
- **Discretionary Elements Help**: Even in algorithmic trading, recognizing behavioral patterns (like "Olivia" trades) added serious edge.

---

## Future Goals

We plan to return for Prosperity 4 aiming for a **Global top 25 finish** — armed with deeper knowledge of competition dynamics, faster development pipelines, and enhanced backtesting tools to hit the ground running.

---

## 🚀 How to Run

This bot is designed for the **IMC Prosperity 3 simulation framework**.

To use:
1. Clone/download this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
 ```

---

## 📜 License

This repository uses the **MIT License** — feel free to fork, modify, or reuse with attribution!  
*(LICENSE file included.)*

---

## 🤝 Connect

If you liked our project, found this helpful, or are interested in discussing further, feel free to connect with me on LinkedIn (https://www.linkedin.com/in/miles2/)

---
