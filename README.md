# Imperial Isle Round 5 Trading Bot 🏝️
Team "Imperial Isle" Final Round Code for IMC Prosperity 3 Global Trading Challenge

This repository contains our trading algorithm built for the **IMC Prosperity Challenge** — Round 5: *Imperial Isle*.
Our team consisted of Miles Mitchell, Shane Reilly, Rohan Chadha, and Miji Trenkel.
We are all MSc Financial Technology student at Imperial College London

The bot implements several modular trading strategies across a variety of the available products, including options, ETPs, and commodities.

## 🛠 Strategy Overview for the final round

- **Kelp Bollinger Band Strategy**  
  - Trading `KELP` using dynamically tuned Bollinger Bands.
  - Adaptive based on recent volatility.

- **Volcanic Rock + Options Trading**  
  - Mean reversion on `VOLCANIC_ROCK` with additional call option strategies.
  - Special logic for deep OTM (`10500`) calls.

- **Squid Ink Extreme Mean Reversion**  
  - Sharp threshold detection on `SQUID_INK` deviations.
  - Aggressive entry/exit points.

- **Rainforest Resin Market Taking**  
  - Efficient market taking and liquidity making on `RAINFOREST_RESIN`.

- **Djembe Rolling Strategy**  
  - Dynamic spread adjustment and EMA-based mean reversion trading for `DJEMBES`.

- **Croissants Arbitrage**  
  - Exploits informed trader (`Olivia`) trades on `CROISSANTS`.

- **Picnic Basket Arbitrage**  
  - Index arbitrage between baskets (`PICNIC_BASKET1`, `PICNIC_BASKET2`) and their components (`CROISSANTS`, `JAMS`, `DJEMBES`).

- **Magnificent Macarons (Sunlight Gradient Strategy)**  
  - Momentum trading on `MAGNIFICENT_MACARONS` using sunlight index trends.
  - Sunlight-driven arbitrage vs local/foreign market prices.

---

## 📈 Features

- Position-aware dynamic sizing and thresholds
- Adaptive stop loss and take profit mechanisms
- Multi-layer state persistence using `jsonpickle`
- Lightweight logging format for backtest efficiency
- Modular architecture — easy to add/remove strategies

---

## 🚀 Running the Bot

This bot is designed to integrate directly into the **IMC Prosperity Challenge** simulation framework and can be utilised with JMerle's backtester,
with the exception of the macaron conversion logic.

To use:

1. Clone/download the repo
2. Plug `imperial_isle_round5.py` into your local simulator
3. Run `Trader().run(state)` at each tick

---

## 📜 License

This repository uses the **MIT License** — feel free to fork, modify, or reuse with attribution!  
*(LICENSE file included.)*

---

## 🤝 Connect

If you liked our project, found this helpful, or are interested in discussing further, feel free to connect with me on LinkedIn (https://www.linkedin.com/in/miles2/)

---
