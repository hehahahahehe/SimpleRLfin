# SimpleRLfin
# 🚀 FinRL Quantitative Trading Agent (2026 Edition)

This repository contains a Deep Reinforcement Learning (DRL) trading system built with **FinRL**, optimized for the 2026 market environment. It features a PPO (Proximal Policy Optimization) agent trained on high-growth AI and Tech stocks.



---

## 🛠 Troubleshooting & Solutions (Dev Log)

During the development on Google Colab (Python 3.12), we encountered and resolved several critical compatibility issues. This log serves as a guide for modern FinRL implementations.

### 1. Module Attribute Errors
* **Problem:** `AttributeError: module 'finrl' has no attribute '__version__'`
* **Solution:** This is a known packaging quirk in recent FinRL versions. It does not impact the core logic. Skip the version check and proceed with importing `finrl.meta` and `finrl.agents`.

### 2. Mandatory Environment Arguments
* **Problem:** `TypeError: __init__() missing 1 required positional argument: 'num_stock_shares'`
* **Solution:** The Gymnasium-based `StockTradingEnv` now requires explicit initialization for holdings and costs.
* **Fix:**
    ```python
    env_kwargs = {
        "num_stock_shares": [0] * len(TICKERS),
        "buy_cost_pct": [0.001] * len(TICKERS),
        "sell_cost_pct": [0.001] * len(TICKERS),
        # ... other params
    }
    ```

### 3. State Space Dimension Mismatch
* **Problem:** `ValueError: could not broadcast input array`
* **Solution:** Hardcoded state space dimensions often fail when tickers or indicators change. We implemented a dynamic calculation:
    $$State\_Space = 1 + 2 \times \text{num\_stocks} + (\text{num\_indicators} \times \text{num\_stocks})$$

### 4. Plotting & Timezone Errors
* **Problem:** `TypeError: tz must be string or tzinfo subclass` during Matplotlib rendering.
* **Solution:** Matplotlib occasionally misinterprets the date column as categorical data. We bypassed this by plotting against a numerical index and mapping the date strings onto the X-axis manually.

---

## 📈 Model Performance (Sample Run)
* **Agent:** PPO (Proximal Policy Optimization)
* **Environment:** Daily OHLCV data for NVDA, AAPL, MSFT, etc.
* **Result:** Achieved a **Sharpe Ratio of 1.195** during the training phase.
