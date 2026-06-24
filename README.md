# Smart Order Router

An ML model that decides how to split and time large equity orders across the trading day to minimize implementation shortfall (slippage), compared against standard execution benchmarks (TWAP, VWAP, POV).

---

## Overview

Executing a large order naively — all at once — moves the market against you. The goal of smart order routing is to break the order into smaller child orders and time them to minimize total execution cost. This project trains a GBM slippage predictor and uses it to greedily optimize the execution schedule interval by interval.

---

## How It Works

### 1. Market Simulation
- 390-interval intraday session (one per minute of a full trading day)
- **U-shaped volume curve** — higher volume at open and close
- **Mean-reverting mid price** with intraday random walk
- **Autocorrelated bid-ask spread** — widens during high volatility
- **Order imbalance** — autocorrelated buy/sell pressure signal

### 2. Slippage Model (Almgren-Chriss Inspired)

```
slippage = spread/2 + temporary_impact + permanent_impact + imbalance_cost
```

- `temporary_impact ∝ participation_rate^1.5 × volatility`
- `permanent_impact ∝ participation_rate × volatility`
- `imbalance_cost` — extra cost when buying into buy-side pressure

### 3. Training Data
- 1,500 random execution scenarios × 390 intervals each
- Each row: (participation rate, volatility, spread, imbalance, time of day, …) → slippage in bps

### 4. GBM Slippage Predictor
- Trained on the simulated data to predict per-interval slippage
- Used at execution time to score candidate participation rates

### 5. ML-Optimized Schedule
- At each interval, grid-search over 10 participation rates (1%–50% of interval volume)
- Score each by `predicted_slippage + urgency_penalty`
- Urgency penalty rises when remaining shares are large relative to time left

---

## Strategies Compared

| Strategy | Logic |
|----------|-------|
| **TWAP** | Equal share slices at every interval |
| **VWAP** | Proportional to historical volume profile |
| **POV** | Fixed 10% of each interval's volume |
| **ML** | Greedy minimization of predicted slippage |

---

## Results (50,000-share order)

The ML router achieves lower implementation shortfall than TWAP and POV by adapting to real-time spread and imbalance conditions. VWAP is a strong baseline; ML improves on it by responding to intraday signals rather than just volume.

---

## Output Plots

| File | Description |
|------|-------------|
| `order_router_comparison.png` | Intraday price, execution schedules, cumulative fills, per-interval slippage, total IS cost bar chart |
| `slippage_model_performance.png` | Predicted vs actual slippage scatter, residual distribution |

---

## Usage

```bash
pip install numpy pandas scikit-learn matplotlib
python smart_order_router.py
```

To change order size:

```python
# In smart_order_router.py, bottom of file:
run(total_shares=100_000, plot=True)
```

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
```
