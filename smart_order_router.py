"""
Smart Order Router
ML model that decides how to split and time large orders to minimize slippage.
Compares TWAP/VWAP baselines against a GBM-optimized execution schedule.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# Market Microstructure Model 

@dataclass
class MarketState:
    """Snapshot of the market at a given intraday interval."""
    interval: int           # 0..N-1 (e.g. 0=9:30, 1=9:31 … for 1-min bars)
    price: float
    spread: float           # bid-ask spread as fraction of price
    volume: float           # interval volume (shares)
    volatility: float       # rolling short-term vol
    order_imbalance: float  # (buy_vol - sell_vol) / total_vol, ∈ [-1, 1]
    time_of_day: float      # 0..1 (fraction of trading day elapsed)


def simulate_intraday_market(
    n_intervals: int = 390,   # full trading day in minutes
    base_price: float = 100.0,
    seed: int = 42,
) -> List[MarketState]:
    """
    Simulate realistic intraday market data with:
    - U-shaped volume curve (high at open/close)
    - Mean-reverting price with intraday trends
    - Spread widening at open/close and during volatility spikes
    """
    rng = np.random.default_rng(seed)
    states = []

    # Volume pattern: U-shaped (higher near open and close)
    t = np.linspace(0, 1, n_intervals)
    vol_pattern = 2.0 + 3.0 * np.exp(-20 * (t - 0.0)**2) + 3.0 * np.exp(-20 * (t - 1.0)**2)
    vol_pattern /= vol_pattern.mean()  # normalize to mean 1

    price = base_price
    rolling_vol = 0.01

    for i in range(n_intervals):
        # Price random walk with mean reversion
        drift = 0.0001 * (base_price - price)
        shock = rng.normal(0, rolling_vol)
        price = max(price * (1 + drift + shock), 1.0)

        # Update rolling volatility
        rolling_vol = 0.95 * rolling_vol + 0.05 * abs(shock)
        rolling_vol = np.clip(rolling_vol, 0.002, 0.05)

        # Spread: wider at open/close and during high vol
        base_spread = 0.0005
        spread = base_spread * (1 + 2 * vol_pattern[i] * rolling_vol / 0.01)
        spread = np.clip(spread, 0.0003, 0.01)

        # Interval volume (shares)
        volume = vol_pattern[i] * 50_000 * rng.lognormal(0, 0.3)



        # Order imbalance (autocorrelated)
        imbalance = rng.normal(0, 0.3)
        if i > 0:
            imbalance = 0.6 * states[-1].order_imbalance + 0.4 * imbalance
        imbalance = np.clip(imbalance, -1, 1)


        states.append(MarketState(
            interval=i,
            price=price,
            spread=spread,
            volume=volume,
            volatility=rolling_vol,
            order_imbalance=imbalance,
            time_of_day=t[i],
        ))

    return states


#  Slippage Model
def compute_slippage(
    order_size: float,
    market: MarketState,
    participation_rate: float,  # what fraction of interval volume we consume
) -> float:
    """
    Almgren-Chriss inspired slippage model.
    slippage = spread/2 + temporary_impact + permanent_impact
    """
    # Participation penalty: more slippage as we consume more of the interval volume
    temp_impact = 0.005 * (participation_rate ** 1.5) * market.volatility / 0.01

    # Permanent impact (moves the market)
    perm_impact = 0.001 * participation_rate * market.volatility / 0.01

    # Spread cost: always pay half the spread
    spread_cost = market.spread / 2

    # Order imbalance: buying into buy imbalance is costlier
    imbalance_cost = 0.001 * max(0, market.order_imbalance) if order_size > 0 else 0

    return spread_cost + temp_impact + perm_impact + imbalance_cost


# Execution Strategies 

def twap_schedule(total_shares: float, n_intervals: int) -> np.ndarray:
    """Time-Weighted Average Price: equal slices every interval."""
    return np.full(n_intervals, total_shares / n_intervals)


def vwap_schedule(total_shares: float, market_states: List[MarketState]) -> np.ndarray:
    """Volume-Weighted Average Price: proportional to expected volume."""
    volumes = np.array([s.volume for s in market_states])
    weights = volumes / volumes.sum()
    return weights * total_shares


def pov_schedule(total_shares: float, market_states: List[MarketState],
                 pov_rate: float = 0.10) -> np.ndarray:
    """Percentage of Volume: trade a fixed % of each interval's volume."""
    sizes = np.array([s.volume * pov_rate for s in market_states])
    # Scale to hit total_shares target
    sizes = sizes * (total_shares / sizes.sum())
    return sizes


def execute_schedule(
    schedule: np.ndarray,
    market_states: List[MarketState],
) -> Dict:
    """Simulate execution of an order schedule, return cost metrics."""
    total_cost = 0.0
    total_shares = schedule.sum()
    arrival_price = market_states[0].price
    executed = []

    for i, (size, state) in enumerate(zip(schedule, market_states)):
        if size <= 0:
            continue
        participation = size / max(state.volume, 1)
        slip = compute_slippage(size, state, participation)
        cost = size * state.price * (1 + slip)   # buying: pay price + slippage
        total_cost += cost
        executed.append({
            "interval": i,
            "size": size,
            "price": state.price,
            "slippage_bps": slip * 10000,
            "participation": participation,
        })

    avg_execution_price = total_cost / total_shares
    implementation_shortfall = (avg_execution_price / arrival_price - 1) * 10000  # bps
    avg_slippage_bps = np.mean([e["slippage_bps"] for e in executed])

    return {
        "avg_exec_price": avg_execution_price,
        "implementation_shortfall_bps": implementation_shortfall,
        "avg_slippage_bps": avg_slippage_bps,
        "executed": pd.DataFrame(executed),
    }


# ML Slippage Predictor & Optimizer 

def generate_training_data(n_scenarios: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate training data: random execution decisions + their realized slippage.
    Each row is one interval-decision with features and realized slippage.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for scenario in range(n_scenarios):
        total_shares = rng.uniform(1000, 100_000)
        n_intervals = rng.integers(30, 390)
        market = simulate_intraday_market(n_intervals=n_intervals, seed=scenario)

        # Random participation rate for this interval
        for i, state in enumerate(market):
            part_rate = rng.uniform(0.01, 0.50)
            size = state.volume * part_rate
            slip = compute_slippage(size, state, part_rate)

            remaining_frac = (n_intervals - i) / n_intervals

            rows.append({
                "participation_rate": part_rate,
                "volatility": state.volatility,
                "spread": state.spread,
                "order_imbalance": state.order_imbalance,
                "time_of_day": state.time_of_day,
                "remaining_fraction": remaining_frac,
                "volume_norm": state.volume / 50_000,
                "slippage_bps": slip * 10000,
            })

    return pd.DataFrame(rows)


def train_slippage_model(df: pd.DataFrame):
    feature_cols = [
        "participation_rate", "volatility", "spread", "order_imbalance",
        "time_of_day", "remaining_fraction", "volume_norm"
    ]
    X = df[feature_cols].values
    y = df["slippage_bps"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return model, scaler, r2, rmse, (X_test, y_test, pred)


def ml_optimized_schedule(
    total_shares: float,
    market_states: List[MarketState],
    model,
    scaler,
    n_grid: int = 10,
) -> np.ndarray:
    """
    Greedy ML-optimized schedule: at each interval, pick the participation rate
    that minimizes predicted slippage, subject to completing the full order.
    """
    n = len(market_states)
    schedule = np.zeros(n)
    remaining = total_shares

    for i, state in enumerate(market_states):
        if remaining <= 0:
            break
        intervals_left = n - i

        # Grid search over participation rates
        best_part = None
        best_slip = np.inf

        for part in np.linspace(0.01, 0.5, n_grid):
            size = state.volume * part
            if size > remaining * 2:     # don't overshoot aggressively
                continue

            features = np.array([[
                part, state.volatility, state.spread, state.order_imbalance,
                state.time_of_day, (intervals_left / n), state.volume / 50_000
            ]])
            pred_slip = model.predict(scaler.transform(features))[0]

            # Urgency penalty: if remaining is large and time is running out
            urgency = max(0, remaining / total_shares - intervals_left / n)
            adjusted = pred_slip + urgency * 5

            if adjusted < best_slip:
                best_slip = adjusted
                best_part = part

        if best_part is None:
            best_part = min(0.5, remaining / max(state.volume, 1))

        size = min(state.volume * best_part, remaining)
        schedule[i] = size
        remaining -= size

    # Distribute any remainder to the last interval
    if remaining > 0:
        schedule[-1] += remaining

    return schedule







# Visualization 
def plot_execution_comparison(
    market_states: List[MarketState],
    schedules: Dict[str, np.ndarray],
    results: Dict[str, Dict],
    save_path: str = None,
):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Smart Order Router — Execution Strategy Comparison", fontsize=14, fontweight="bold")

    prices = np.array([s.price for s in market_states])
    volumes = np.array([s.volume for s in market_states])
    times = np.array([s.time_of_day for s in market_states])

    colors = {"TWAP": "#3498db", "VWAP": "#2ecc71", "POV": "#e67e22", "ML": "#e74c3c"}




    # 1. Intraday price
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, prices, color="black", linewidth=1.2, label="Mid price")
    ax1b = ax1.twinx()
    ax1b.bar(times, volumes / 1000, width=1/len(times), alpha=0.2, color="gray")
    ax1b.set_ylabel("Volume (k shares)", fontsize=8)
    ax1.set_xlabel("Time of day"); ax1.set_ylabel("Price")
    ax1.set_title("Simulated Intraday Price & Volume")
    ax1.legend(fontsize=8)

    # 2. Execution schedules (share quantities)

    ax2 = fig.add_subplot(gs[1, 0])
    for name, sched in schedules.items():
        ax2.plot(times, sched, label=name, color=colors.get(name, "gray"), linewidth=1.5)
    ax2.set_xlabel("Time of day"); ax2.set_ylabel("Shares per interval")
    ax2.set_title("Execution Schedules")
    ax2.legend(fontsize=8)



    # 3. Cumulative shares
    ax3 = fig.add_subplot(gs[1, 1])
    for name, sched in schedules.items():
        ax3.plot(times, np.cumsum(sched), label=name, color=colors.get(name, "gray"), linewidth=1.5)
    ax3.set_xlabel("Time of day"); ax3.set_ylabel("Cumulative shares")
    ax3.set_title("Cumulative Execution")
    ax3.legend(fontsize=8)





    # 4. Slippage per interval
    ax4 = fig.add_subplot(gs[2, 0])
    for name, res in results.items():
        exe = res["executed"]
        ax4.plot(exe["interval"] / len(market_states), exe["slippage_bps"],
                 label=name, color=colors.get(name, "gray"), alpha=0.7, linewidth=1.2)
    ax4.set_xlabel("Time of day"); ax4.set_ylabel("Slippage (bps)")
    ax4.set_title("Per-Interval Slippage")
    ax4.legend(fontsize=8)

    # 5. Summary bar chart


    ax5 = fig.add_subplot(gs[2, 1])
    names = list(results.keys())
    is_bps = [results[n]["implementation_shortfall_bps"] for n in names]
    bar_colors = [colors.get(n, "gray") for n in names]
    bars = ax5.bar(names, is_bps, color=bar_colors, alpha=0.85)
    ax5.set_ylabel("Implementation Shortfall (bps)")
    ax5.set_title("Total Cost Comparison")
    for bar, val in zip(bars, is_bps):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()




def plot_slippage_model(test_data, save_path: str = None):
    X_test, y_test, pred = test_data
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ML Slippage Model — Test Set Performance", fontsize=12, fontweight="bold")

    axes[0].scatter(y_test, pred, alpha=0.15, s=5, color="#3498db")
    mn, mx = y_test.min(), y_test.max()
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.2)
    axes[0].set_xlabel("Actual slippage (bps)"); axes[0].set_ylabel("Predicted slippage (bps)")
    axes[0].set_title("Predicted vs Actual")

    residuals = pred - y_test
    axes[1].hist(residuals, bins=60, color="#2ecc71", alpha=0.75, edgecolor="white")
    axes[1].axvline(0, color="red", lw=1.5)
    axes[1].set_xlabel("Residual (bps)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()


#  Main

def run(total_shares: float = 50_000, plot: bool = True):
    print("\n" + "="*60)
    print("  Smart Order Router — ML-Optimized Execution")
    print("="*60 + "\n")

    print("\n Simulating intraday market (390-minute trading day) …")
    market = simulate_intraday_market(n_intervals=390, seed=1)
    print(f"  Price range: ${min(s.price for s in market):.2f} – ${max(s.price for s in market):.2f}")
    print(f"  Total market volume: {sum(s.volume for s in market):,.0f} shares")

    print(f"\n Order size: {total_shares:,.0f} shares "
          f"({100*total_shares/sum(s.volume for s in market):.1f}% of daily volume)")

    print("\n Generating training data for slippage model …")
    train_df = generate_training_data(n_scenarios=1500)
    print(f"  {len(train_df):,} samples generated")


    print(" Training GBM slippage predictor …")
    model, scaler, r2, rmse, test_data = train_slippage_model(train_df)
    print(f"  Slippage model: R²={r2:.4f}, RMSE={rmse:.4f} bps")

    print("\n Building execution schedules …")
    twap  = twap_schedule(total_shares, len(market))
    vwap  = vwap_schedule(total_shares, market)
    pov   = pov_schedule(total_shares, market, pov_rate=0.10)
    ml_sched = ml_optimized_schedule(total_shares, market, model, scaler)

    schedules = {"TWAP": twap, "VWAP": vwap, "POV": pov, "ML": ml_sched}

    print("\n Simulating execution and measuring costs …")
    results = {name: execute_schedule(sched, market) for name, sched in schedules.items()}

    print(f"\n{'─'*60}")
    print(f"  {'Strategy':<10} {'IS (bps)':>10} {'Avg Slip (bps)':>15} {'Avg Price':>12}")
    print(f"{'─'*60}")
    for name, res in results.items():
        print(f"  {name:<10} {res['implementation_shortfall_bps']:>10.2f} "
              f"{res['avg_slippage_bps']:>15.2f} "
              f"${res['avg_exec_price']:>10.4f}")

    best = min(results, key=lambda n: results[n]["implementation_shortfall_bps"])
    print(f"{'─'*60}")
    print(f"  Best strategy: {best} "
          f"({results[best]['implementation_shortfall_bps']:.2f} bps IS)\n")

    if plot:
        print(" Generating plots …")
        plot_execution_comparison(market, schedules, results,
                                  save_path="order_router_comparison.png")
        plot_slippage_model(test_data, save_path="slippage_model_performance.png")

    return results, model


if __name__ == "__main__":
    run(total_shares=50_000, plot=True)
