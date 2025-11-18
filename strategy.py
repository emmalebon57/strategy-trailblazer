import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

import cpz
from cpz.execution.models import OrderSubmitRequest
from cpz.execution.enums import OrderSide, OrderType, TimeInForce

#credentials
os.environ["CPZ_AI_API_KEY"] = "cpz_key_bb9411566c9144368e1552cd" 
os.environ["CPZ_AI_API_SECRET"] = "cpz_secret_4es5p4n3a1vc3x5s6v2xl693d2p43ch3u5q6b18x434r1c2z"  
os.environ["CPZ_STRATEGY_ID"] = "bb931b2e-0f67-44e1-933f-6f715038d224"  

#settings
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
HISTORY_DAYS = 365

#kelly Strategy Parameters
KELLY_LONG_THRESHOLD = 0.02      #min for long positions
KELLY_SHORT_THRESHOLD = -0.03    #max for short positions  
KELLY_FRACTION = 0.5             #half criterion used
MAX_POSITION_SIZE = 0.15         #0.15 max per position
MIN_DATA_POINTS = 100            #points for garch/arima

# ==========================
# CPZ AI / Execution Config
# ==========================
PLACE_ORDERS = True  
BROKER = "alpaca"
BROKER_ENVIRONMENT = "paper"  

STRATEGY_ID = os.environ.get("CPZ_STRATEGY_ID", "3080ca94-7598-46fc-9ea3-1159c46362e8")

def get_alpaca_portfolio_value():
    """Get actual portfolio value from Alpaca account"""
    try:
        client = cpz.clients.sync.CPZClient()
        client.execution.use_broker(BROKER, environment=BROKER_ENVIRONMENT)
        account = client.execution.get_account()
        portfolio_value = float(account.buying_power) * 0.75
        print(f"Connected to Alpaca - Buying Power: ${float(account.buying_power):,.2f}")
        print(f" Using Portfolio Value: ${portfolio_value:,.2f}")
        return portfolio_value, client
    except Exception as e:
        print(f"Failed to connect to Alpaca: {e}")
        return None, None

def forecast_arima_garch(prices, forecast_days=1):
    """Forecast next day return and volatility using ARIMA+GARCH"""
    returns = prices.pct_change().dropna()
    
    if len(returns) < MIN_DATA_POINTS:
        return np.nan, np.nan, {"error": "insufficient_data", "data_points": len(returns)}
    
    try:
        # ARIMA for mean forecast
        arima_model = ARIMA(returns, order=(1,0,1))
        arima_fit = arima_model.fit()
        predicted_return = arima_fit.forecast(steps=forecast_days).iloc[0] #only the 1st day
        
        # GARCH for volatility forecast  
        garch_model = arch_model(returns, vol='Garch', p=1, q=1)
        garch_fit = garch_model.fit(disp='off')
        vol_forecast = garch_fit.forecast(horizon=forecast_days)
        predicted_vol = np.sqrt(vol_forecast.variance.values[-1, -1])
        
        diagnostics = { "arima_success": True,"garch_success": True,"data_points": len(returns),"historical_mean": returns.mean(),"historical_vol": returns.std()}
        
        return predicted_return, predicted_vol, diagnostics
        
    except Exception as e:
        print(f"Forecasting error: {e}")
        return np.nan, np.nan, {"error": str(e), "data_points": len(returns)}

def fetch_prices(tickers, period_days=HISTORY_DAYS):
    """Fetch adjusted close prices for tickers using yfinance."""
    end = datetime.now()
    start = end - timedelta(days=period_days)
    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        group_by='ticker',
        threads=True,
    )
    prices = {}
    current_prices = {}

    if isinstance(tickers, str) or len(tickers) == 1:
        t = tickers if isinstance(tickers, str) else tickers[0]
        if 'Adj Close' in data.columns:
            prices[t] = data['Adj Close'].dropna()
            current_prices[t] = data['Adj Close'].iloc[-1]
        elif 'Close' in data.columns:
            prices[t] = data['Close'].dropna()
            current_prices[t] = data['Close'].iloc[-1]
    else:
        for t in tickers:
            try:
                if hasattr(data.columns, "levels") and t in data.columns.levels[0]:
                    ser = data[t].get('Adj Close') if 'Adj Close' in data[t].columns else data[t].get('Close')
                    if ser is not None:
                        prices[t] = ser.dropna()
                        current_prices[t] = ser.iloc[-1]
            except Exception:
                continue

    return prices, current_prices

def arima_garch_kelly_strategy( tickers=TICKERS,place_orders: bool = PLACE_ORDERS,strategy_id: str = STRATEGY_ID,):
    print("ARIMA-GARCH Kelly Strategy with Short Selling")
    #portfolio
    portfolio_value, client = get_alpaca_portfolio_value()

    prices_map, current_prices = fetch_prices(tickers, HISTORY_DAYS)
    signals = {}
    diagnostics_data = {}

    # Calculate signals for all tickers using ARIMA-GARCH
    for t in tickers:
        if t not in prices_map or len(prices_map[t]) < MIN_DATA_POINTS:
            print(f"{t}: insufficient data. Skipping.")
            signals[t] = {
                "signal": 0, 
                "kelly_fraction": np.nan,
                "predicted_return": np.nan,
                "predicted_vol": np.nan
            }
            continue

        series = prices_map[t].sort_index()
        predicted_return, predicted_vol, diagnostics = forecast_arima_garch(series)
        diagnostics_data[t] = diagnostics

        if pd.isna(predicted_return) or pd.isna(predicted_vol) or predicted_vol == 0:
            kelly_fraction = 0
            signal = 0
        else:
            #Kelly criterion
            kelly_fraction = predicted_return / (predicted_vol ** 2)
            
            #thresholds for signals
            if kelly_fraction > KELLY_LONG_THRESHOLD:
                signal = 1  # Buy 
            elif kelly_fraction < KELLY_SHORT_THRESHOLD:
                signal = -1  # Short sell
            else:
                signal = 0

        signals[t] = {
            "signal": int(signal),
            "kelly_fraction": float(kelly_fraction),
            "predicted_return": float(predicted_return) if not pd.isna(predicted_return) else np.nan,
            "predicted_vol": float(predicted_vol) if not pd.isna(predicted_vol) else np.nan,
        }

    # Display model diagnostics
    print("\n=== ARIMA-GARCH Diagnostics ===")
    for t in tickers:
        if t in diagnostics_data:
            diag = diagnostics_data[t]
            if "error" in diag:
                print(f"{t}:  Error - {diag['error']} (data points: {diag.get('data_points', 0)})")
            else:
                status = "good" if diag.get("arima_success", False) and diag.get("garch_success", False) else "wrong"
                print(f"{t}: {status} Data: {diag.get('data_points', 0)} points")
                if signals[t]["predicted_return"] is not np.nan:
                    print(f"     Predicted: μ={signals[t]['predicted_return']:.6f}, σ={signals[t]['predicted_vol']:.6f}")
                    print(f"     Historical: μ={diag.get('historical_mean', 0):.6f}, σ={diag.get('historical_vol', 0):.6f}")
                    print(f"     Kelly Fraction: {signals[t]['kelly_fraction']:.4f}")

    #portfolio allocation logic using Kelly criterion
    long_positions = [t for t in tickers if signals[t]["signal"] == 1]
    short_positions = [t for t in tickers if signals[t]["signal"] == -1]
    active_positions = long_positions + short_positions

    portfolio_allocation = {}

    if active_positions:
        #calculate total Kelly weights for normalization
        total_kelly_long = sum(max(signals[t]["kelly_fraction"], 0) for t in long_positions)
        total_kelly_short = sum(abs(min(signals[t]["kelly_fraction"], 0)) for t in short_positions)

        for t in tickers:
            signal = signals[t]["signal"] 
            current_price = current_prices.get(t, 0)
            kelly_fraction = signals[t]["kelly_fraction"]

            if signal != 0:
                if signal == 1:  # Long position
                    if total_kelly_long > 0:
                        weight = max(kelly_fraction, 0) / total_kelly_long
                    else:
                        weight = 0
                else:  # Short position
                    if total_kelly_short > 0:
                        weight = abs(min(kelly_fraction, 0)) / total_kelly_short
                    else:
                        weight = 0
                
                #apply Kelly fraction
                dollar_allocation = weight * portfolio_value * KELLY_FRACTION
                
                #risk management
                if abs(dollar_allocation) > MAX_POSITION_SIZE * portfolio_value:
                    dollar_allocation = MAX_POSITION_SIZE * portfolio_value if signal == 1 else -MAX_POSITION_SIZE * portfolio_value
                
                shares = dollar_allocation / current_price if current_price > 0 else 0

                portfolio_allocation[t] = {
                    'allocation_pct': (dollar_allocation / portfolio_value) * 100,
                    'dollar_amount': dollar_allocation,
                    'shares': shares if signal == 1 else -shares,
                    'current_price': current_price,
                    'signal': signal,
                    'kelly_fraction': kelly_fraction,
                    'predicted_return': signals[t]['predicted_return'],
                    'predicted_vol': signals[t]['predicted_vol'],
                    'position_type': 'LONG' if signal == 1 else 'SHORT',
                }
            else:
                portfolio_allocation[t] = {
                    'allocation_pct': 0.0,
                    'dollar_amount': 0.0,
                    'shares': 0,
                    'current_price': current_price,
                    'signal': 0,
                    'kelly_fraction': kelly_fraction,
                    'predicted_return': signals[t]['predicted_return'],
                    'predicted_vol': signals[t]['predicted_vol'],
                    'position_type': 'CASH',
                }
    else:
        # No active positions - all cash
        for t in tickers:
            portfolio_allocation[t] = {
                'allocation_pct': 0.0,
                'dollar_amount': 0.0,
                'shares': 0,
                'current_price': current_prices.get(t, 0),
                'signal': 0,
                'kelly_fraction': signals.get(t, {}).get('kelly_fraction', np.nan),
                'predicted_return': signals.get(t, {}).get('predicted_return', np.nan),
                'predicted_vol': signals.get(t, {}).get('predicted_vol', np.nan),
                'position_type': 'CASH',
            }

    #results
    print("\n=== Portfolio Allocation ===")
    total_long = 0
    total_short = 0
    total_shares_long = 0
    total_shares_short = 0

    for t in tickers:
        alloc = portfolio_allocation[t]
        position_value = abs(alloc['dollar_amount'])

        if alloc['signal'] == 1:  # Long
            print(
                f"{t}: {alloc['position_type']:6} | "
                f"Kelly = {alloc['kelly_fraction']:7.4f} | "
                f"Pred Return = {alloc['predicted_return']:7.4f} | "
                f"Pred Vol = {alloc['predicted_vol']:7.4f} | "
                f"Allocation = {alloc['allocation_pct']:6.1f}% | "
                f"Value = ${position_value:>8,.2f} | "
                f"Shares = {alloc['shares']:6.0f} @ ${alloc['current_price']:.2f}"
            )
            total_long += alloc['allocation_pct']
            total_shares_long += alloc['shares']

        elif alloc['signal'] == -1:  # Short
            print(
                f"{t}: {alloc['position_type']:6} | "
                f"Kelly = {alloc['kelly_fraction']:7.4f} | "
                f"Pred Return = {alloc['predicted_return']:7.4f} | "
                f"Pred Vol = {alloc['predicted_vol']:7.4f} | "
                f"Allocation = {alloc['allocation_pct']:6.1f}% | "
                f"Value = ${position_value:>8,.2f} | "
                f"Shares = {alloc['shares']:6.0f} @ ${alloc['current_price']:.2f}"
            )
            total_short += alloc['allocation_pct']
            total_shares_short += abs(alloc['shares'])

        else:  # Cash
            print(
                f"{t}: {alloc['position_type']:6} | "
                f"Kelly = {alloc['kelly_fraction']:7.4f} | "
                f"Pred Return = {alloc['predicted_return']:7.4f} | "
                f"Pred Vol = {alloc['predicted_vol']:7.4f} | "
                f"Allocation = {alloc['allocation_pct']:6.1f}% | "
                f"Value = ${position_value:>8,.2f} | "
                f"Shares = {alloc['shares']:6.0f} @ ${alloc['current_price']:.2f}"
            )

    print(f"\n=== Portfolio Summary ===")
    print(f"Long Positions: {len(long_positions)} | Total Long: {total_long:.1f}% | Total Shares: {total_shares_long:.0f}")
    print(f"Short Positions: {len(short_positions)} | Total Short: {total_short:.1f}% | Total Shares: {total_shares_short:.0f}")
    print(f"Net Exposure: {total_long + total_short:.1f}%")
    print(f"Gross Exposure: {total_long + abs(total_short):.1f}%")
    print(f"Cash: {100 - (total_long + abs(total_short)):.1f}%")

    total_invested = sum(abs(alloc['dollar_amount']) for alloc in portfolio_allocation.values())
    print(f"Total Invested: ${total_invested:,.2f}")
    print(f"Remaining Cash: ${portfolio_value - total_invested:,.2f}")

    # ==========================
    # CPZ Order Placement
    # ==========================
    orders = []
    if place_orders:
        try:
            print("\n=== Submitting Orders via CPZ AI ===")
            for symbol, alloc in portfolio_allocation.items():
                signal = alloc["signal"]
                shares = alloc["shares"]

                if signal == 0 or shares == 0:
                    continue

                qty = int(abs(round(shares)))
                if qty <= 0:
                    continue

                side = OrderSide.BUY if signal == 1 else OrderSide.SELL

                order = client.execution.submit_order(OrderSubmitRequest(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    strategy_id=strategy_id,
                ))

                print(f"{side.name} {qty} {symbol} Order ID: {order.id}, Status: {order.status}")
                orders.append(order)

            if not orders:
                print("No orders submitted (no active signals).")
            else:
                print(f"\n Successfully placed {len(orders)} orders!")
                
        except Exception as e:
            print(f" Error during order placement: {e}")

    return {
        "portfolio_allocation": portfolio_allocation,
        "long_positions": long_positions,
        "short_positions": short_positions,
        "net_exposure_pct": total_long + total_short,
        "gross_exposure_pct": total_long + abs(total_short),
        "total_portfolio_value": portfolio_value,
        "orders": orders,
    }

if __name__ == "__main__":
    results = arima_garch_kelly_strategy()