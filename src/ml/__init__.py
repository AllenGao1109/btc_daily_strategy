"""Machine-learning research package for the BTC daily strategy.

Models here only ever PRODUCE A SIGNAL (a target exposure weight in [-2, 2]).
The existing backtest engine still owns lagging, fees, leverage, funding, and
liquidation. The three inviolable rules apply to every model:

  1. No lookahead  - features for the bar being predicted use only past data;
     the engine lags the produced signal by one day.
  2. No test tuning - models are fit walk-forward (expanding train -> OOS
     predict); the test window is evaluated once, never used for selection.
  3. Same accounting - predictions flow through src.backtest.run_backtest.

Daily BTC is a tiny dataset (~3.4k rows); overfitting risk is severe. Treat any
in-sample edge with suspicion and trust only walk-forward OOS results.
"""
