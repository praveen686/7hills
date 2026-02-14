import os
import logging
import json
import pandas as pd
from datetime import datetime

class TradeLogger:
    """
    Handles logging of trades, signals, and P&L for QuantKubera.
    """
    def __init__(self, log_dir='logs/trading'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.trade_file = os.path.join(log_dir, 'trades.csv')
        self.signal_file = os.path.join(log_dir, 'signals.csv')
        self.pnl_file = os.path.join(log_dir, 'pnl.csv')
        
        self._init_files()
        
        # Standard logger for debug purposes
        self.logger = logging.getLogger('QuantKubera.TradeLogger')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, 'trading.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def _init_files(self):
        """Initialize CSV files with headers if they don't exist."""
        if not os.path.exists(self.trade_file):
            pd.DataFrame(columns=[
                'timestamp', 'ticker', 'side', 'price', 'quantity', 'order_id', 'status'
            ]).to_csv(self.trade_file, index=False)
            
        if not os.path.exists(self.signal_file):
            pd.DataFrame(columns=[
                'timestamp', 'ticker', 'primary_signal', 'confidence', 'final_weight'
            ]).to_csv(self.signal_file, index=False)
            
        if not os.path.exists(self.pnl_file):
            pd.DataFrame(columns=[
                'timestamp', 'daily_pnl', 'total_pnl', 'max_drawdown'
            ]).to_csv(self.pnl_file, index=False)

    def log_signal(self, ticker, primary_signal, confidence, final_weight):
        """Logs a generated trading signal."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'primary_signal': primary_signal,
            'confidence': confidence,
            'final_weight': final_weight
        }
        pd.DataFrame([data]).to_csv(self.signal_file, mode='a', header=False, index=False)
        self.logger.info(f"SIGNAL | {ticker} | Signal: {primary_signal:.4f} | Conf: {confidence:.4f} | Weight: {final_weight:.4f}")

    def log_trade(self, ticker, side, price, quantity, order_id, status='EXECUTED'):
        """Logs an executed trade."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'side': side,
            'price': price,
            'quantity': quantity,
            'order_id': order_id,
            'status': status
        }
        pd.DataFrame([data]).to_csv(self.trade_file, mode='a', header=False, index=False)
        self.logger.info(f"TRADE | {ticker} | {side} | Px: {price} | Qty: {quantity} | ID: {order_id} | {status}")

    def log_pnl(self, daily_pnl, total_pnl, max_drawdown):
        """Logs daily P&L and risk metrics."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'daily_pnl': daily_pnl,
            'total_pnl': total_pnl,
            'max_drawdown': max_drawdown
        }
        pd.DataFrame([data]).to_csv(self.pnl_file, mode='a', header=False, index=False)
        self.logger.info(f"PNL | Daily: {daily_pnl:.2f} | Total: {total_pnl:.2f} | Drawdown: {max_drawdown:.2%}")
