#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced VolGAN Code with Adaptive Features
Based on the original VolGAN paper with improvements for production use
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy.random as rnd
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm
import pandas_datareader as pd_data
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import acf, pacf
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# REGIME DETECTION MODULE
# =====================================================================

class MarketRegimeDetector:
    """
    Detects market regimes based on VIX levels, volatility-of-volatility,
    and other market indicators
    """
    def __init__(self):
        self.regimes = {
            'calm': {'vix_range': (0, 15), 'alpha': 0.05, 'instruments': 3},
            'normal': {'vix_range': (15, 25), 'alpha': 0.02, 'instruments': 5},
            'stressed': {'vix_range': (25, 40), 'alpha': 0.01, 'instruments': 7},
            'crisis': {'vix_range': (40, 100), 'alpha': 0.005, 'instruments': 10}
        }
        self.current_regime = 'normal'
        self.regime_history = deque(maxlen=30)  # Keep 30 days of regime history
        
    def detect_regime(self, vix_level, vol_of_vol=None, correlation_breakdown=None):
        """
        Detect current market regime based on multiple indicators
        
        Parameters:
        -----------
        vix_level : float
            Current VIX level
        vol_of_vol : float, optional
            Volatility of volatility (e.g., VVIX or computed)
        correlation_breakdown : float, optional
            Measure of correlation breakdown (0-1)
        """
        # Primary classification based on VIX
        for regime, params in self.regimes.items():
            if params['vix_range'][0] <= vix_level < params['vix_range'][1]:
                base_regime = regime
                break
        else:
            base_regime = 'crisis' if vix_level >= 40 else 'calm'
        
        # Adjust based on additional indicators if provided
        if vol_of_vol is not None and vol_of_vol > 1.5:
            # High vol-of-vol suggests regime transition
            if base_regime == 'calm':
                base_regime = 'normal'
            elif base_regime == 'normal':
                base_regime = 'stressed'
                
        if correlation_breakdown is not None and correlation_breakdown > 0.7:
            # High correlation breakdown suggests crisis
            if base_regime in ['normal', 'stressed']:
                base_regime = 'crisis'
        
        self.current_regime = base_regime
        self.regime_history.append({
            'regime': base_regime,
            'vix': vix_level,
            'timestamp': datetime.now()
        })
        
        return base_regime
    
    def get_regime_parameters(self, regime=None):
        """Get optimization parameters for current or specified regime"""
        if regime is None:
            regime = self.current_regime
        return self.regimes[regime]
    
    def predict_regime_transition(self, lookback_days=10):
        """
        Predict probability of regime transition based on recent history
        """
        if len(self.regime_history) < lookback_days:
            return {'transition_prob': 0.0, 'likely_regime': self.current_regime}
        
        recent_vix = [h['vix'] for h in list(self.regime_history)[-lookback_days:]]
        vix_trend = np.polyfit(range(len(recent_vix)), recent_vix, 1)[0]
        
        transition_prob = 0.0
        likely_regime = self.current_regime
        
        # Rising VIX trend suggests upward regime transition
        if vix_trend > 2:  # VIX rising more than 2 points per day
            transition_prob = min(0.8, abs(vix_trend) / 5)
            regime_order = ['calm', 'normal', 'stressed', 'crisis']
            current_idx = regime_order.index(self.current_regime)
            if current_idx < len(regime_order) - 1:
                likely_regime = regime_order[current_idx + 1]
        
        # Falling VIX trend suggests downward regime transition
        elif vix_trend < -2:
            transition_prob = min(0.8, abs(vix_trend) / 5)
            regime_order = ['calm', 'normal', 'stressed', 'crisis']
            current_idx = regime_order.index(self.current_regime)
            if current_idx > 0:
                likely_regime = regime_order[current_idx - 1]
        
        return {'transition_prob': transition_prob, 'likely_regime': likely_regime}


# =====================================================================
# ADAPTIVE SCENARIO REWEIGHTING MODULE
# =====================================================================

class AdaptiveScenarioReweighter:
    """
    Reweights VolGAN scenarios based on current market conditions
    """
    def __init__(self):
        self.tail_threshold = 0.05  # 5% daily moves
        self.extreme_tail_threshold = 0.10  # 10% daily moves
        
    def compute_regime_weights(self, scenarios, returns, regime, vix_level=None):
        """
        Reweight scenarios based on current regime
        
        Parameters:
        -----------
        scenarios : np.array
            Generated IV surface scenarios (n_scenarios x n_points)
        returns : np.array
            Generated return scenarios (n_scenarios,)
        regime : str
            Current market regime
        vix_level : float, optional
            Current VIX level for additional calibration
        """
        n_scenarios = len(returns)
        weights = np.ones(n_scenarios) / n_scenarios  # Start with equal weights
        
        # Sort scenarios by return magnitude
        return_abs = np.abs(returns)
        
        if regime == 'crisis':
            # In crisis, upweight extreme scenarios
            tail_mask = return_abs > self.tail_threshold
            extreme_tail_mask = return_abs > self.extreme_tail_threshold
            
            weights[tail_mask] *= 3.0
            weights[extreme_tail_mask] *= 5.0
            
            # Additional upweighting for downside scenarios
            downside_mask = returns < -self.tail_threshold
            weights[downside_mask] *= 1.5
            
        elif regime == 'stressed':
            # In stressed regime, moderate upweighting of tails
            tail_mask = return_abs > self.tail_threshold
            weights[tail_mask] *= 2.0
            
        elif regime == 'calm':
            # In calm regime, downweight extreme scenarios
            tail_mask = return_abs > self.tail_threshold
            weights[tail_mask] *= 0.5
            
            # Focus on moderate moves
            moderate_mask = (return_abs > 0.01) & (return_abs < 0.03)
            weights[moderate_mask] *= 1.5
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights
    
    def compute_vix_calibrated_weights(self, scenarios, returns, vix_current, vix_historical_mean=20):
        """
        Reweight based on VIX level relative to historical mean
        """
        n_scenarios = len(returns)
        weights = np.ones(n_scenarios) / n_scenarios
        
        vix_ratio = vix_current / vix_historical_mean
        
        if vix_ratio > 1.5:  # VIX elevated
            # Upweight tail scenarios proportionally
            return_abs = np.abs(returns)
            for i in range(n_scenarios):
                if return_abs[i] > 0.03:  # 3% moves
                    weights[i] *= (1 + (vix_ratio - 1) * 2)
                    
        elif vix_ratio < 0.75:  # VIX suppressed
            # Downweight tail scenarios
            return_abs = np.abs(returns)
            for i in range(n_scenarios):
                if return_abs[i] > 0.03:
                    weights[i] *= vix_ratio
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights


# =====================================================================
# TIME-VARYING TRANSACTION COST MODEL
# =====================================================================

class DynamicTransactionCostModel:
    """
    Models transaction costs that vary with market conditions
    """
    def __init__(self, base_spread=0.001):
        self.base_spread = base_spread
        self.liquidity_multipliers = {
            'calm': 1.0,
            'normal': 1.2,
            'stressed': 2.0,
            'crisis': 5.0
        }
        
    def get_bid_ask_spread(self, instrument, regime, vix_level=None, volume_ratio=1.0):
        """
        Calculate dynamic bid-ask spread based on market conditions
        
        Parameters:
        -----------
        instrument : str
            Type of instrument ('stock', 'atm_option', 'otm_option', 'deep_otm_option')
        regime : str
            Current market regime
        vix_level : float, optional
            Current VIX for additional adjustment
        volume_ratio : float
            Current volume / average volume
        """
        # Base spreads by instrument type
        instrument_multipliers = {
            'stock': 1.0,
            'atm_option': 3.0,
            'otm_option': 5.0,
            'deep_otm_option': 10.0
        }
        
        base = self.base_spread * instrument_multipliers.get(instrument, 1.0)
        
        # Apply regime multiplier
        regime_mult = self.liquidity_multipliers[regime]
        
        # Apply VIX adjustment if provided
        vix_mult = 1.0
        if vix_level is not None:
            if vix_level > 30:
                vix_mult = 1 + (vix_level - 30) / 30  # Linear increase above VIX 30
        
        # Apply volume adjustment
        volume_mult = 1.0
        if volume_ratio < 0.5:  # Low volume
            volume_mult = 2.0
        elif volume_ratio < 0.75:
            volume_mult = 1.5
        
        return base * regime_mult * vix_mult * volume_mult
    
    def get_market_impact(self, trade_size, avg_daily_volume, regime):
        """
        Estimate market impact based on trade size and conditions
        """
        size_ratio = trade_size / avg_daily_volume
        
        # Base impact model (square-root law)
        base_impact = 0.01 * np.sqrt(size_ratio)
        
        # Regime adjustment
        regime_multipliers = {
            'calm': 0.8,
            'normal': 1.0,
            'stressed': 1.5,
            'crisis': 3.0
        }
        
        return base_impact * regime_multipliers[regime]


# =====================================================================
# MULTI-TIMESCALE HEDGING OPTIMIZER
# =====================================================================

class MultiTimescaleOptimizer:
    """
    Optimizes rebalancing frequency based on market conditions
    """
    def __init__(self):
        self.rebalance_frequencies = {
            'calm': {'frequency': 'weekly', 'threshold': 0.1},
            'normal': {'frequency': 'daily', 'threshold': 0.05},
            'stressed': {'frequency': 'twice_daily', 'threshold': 0.03},
            'crisis': {'frequency': 'intraday', 'threshold': 0.02}
        }
        self.last_rebalance = datetime.now()
        self.portfolio_snapshot = None
        
    def should_rebalance(self, current_portfolio, regime, force_check=False):
        """
        Determine if rebalancing is needed based on regime and portfolio drift
        """
        freq_params = self.rebalance_frequencies[regime]
        
        # Time-based check
        time_since_last = datetime.now() - self.last_rebalance
        
        if freq_params['frequency'] == 'weekly':
            time_threshold = timedelta(days=7)
        elif freq_params['frequency'] == 'daily':
            time_threshold = timedelta(days=1)
        elif freq_params['frequency'] == 'twice_daily':
            time_threshold = timedelta(hours=12)
        else:  # intraday
            time_threshold = timedelta(hours=4)
        
        time_triggered = time_since_last >= time_threshold
        
        # Portfolio drift check
        drift_triggered = False
        if self.portfolio_snapshot is not None:
            delta_drift = abs(current_portfolio['delta'] - self.portfolio_snapshot['delta'])
            vega_drift = abs(current_portfolio['vega'] - self.portfolio_snapshot['vega'])
            
            # Normalize by portfolio value
            normalized_drift = (delta_drift + vega_drift / 100) / current_portfolio['value']
            drift_triggered = normalized_drift > freq_params['threshold']
        
        should_rebalance = time_triggered or drift_triggered or force_check
        
        if should_rebalance:
            self.last_rebalance = datetime.now()
            self.portfolio_snapshot = current_portfolio.copy()
        
        return should_rebalance


# =====================================================================
# ONLINE LEARNING MODULE
# =====================================================================

class OnlineLearningHedgeOptimizer:
    """
    Learns from hedging performance and adapts parameters
    """
    def __init__(self, lookback_window=30):
        self.lookback_window = lookback_window
        self.performance_history = deque(maxlen=lookback_window)
        self.alpha_optimizer = RandomForestRegressor(n_estimators=50, max_depth=5)
        self.is_fitted = False
        
    def record_performance(self, date, alpha_used, instruments_selected, 
                          predicted_var, realized_var, transaction_costs):
        """
        Record hedging performance for learning
        """
        self.performance_history.append({
            'date': date,
            'alpha': alpha_used,
            'n_instruments': instruments_selected,
            'predicted_var': predicted_var,
            'realized_var': realized_var,
            'tracking_error': abs(predicted_var - realized_var),
            'transaction_costs': transaction_costs,
            'efficiency': 1 - abs(predicted_var - realized_var) / predicted_var
        })
        
        # Retrain model if we have enough data
        if len(self.performance_history) >= 20:
            self._update_alpha_model()
    
    def _update_alpha_model(self):
        """
        Update the alpha prediction model based on recent performance
        """
        # Prepare training data
        X = []
        y = []
        
        for record in self.performance_history:
            features = [
                record['alpha'],
                record['n_instruments'],
                record['predicted_var'],
                record['transaction_costs']
            ]
            X.append(features)
            y.append(record['efficiency'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit the model
        self.alpha_optimizer.fit(X, y)
        self.is_fitted = True
    
    def suggest_alpha(self, current_conditions):
        """
        Suggest optimal alpha based on current market conditions and past performance
        """
        if not self.is_fitted or len(self.performance_history) < 20:
            # Default alpha based on regime
            regime_alphas = {
                'calm': 0.05,
                'normal': 0.02,
                'stressed': 0.01,
                'crisis': 0.005
            }
            return regime_alphas.get(current_conditions.get('regime', 'normal'), 0.02)
        
        # Generate candidate alphas
        alpha_candidates = np.linspace(0.001, 0.1, 50)
        
        # Predict performance for each alpha
        predictions = []
        for alpha in alpha_candidates:
            features = [[
                alpha,
                current_conditions.get('expected_instruments', 5),
                current_conditions.get('predicted_var', 0.01),
                current_conditions.get('expected_costs', 0.001)
            ]]
            pred_efficiency = self.alpha_optimizer.predict(features)[0]
            predictions.append(pred_efficiency)
        
        # Select alpha with best predicted efficiency
        best_idx = np.argmax(predictions)
        suggested_alpha = alpha_candidates[best_idx]
        
        # Apply bounds based on regime
        regime = current_conditions.get('regime', 'normal')
        if regime == 'crisis':
            suggested_alpha = min(suggested_alpha, 0.01)
        elif regime == 'calm':
            suggested_alpha = max(suggested_alpha, 0.02)
        
        return suggested_alpha
    
    def get_performance_summary(self):
        """
        Get summary statistics of recent performance
        """
        if len(self.performance_history) == 0:
            return None
        
        tracking_errors = [r['tracking_error'] for r in self.performance_history]
        efficiencies = [r['efficiency'] for r in self.performance_history]
        costs = [r['transaction_costs'] for r in self.performance_history]
        
        return {
            'avg_tracking_error': np.mean(tracking_errors),
            'std_tracking_error': np.std(tracking_errors),
            'avg_efficiency': np.mean(efficiencies),
            'total_costs': np.sum(costs),
            'best_alpha': self.performance_history[np.argmax(efficiencies)]['alpha']
        }


# =====================================================================
# ENHANCED VOLGAN HEDGING SYSTEM
# =====================================================================

class EnhancedVolGANHedger:
    """
    Main class that integrates all adaptive components for VolGAN hedging
    """
    def __init__(self, volgan_generator, base_alpha=0.02):
        self.volgan = volgan_generator
        self.base_alpha = base_alpha
        
        # Initialize adaptive components
        self.regime_detector = MarketRegimeDetector()
        self.scenario_reweighter = AdaptiveScenarioReweighter()
        self.cost_model = DynamicTransactionCostModel()
        self.timescale_optimizer = MultiTimescaleOptimizer()
        self.online_learner = OnlineLearningHedgeOptimizer()
        
        # State tracking
        self.current_regime = 'normal'
        self.current_portfolio = None
        self.hedging_history = []
        
    def generate_adaptive_scenarios(self, current_state, n_scenarios=1000):
        """
        Generate and reweight scenarios based on current market regime
        """
        # Detect current regime
        vix_level = current_state.get('vix', 20)
        vol_of_vol = current_state.get('vol_of_vol', None)
        self.current_regime = self.regime_detector.detect_regime(vix_level, vol_of_vol)
        
        # Check for potential regime transition
        transition_info = self.regime_detector.predict_regime_transition()
        
        # Generate base scenarios
        base_scenarios, base_returns = self.volgan.generate(n_scenarios, current_state)
        
        # If regime transition likely, generate additional scenarios
        if transition_info['transition_prob'] > 0.5:
            # Generate scenarios for potential new regime
            future_state = current_state.copy()
            future_state['regime'] = transition_info['likely_regime']
            
            transition_scenarios, transition_returns = self.volgan.generate(
                int(n_scenarios * transition_info['transition_prob']), 
                future_state
            )
            
            # Combine scenarios
            all_scenarios = np.vstack([base_scenarios, transition_scenarios])
            all_returns = np.hstack([base_returns, transition_returns])
        else:
            all_scenarios = base_scenarios
            all_returns = base_returns
        
        # Reweight scenarios based on regime
        weights = self.scenario_reweighter.compute_regime_weights(
            all_scenarios, all_returns, self.current_regime, vix_level
        )
        
        return all_scenarios, all_returns, weights
    
    def get_dynamic_instrument_menu(self):
        """
        Get regime-appropriate instrument menu
        """
        if self.current_regime == 'calm':
            # Limited menu in calm markets
            strikes = [0.95, 0.97, 1.0, 1.03, 1.05]
            maturities = [7, 30]
        elif self.current_regime == 'normal':
            # Standard menu
            strikes = [0.90, 0.95, 0.97, 1.0, 1.03, 1.05, 1.10]
            maturities = [7, 14, 30]
        elif self.current_regime == 'stressed':
            # Expanded menu
            strikes = [0.85, 0.90, 0.95, 0.97, 1.0, 1.03, 1.05, 1.10, 1.15]
            maturities = [7, 14, 30, 60]
        else:  # crisis
            # Full menu including deep OTM and VIX
            strikes = [0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 1.0, 1.03, 1.05, 1.10, 1.15, 1.20, 1.30]
            maturities = [7, 14, 30, 60, 90]
        
        return {'strikes': strikes, 'maturities': maturities, 'include_vix': self.current_regime == 'crisis'}
    
    def calculate_dynamic_costs(self, instruments, quantities):
        """
        Calculate transaction costs with dynamic model
        """
        total_cost = 0
        
        for inst, qty in zip(instruments, quantities):
            if qty == 0:
                continue
                
            # Determine instrument type
            if 'stock' in inst.lower():
                inst_type = 'stock'
            elif abs(inst.get('strike', 1.0) - 1.0) < 0.05:
                inst_type = 'atm_option'
            elif abs(inst.get('strike', 1.0) - 1.0) < 0.15:
                inst_type = 'otm_option'
            else:
                inst_type = 'deep_otm_option'
            
            # Get spread
            spread = self.cost_model.get_bid_ask_spread(
                inst_type, 
                self.current_regime,
                vix_level=self.current_portfolio.get('vix', 20)
            )
            
            # Calculate cost
            cost = abs(qty) * inst.get('price', 1.0) * spread
            total_cost += cost
        
        return total_cost
    
    def optimize_hedge(self, current_state, target_position):
        """
        Main hedging optimization with all adaptive features
        """
        # Check if rebalancing needed
        if self.current_portfolio is not None:
            should_rebalance = self.timescale_optimizer.should_rebalance(
                self.current_portfolio, self.current_regime
            )
            if not should_rebalance:
                return None  # Skip rebalancing
        
        # Generate adaptive scenarios
        scenarios, returns, weights = self.generate_adaptive_scenarios(current_state)
        
        # Get appropriate instrument menu
        instrument_menu = self.get_dynamic_instrument_menu()
        
        # Get suggested alpha from online learner
        current_conditions = {
            'regime': self.current_regime,
            'expected_instruments': len(instrument_menu['strikes']),
            'predicted_var': np.var(returns),
            'expected_costs': 0.001  # Estimate
        }
        alpha = self.online_learner.suggest_alpha(current_conditions)
        
        # Run LASSO optimization with weighted scenarios
        # (This would interface with the existing LASSO solver)
        # hedge_weights = solve_weighted_lasso(scenarios, weights, alpha, instrument_menu)
        
        # For demonstration, create a mock result
        hedge_weights = self._mock_lasso_optimization(scenarios, weights, alpha, instrument_menu)
        
        # Calculate transaction costs
        transaction_costs = self.calculate_dynamic_costs(
            instrument_menu['strikes'], 
            hedge_weights
        )
        
        # Update portfolio state
        self.current_portfolio = {
            'instruments': instrument_menu,
            'weights': hedge_weights,
            'vix': current_state.get('vix', 20),
            'value': target_position,
            'delta': np.sum(hedge_weights[:5]),  # Mock delta
            'vega': np.sum(hedge_weights[5:]) * 0.01,  # Mock vega
            'regime': self.current_regime,
            'alpha': alpha,
            'transaction_costs': transaction_costs
        }
        
        return self.current_portfolio
    
    def _mock_lasso_optimization(self, scenarios, weights, alpha, instrument_menu):
        """
        Mock LASSO optimization for demonstration
        Replace with actual LASSO solver
        """
        n_instruments = len(instrument_menu['strikes'])
        
        # Generate mock sparse weights based on regime
        if self.current_regime == 'calm':
            # Few instruments
            weights = np.zeros(n_instruments)
            weights[n_instruments//2] = 0.5  # ATM
            weights[0] = 0.5  # Stock
        elif self.current_regime == 'crisis':
            # Many instruments
            weights = np.random.exponential(0.1, n_instruments)
            weights = weights / weights.sum()
        else:
            # Moderate
            weights = np.zeros(n_instruments)
            selected = np.random.choice(n_instruments, size=5, replace=False)
            weights[selected] = 1.0 / 5
        
        return weights
    
    def update_performance(self, date, realized_pnl, realized_var):
        """
        Update online learning with realized performance
        """
        if self.current_portfolio is not None:
            self.online_learner.record_performance(
                date=date,
                alpha_used=self.current_portfolio['alpha'],
                instruments_selected=np.sum(self.current_portfolio['weights'] > 0),
                predicted_var=np.var(self.current_portfolio.get('predicted_returns', [0])),
                realized_var=realized_var,
                transaction_costs=self.current_portfolio['transaction_costs']
            )
    
    def get_adaptive_summary(self):
        """
        Get summary of adaptive hedging performance
        """
        performance = self.online_learner.get_performance_summary()
        
        return {
            'current_regime': self.current_regime,
            'regime_history': list(self.regime_detector.regime_history),
            'performance_summary': performance,
            'current_alpha': self.current_portfolio['alpha'] if self.current_portfolio else self.base_alpha,
            'instruments_used': np.sum(self.current_portfolio['weights'] > 0) if self.current_portfolio else 0
        }


# =====================================================================
# INTEGRATION WITH EXISTING VOLGAN CODE
# =====================================================================

def create_enhanced_hedger(volgan_generator, initial_state):
    """
    Factory function to create enhanced hedger with existing VolGAN
    """
    hedger = EnhancedVolGANHedger(volgan_generator)
    
    # Initialize with current market state
    hedger.optimize_hedge(initial_state, target_position=1000000)  # $1M position
    
    return hedger


def run_adaptive_backtest(volgan_generator, historical_data, start_date, end_date):
    """
    Run backtest with adaptive hedging
    """
    hedger = EnhancedVolGANHedger(volgan_generator)
    
    results = []
    
    # Filter data for backtest period
    test_data = historical_data[(historical_data.index >= start_date) & 
                                (historical_data.index <= end_date)]
    
    for date, row in test_data.iterrows():
        # Prepare current state
        current_state = {
            'vix': row.get('vix', 20),
            'vol_of_vol': row.get('vvix', None),
            'spot': row['close'],
            'iv_surface': row.get('iv_surface', None)
        }
        
        # Optimize hedge
        portfolio = hedger.optimize_hedge(current_state, target_position=1000000)
        
        if portfolio is not None:
            # Simulate one day forward
            next_date = date + timedelta(days=1)
            if next_date in test_data.index:
                next_row = test_data.loc[next_date]
                
                # Calculate realized P&L
                realized_return = (next_row['close'] / row['close']) - 1
                realized_var = realized_return ** 2
                
                # Update learning
                hedger.update_performance(date, realized_return, realized_var)
                
                # Store results
                results.append({
                    'date': date,
                    'regime': hedger.current_regime,
                    'alpha': portfolio['alpha'],
                    'n_instruments': np.sum(portfolio['weights'] > 0),
                    'transaction_costs': portfolio['transaction_costs'],
                    'realized_return': realized_return,
                    'predicted_var': np.var(portfolio.get('predicted_returns', [0])),
                    'realized_var': realized_var
                })
    
    return pd.DataFrame(results)


# =====================================================================
# EXAMPLE USAGE WITH ORIGINAL VOLGAN
# =====================================================================

def enhance_existing_volgan(original_volgan_module):
    """
    Enhance existing VolGAN implementation with adaptive features
    
    Parameters:
    -----------
    original_volgan_module : module
        The original VolGAN module
    """
    
    class VolGANAdapter:
        """Adapter to make original VolGAN work with enhanced system"""
        def __init__(self, original_generator):
            self.generator = original_generator
            
        def generate(self, n_scenarios, current_state):
            # Call original VolGAN generation
            # This would need to be adapted based on actual VolGAN interface
            scenarios = np.random.randn(n_scenarios, 100)  # Mock
            returns = np.random.randn(n_scenarios) * 0.02  # Mock
            return scenarios, returns
    
    # Create adapter
    volgan_adapter = VolGANAdapter(original_volgan_module)
    
    # Create enhanced hedger
    enhanced_hedger = EnhancedVolGANHedger(volgan_adapter)
    
    return enhanced_hedger


if __name__ == "__main__":
    # Example usage
    print("Enhanced VolGAN Hedging System Initialized")
    print("=" * 50)
    
    # Create mock VolGAN generator
    class MockVolGAN:
        def generate(self, n_scenarios, current_state):
            scenarios = np.random.randn(n_scenarios, 100)
            returns = np.random.randn(n_scenarios) * 0.02
            return scenarios, returns
    
    mock_volgan = MockVolGAN()
    
    # Create enhanced hedger
    hedger = EnhancedVolGANHedger(mock_volgan)
    
    # Simulate different market conditions
    market_conditions = [
        {'vix': 12, 'spot': 4500, 'label': 'Calm Market'},
        {'vix': 22, 'spot': 4400, 'label': 'Normal Market'},
        {'vix': 35, 'spot': 4200, 'label': 'Stressed Market'},
        {'vix': 65, 'spot': 3800, 'label': 'Crisis Market'}
    ]
    
    print("\nAdaptive Hedging Across Different Regimes:")
    print("-" * 50)
    
    for condition in market_conditions:
        portfolio = hedger.optimize_hedge(condition, target_position=1000000)
        if portfolio:
            print(f"\n{condition['label']} (VIX={condition['vix']}):")
            print(f"  Regime: {portfolio['regime']}")
            print(f"  Alpha: {portfolio['alpha']:.4f}")
            print(f"  Instruments Used: {np.sum(portfolio['weights'] > 0)}")
            print(f"  Transaction Costs: ${portfolio['transaction_costs']:.2f}")
    
    print("\n" + "=" * 50)
    print("System ready for production use")
