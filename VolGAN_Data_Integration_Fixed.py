#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Data Integration Module for Enhanced VolGAN
Handles loading and preprocessing of OptionMetrics data files
Corrected for 25x20 grid structure
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import os
from tqdm import tqdm

# Import original VolGAN functions
import VolGAN

# Import enhanced modules
from VolGAN_Enhanced import (
    MarketRegimeDetector,
    AdaptiveScenarioReweighter,
    DynamicTransactionCostModel,
    MultiTimescaleOptimizer,
    OnlineLearningHedgeOptimizer,
    EnhancedVolGANHedger
)


class VolGANDataLoader:
    """
    Handles loading and preprocessing of OptionMetrics data for VolGAN
    Fixed for 25x20 grid structure
    """
    
    def __init__(self, datapath, surfacepath, m=None, tau=None):
        """
        Parameters:
        -----------
        datapath : str
            Path to directory containing data.csv (raw OptionMetrics data)
        surfacepath : str
            Path to directory containing surfaces_transform.csv (preprocessed surfaces)
        m : np.array, optional
            Moneyness grid (will be inferred from data if not provided)
        tau : np.array, optional
            Time to maturity grid (will be inferred from data if not provided)
        """
        self.datapath = datapath
        self.surfacepath = surfacepath
        
        # Load the data files
        self.raw_data = None
        self.surfaces_data = None
        self.m = m
        self.tau = tau
        
        # Store additional metadata
        self.spot_prices = None
        self.risk_free_rates = None
        
        self._load_data()
        
    def _load_data(self):
        """Load the CSV files"""
        # Load raw OptionMetrics data if available
        data_file = os.path.join(self.datapath, 'data.csv')
        if os.path.exists(data_file):
            print(f"Loading raw data from {data_file}...")
            self.raw_data = pd.read_csv(data_file)
            
            # Parse date column if exists
            if 'date' in self.raw_data.columns:
                self.raw_data['date'] = pd.to_datetime(self.raw_data['date'])
                self.raw_data.set_index('date', inplace=True)
        else:
            print(f"Note: {data_file} not found (optional)")
        
        # Load preprocessed surfaces
        surface_file = os.path.join(self.surfacepath, 'surfaces_transform.csv')
        if os.path.exists(surface_file):
            print(f"Loading surface data from {surface_file}...")
            self.surfaces_data = pd.read_csv(surface_file)
            
            # Parse date column
            if 'date' in self.surfaces_data.columns:
                self.surfaces_data['date'] = pd.to_datetime(self.surfaces_data['date'])
                self.surfaces_data.set_index('date', inplace=True)
            
            # Extract spot prices and risk-free rates if available
            if 'spot_price' in self.surfaces_data.columns:
                # Handle the case where spot_price might be a string representation of array
                self.spot_prices = self.surfaces_data['spot_price'].apply(
                    lambda x: float(str(x).strip('[]').split(',')[0].replace('np.float64(', '').replace(')', ''))
                    if '[' in str(x) else float(x)
                )
            
            if 'risk_free_rate' in self.surfaces_data.columns:
                self.risk_free_rates = self.surfaces_data['risk_free_rate']
            
            # Infer grid dimensions from column names
            self._infer_grid_from_columns()
        else:
            raise FileNotFoundError(f"Surface file {surface_file} not found!")
    
    def _infer_grid_from_columns(self):
        """Infer moneyness and maturity grids from column names"""
        if self.surfaces_data is not None:
            # Get surface columns (those matching m_XX_tau_YY pattern)
            surface_cols = [col for col in self.surfaces_data.columns 
                          if col.startswith('m_') and '_tau_' in col]
            
            if len(surface_cols) == 0:
                raise ValueError("No surface columns found in data!")
            
            # Parse column names to find unique m and tau indices
            m_indices = set()
            tau_indices = set()
            
            for col in surface_cols:
                parts = col.split('_')
                if len(parts) == 4:  # m_XX_tau_YY
                    m_idx = int(parts[1])
                    tau_idx = int(parts[3])
                    m_indices.add(m_idx)
                    tau_indices.add(tau_idx)
            
            n_m = len(m_indices)
            n_tau = len(tau_indices)
            
            print(f"Detected grid structure: {n_m} moneyness x {n_tau} maturity points")
            print(f"Total surface points: {n_m * n_tau} (found {len(surface_cols)} columns)")
            
            # Set default grids if not provided
            if self.m is None:
                # Default moneyness grid (25 points from 0.5 to 1.5)
                self.m = np.linspace(0.5, 1.5, n_m)
                print(f"Using default moneyness grid: {self.m[0]:.2f} to {self.m[-1]:.2f}")
            
            if self.tau is None:
                # Default maturity grid (20 points from 7 to 365 days)
                tau_days = np.linspace(7, 365, n_tau)
                self.tau = tau_days / 365  # Convert to years
                print(f"Using default maturity grid: {tau_days[0]:.0f} to {tau_days[-1]:.0f} days")
    
    def get_surface_vector(self, date):
        """
        Get flattened IV surface vector for a specific date
        
        Parameters:
        -----------
        date : datetime or str
            Date to retrieve surface for
        
        Returns:
        --------
        np.array : Flattened IV surface vector
        """
        if self.surfaces_data is None:
            raise ValueError("Surface data not loaded")
        
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        if date not in self.surfaces_data.index:
            # Find closest date
            closest_date = self.surfaces_data.index[
                self.surfaces_data.index.get_indexer([date], method='nearest')[0]
            ]
            print(f"Date {date} not found, using closest date: {closest_date}")
            date = closest_date
        
        # Get only the surface columns in the correct order
        surface_cols = []
        for m_idx in range(len(self.m)):
            for tau_idx in range(len(self.tau)):
                col_name = f"m_{m_idx:02d}_tau_{tau_idx:02d}"
                if col_name in self.surfaces_data.columns:
                    surface_cols.append(col_name)
        
        if len(surface_cols) != len(self.m) * len(self.tau):
            print(f"Warning: Expected {len(self.m) * len(self.tau)} columns, found {len(surface_cols)}")
        
        return self.surfaces_data.loc[date, surface_cols].values.astype(np.float64)
    
    def get_surface_matrix(self, date):
        """
        Get IV surface as matrix for a specific date
        
        Parameters:
        -----------
        date : datetime or str
            Date to retrieve surface for
        
        Returns:
        --------
        np.array : IV surface matrix (moneyness x maturity)
        """
        vector = self.get_surface_vector(date)
        return self.detangle_surface(vector)
    
    def detangle_surface(self, surface_vector):
        """
        Convert flattened surface vector to matrix form
        
        Parameters:
        -----------
        surface_vector : np.array
            Flattened IV surface
        
        Returns:
        --------
        np.array : Surface matrix (moneyness x maturity)
        """
        if self.m is None or self.tau is None:
            raise ValueError("Grid dimensions not set")
        
        expected_size = len(self.m) * len(self.tau)
        if len(surface_vector) != expected_size:
            print(f"Warning: Vector size {len(surface_vector)} doesn't match expected {expected_size}")
            print(f"Grid: {len(self.m)} moneyness x {len(self.tau)} maturity")
            
            # Try to reshape anyway if it's close
            if len(surface_vector) == 500 and expected_size == 500:
                # Standard 25x20 grid
                return surface_vector.reshape(25, 20)
        
        return surface_vector.reshape(len(self.m), len(self.tau))
    
    def entangle_surface(self, surface_matrix):
        """
        Convert surface matrix to flattened vector
        
        Parameters:
        -----------
        surface_matrix : np.array
            IV surface matrix
        
        Returns:
        --------
        np.array : Flattened surface vector
        """
        return surface_matrix.flatten()
    
    def calculate_vix_proxy(self, date):
        """
        Calculate VIX proxy from IV surface
        
        Parameters:
        -----------
        date : datetime or str
            Date to calculate VIX for
        
        Returns:
        --------
        float : VIX estimate
        """
        surface_matrix = self.get_surface_matrix(date)
        
        # Find ATM volatility at 30 days
        m_idx = np.argmin(np.abs(self.m - 1.0))  # ATM
        
        # Find closest to 30 days
        tau_30_days = 30 / 365
        tau_idx = np.argmin(np.abs(self.tau - tau_30_days))
        
        atm_vol = surface_matrix[m_idx, tau_idx]
        
        # Simple VIX approximation (multiply by 100 to get percentage)
        vix = atm_vol * 100
        
        return vix
    
    def get_spot_price(self, date):
        """
        Get underlying spot price for a date
        
        Parameters:
        -----------
        date : datetime or str
            Date to get spot price for
        
        Returns:
        --------
        float : Spot price
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # First try from surfaces_transform.csv
        if self.spot_prices is not None and date in self.spot_prices.index:
            return self.spot_prices.loc[date]
        
        # Then try from raw data
        if self.raw_data is not None and date in self.raw_data.index:
            if 'spot_price' in self.raw_data.columns:
                return self.raw_data.loc[date, 'spot_price']
            elif 'underlying_price' in self.raw_data.columns:
                return self.raw_data.loc[date, 'underlying_price']
        
        # Return default if not found
        print(f"Warning: Spot price not found for {date}, using default 4500")
        return 4500.0
    
    def get_risk_free_rate(self, date):
        """
        Get risk-free rate for a date
        
        Parameters:
        -----------
        date : datetime or str
            Date to get risk-free rate for
        
        Returns:
        --------
        float : Risk-free rate
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        if self.risk_free_rates is not None and date in self.risk_free_rates.index:
            return self.risk_free_rates.loc[date]
        
        # Default risk-free rate
        return 0.02
    
    def get_data_info(self):
        """
        Get summary information about loaded data
        
        Returns:
        --------
        dict : Data summary
        """
        info = {
            'has_surfaces': self.surfaces_data is not None,
            'has_raw_data': self.raw_data is not None,
            'grid_shape': (len(self.m), len(self.tau)) if self.m is not None else None,
            'moneyness_range': (self.m.min(), self.m.max()) if self.m is not None else None,
            'maturity_range_days': (self.tau.min() * 365, self.tau.max() * 365) if self.tau is not None else None
        }
        
        if self.surfaces_data is not None:
            info['date_range'] = (self.surfaces_data.index.min(), self.surfaces_data.index.max())
            info['total_days'] = len(self.surfaces_data)
            info['has_spot_prices'] = self.spot_prices is not None
            info['has_risk_free_rates'] = self.risk_free_rates is not None
        
        return info


# Keep the rest of the EnhancedVolGANDataIntegration class the same
class EnhancedVolGANDataIntegration:
    """
    Complete integration of enhanced VolGAN with OptionMetrics data
    """
    
    def __init__(self, gen_model, disc_model, datapath, surfacepath, 
                 m=None, tau=None, device='cpu'):
        """
        Parameters:
        -----------
        gen_model : torch.nn.Module
            Trained VolGAN generator
        disc_model : torch.nn.Module
            Trained VolGAN discriminator
        datapath : str
            Path to directory with data.csv
        surfacepath : str
            Path to directory with surfaces_transform.csv
        m : np.array, optional
            Moneyness grid
        tau : np.array, optional
            Maturity grid
        device : str
            Computation device
        """
        # Load data
        self.data_loader = VolGANDataLoader(datapath, surfacepath, m, tau)
        
        # Set grids
        self.m = self.data_loader.m
        self.tau = self.data_loader.tau
        
        # Store models
        self.gen = gen_model
        self.disc = disc_model
        self.device = device
        
        # Initialize enhanced components
        self.regime_detector = MarketRegimeDetector()
        self.scenario_reweighter = AdaptiveScenarioReweighter()
        self.cost_model = DynamicTransactionCostModel()
        self.timescale_optimizer = MultiTimescaleOptimizer()
        self.online_learner = OnlineLearningHedgeOptimizer()
        
        # Print data info
        info = self.data_loader.get_data_info()
        print(f"Enhanced VolGAN initialized with data from {datapath}")
        print(f"Grid: {info['grid_shape'][0]}x{info['grid_shape'][1]} points")
        if info.get('date_range'):
            print(f"Date range: {info['date_range'][0].date()} to {info['date_range'][1].date()}")
    
    def prepare_condition_from_date(self, date):
        """
        Prepare condition tensor from historical data for a specific date
        
        Parameters:
        -----------
        date : datetime or str
            Date to prepare condition for
        
        Returns:
        --------
        torch.Tensor : Condition tensor for VolGAN
        """
        # Get surface vector
        surface_vector = self.data_loader.get_surface_vector(date)
        
        # Convert to tensor
        condition = torch.tensor(surface_vector, dtype=torch.float32).to(self.device)
        
        return condition.unsqueeze(0)  # Add batch dimension
    
    def run_adaptive_hedging(self, date, n_scenarios=1000, target_position=1000000):
        """
        Run complete adaptive hedging for a specific date
        
        Parameters:
        -----------
        date : datetime or str
            Date to hedge for
        n_scenarios : int
            Number of scenarios to generate
        target_position : float
            Position size to hedge
        
        Returns:
        --------
        dict : Hedging results
        """
        # Prepare condition
        condition = self.prepare_condition_from_date(date)
        
        # Calculate VIX
        vix = self.data_loader.calculate_vix_proxy(date)
        
        # Get spot price and risk-free rate
        spot = self.data_loader.get_spot_price(date)
        rf_rate = self.data_loader.get_risk_free_rate(date)
        
        # Detect regime
        regime = self.regime_detector.detect_regime(vix)
        
        print(f"\nDate: {date}")
        print(f"VIX: {vix:.1f}, Spot: {spot:.2f}, RF: {rf_rate:.4f}, Regime: {regime}")
        
        # Generate scenarios with regime-based reweighting
        scenarios, returns, weights = self._generate_adaptive_scenarios(
            condition, n_scenarios, vix, regime
        )
        
        # Perform adaptive LASSO hedging
        hedge_weights, alpha = self._adaptive_lasso_hedging(
            scenarios, returns, weights, regime
        )
        
        # Calculate costs
        transaction_costs = self._calculate_costs(hedge_weights, regime, vix)
        
        results = {
            'date': date,
            'regime': regime,
            'vix': vix,
            'spot': spot,
            'risk_free_rate': rf_rate,
            'alpha': alpha,
            'hedge_weights': hedge_weights,
            'n_instruments': np.sum(np.abs(hedge_weights) > 1e-4),
            'transaction_costs': transaction_costs,
            'scenarios': scenarios,
            'returns': returns,
            'weights': weights
        }
        
        return results
    
    def _generate_adaptive_scenarios(self, condition, n_scenarios, vix, regime):
        """Generate scenarios with adaptive reweighting"""
        # Check if generator exists and has proper attributes
        if self.gen is None:
            # Create mock scenarios for testing
            print("Warning: No generator model, creating mock scenarios")
            surfaces_np = np.random.randn(n_scenarios, len(self.m) * len(self.tau)) * 0.1 + 0.2
            returns = np.random.randn(n_scenarios) * 0.02
        else:
            with torch.no_grad():
                # Determine noise dimension
                if hasattr(self.gen, 'noise_dim'):
                    noise_dim = self.gen.noise_dim
                else:
                    noise_dim = 100  # Default
                
                # Generate base scenarios
                noise = torch.randn(n_scenarios, noise_dim).to(self.device)
                condition_repeated = condition.repeat(n_scenarios, 1)
                
                # Generate surfaces
                generated_surfaces = self.gen(noise, condition_repeated)
                
            # Convert to numpy
            surfaces_np = generated_surfaces.cpu().numpy()
            
            # Calculate returns from surfaces
            returns = self._calculate_returns_from_surfaces(surfaces_np)
        
        # Apply regime-based reweighting
        weights = self.scenario_reweighter.compute_regime_weights(
            surfaces_np, returns, regime, vix
        )
        
        # Check for regime transition
        transition_info = self.regime_detector.predict_regime_transition()
        
        if transition_info['transition_prob'] > 0.3:
            # Generate additional transition scenarios
            n_transition = int(n_scenarios * transition_info['transition_prob'])
            
            if self.gen is not None:
                # Modify condition for stressed scenario
                stressed_condition = condition * 1.5  # Increase all vols
                
                with torch.no_grad():
                    noise_transition = torch.randn(n_transition, noise_dim).to(self.device)
                    condition_transition = stressed_condition.repeat(n_transition, 1)
                    transition_surfaces = self.gen(noise_transition, condition_transition)
                
                # Combine scenarios
                all_surfaces = np.vstack([surfaces_np, transition_surfaces.cpu().numpy()])
                all_returns = np.hstack([returns, self._calculate_returns_from_surfaces(
                    transition_surfaces.cpu().numpy()
                )])
            else:
                # Mock stressed scenarios
                stressed_surfaces = np.random.randn(n_transition, len(self.m) * len(self.tau)) * 0.15 + 0.25
                stressed_returns = np.random.randn(n_transition) * 0.04
                
                all_surfaces = np.vstack([surfaces_np, stressed_surfaces])
                all_returns = np.hstack([returns, stressed_returns])
            
            # Recompute weights
            weights = self.scenario_reweighter.compute_regime_weights(
                all_surfaces, all_returns, regime, vix
            )
            
            return all_surfaces, all_returns, weights
        
        return surfaces_np, returns, weights
    
    def _calculate_returns_from_surfaces(self, surfaces):
        """Calculate implied returns from generated surfaces"""
        n_scenarios = surfaces.shape[0]
        returns = np.zeros(n_scenarios)
        
        for i in range(n_scenarios):
            # Reshape surface
            surface = surfaces[i].reshape(len(self.m), len(self.tau))
            
            # Get ATM short-term vol
            m_idx = np.argmin(np.abs(self.m - 1.0))
            tau_idx = 0
            
            atm_vol = surface[m_idx, tau_idx]
            
            # Generate return from vol
            returns[i] = np.random.normal(0, atm_vol / np.sqrt(252))
        
        return returns
    
    def _adaptive_lasso_hedging(self, scenarios, returns, weights, regime):
        """Perform adaptive LASSO with regime-specific parameters"""
        # Get regime parameters
        regime_params = self.regime_detector.get_regime_parameters(regime)
        base_alpha = regime_params['alpha']
        
        # Get suggested alpha from online learning
        if self.online_learner.is_fitted:
            current_conditions = {
                'regime': regime,
                'expected_instruments': regime_params['instruments'],
                'predicted_var': np.var(returns),
                'expected_costs': 0.001
            }
            alpha = self.online_learner.suggest_alpha(current_conditions)
        else:
            alpha = base_alpha
        
        # Get instrument menu based on regime
        instrument_menu = self._get_regime_instruments(regime)
        
        # Calculate option payoffs
        n_instruments = len(instrument_menu['strikes']) + 1  # +1 for underlying
        
        # Simplified hedging (in practice, use actual LASSO from VolGAN)
        hedge_weights = np.zeros(n_instruments)
        
        if regime == 'calm':
            # Mostly underlying
            hedge_weights[0] = 0.8
            hedge_weights[1] = 0.2  # ATM option
        elif regime == 'normal':
            # Balanced
            hedge_weights[0] = 0.6
            hedge_weights[1:3] = 0.2
        elif regime == 'stressed':
            # More options
            hedge_weights[0] = 0.4
            hedge_weights[1:5] = 0.15
        else:  # crisis
            # Many instruments
            hedge_weights[0] = 0.3
            hedge_weights[1:8] = 0.1
        
        # Normalize
        hedge_weights = hedge_weights / hedge_weights.sum()
        
        return hedge_weights, alpha
    
    def _get_regime_instruments(self, regime):
        """Get appropriate instruments for regime"""
        if regime == 'calm':
            return {'strikes': [0.95, 1.0, 1.05], 'maturities': [7, 30]}
        elif regime == 'normal':
            return {'strikes': [0.90, 0.95, 1.0, 1.05, 1.10], 'maturities': [7, 30, 60]}
        elif regime == 'stressed':
            return {'strikes': [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15], 
                   'maturities': [7, 30, 60, 90]}
        else:  # crisis
            return {'strikes': np.arange(0.70, 1.31, 0.05).tolist(), 
                   'maturities': [7, 14, 30, 60, 90, 180]}
    
    def _calculate_costs(self, hedge_weights, regime, vix):
        """Calculate transaction costs"""
        instrument_menu = self._get_regime_instruments(regime)
        total_cost = 0
        
        # Underlying cost
        total_cost += abs(hedge_weights[0]) * self.cost_model.get_bid_ask_spread(
            'stock', regime, vix
        )
        
        # Option costs
        for i, strike in enumerate(instrument_menu['strikes']):
            if i+1 < len(hedge_weights):
                weight = hedge_weights[i+1]
                if abs(weight) > 1e-6:
                    # Classify option
                    if abs(strike - 1.0) < 0.03:
                        inst_type = 'atm_option'
                    elif abs(strike - 1.0) < 0.10:
                        inst_type = 'otm_option'
                    else:
                        inst_type = 'deep_otm_option'
                    
                    spread = self.cost_model.get_bid_ask_spread(inst_type, regime, vix)
                    total_cost += abs(weight) * spread
        
        return total_cost
    
    def backtest_adaptive_hedging(self, start_date, end_date, rebalance_freq='daily'):
        """
        Run backtest with adaptive hedging using historical data
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date for backtest
        end_date : str or datetime
            End date for backtest
        rebalance_freq : str
            Rebalancing frequency ('daily', 'weekly', 'adaptive')
        
        Returns:
        --------
        pd.DataFrame : Backtest results
        """
        if self.data_loader.surfaces_data is None:
            raise ValueError("No surface data loaded for backtest")
        
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter dates
        backtest_dates = self.data_loader.surfaces_data.index[
            (self.data_loader.surfaces_data.index >= start_date) & 
            (self.data_loader.surfaces_data.index <= end_date)
        ]
        
        print(f"\nRunning backtest from {start_date.date()} to {end_date.date()}")
        print(f"Total trading days: {len(backtest_dates)}")
        
        results = []
        
        for i, date in enumerate(tqdm(backtest_dates, desc="Backtesting")):
            # Skip first day
            if i == 0:
                continue
            
            # Check rebalancing
            should_rebalance = False
            if rebalance_freq == 'daily':
                should_rebalance = True
            elif rebalance_freq == 'weekly' and i % 5 == 0:
                should_rebalance = True
            elif rebalance_freq == 'adaptive':
                # Use regime-based rebalancing
                vix = self.data_loader.calculate_vix_proxy(date)
                regime = self.regime_detector.detect_regime(vix)
                
                portfolio = {'value': 1000000, 'delta': 1.0, 'vega': 0.01}
                should_rebalance = self.timescale_optimizer.should_rebalance(
                    portfolio, regime
                )
            
            if should_rebalance:
                # Run adaptive hedging
                hedge_results = self.run_adaptive_hedging(date, n_scenarios=500)  # Reduced for speed
                
                # Calculate realized performance (would need next day's data)
                if i < len(backtest_dates) - 1:
                    next_date = backtest_dates[i + 1]
                    next_spot = self.data_loader.get_spot_price(next_date)
                    current_spot = hedge_results['spot']
                    
                    realized_return = (next_spot / current_spot) - 1
                    
                    # Update online learner
                    self.online_learner.record_performance(
                        date=date,
                        alpha_used=hedge_results['alpha'],
                        instruments_selected=hedge_results['n_instruments'],
                        predicted_var=np.var(hedge_results['returns']),
                        realized_var=realized_return ** 2,
                        transaction_costs=hedge_results['transaction_costs']
                    )
                    
                    results.append({
                        'date': date,
                        'regime': hedge_results['regime'],
                        'vix': hedge_results['vix'],
                        'alpha': hedge_results['alpha'],
                        'n_instruments': hedge_results['n_instruments'],
                        'transaction_costs': hedge_results['transaction_costs'],
                        'realized_return': realized_return,
                        'predicted_std': np.std(hedge_results['returns']),
                        'realized_std': abs(realized_return)
                    })
        
        results_df = pd.DataFrame(results)
        
        # Calculate performance metrics
        if len(results_df) > 0:
            tracking_error = np.mean(np.abs(
                results_df['predicted_std'] - results_df['realized_std']
            ))
            total_costs = results_df['transaction_costs'].sum()
            
            print(f"\nBacktest Results:")
            print(f"Average Tracking Error: {tracking_error:.4f}")
            print(f"Total Transaction Costs: {total_costs:.4f}")
            print(f"Average Instruments Used: {results_df['n_instruments'].mean():.1f}")
            
            # Results by regime
            print("\nPerformance by Regime:")
            for regime in results_df['regime'].unique():
                regime_data = results_df[results_df['regime'] == regime]
                print(f"  {regime}: {len(regime_data)} days, "
                      f"{regime_data['n_instruments'].mean():.1f} instruments")
        
        return results_df


# Utility functions remain the same but updated for the fixed loader
def test_data_loading(datapath, surfacepath):
    """
    Test function to verify data loading works correctly
    """
    print("Testing Data Loading")
    print("=" * 60)
    
    try:
        # Create data loader
        loader = VolGANDataLoader(datapath, surfacepath)
        
        # Print info
        info = loader.get_data_info()
        print(f"Data loaded successfully!")
        print(f"Grid shape: {info['grid_shape']}")
        print(f"Date range: {info['date_range'][0]} to {info['date_range'][1]}")
        print(f"Total days: {info['total_days']}")
        
        # Test getting a surface
        if loader.surfaces_data is not None:
            test_date = loader.surfaces_data.index[0]
            
            # Get surface vector
            surface_vector = loader.get_surface_vector(test_date)
            print(f"\nSurface vector shape: {surface_vector.shape}")
            
            # Get surface matrix
            surface_matrix = loader.get_surface_matrix(test_date)
            print(f"Surface matrix shape: {surface_matrix.shape}")
            
            # Calculate VIX
            vix = loader.calculate_vix_proxy(test_date)
            print(f"VIX proxy for {test_date.date()}: {vix:.1f}")
            
            # Get spot price
            spot = loader.get_spot_price(test_date)
            print(f"Spot price: {spot:.2f}")
            
            return loader
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test with sample paths
    print("Fixed VolGAN Data Integration Module")
    print("=" * 60)
    
    # Example paths (update these to your actual paths)
    DATAPATH = "./data"
    SURFACEPATH = "./surfaces"
    
    # Test loading
    loader = test_data_loading(DATAPATH, SURFACEPATH)
    
    if loader is not None:
        print("\n✅ Data loading successful! The module is ready to use.")
    else:
        print("\n❌ Data loading failed. Please check your file paths and data format.")
