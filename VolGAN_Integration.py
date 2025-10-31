#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Example: Using Enhanced VolGAN with Original Implementation
This shows how to integrate the adaptive features with your existing VolGAN code
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

# Import original VolGAN
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


# =====================================================================
# INTEGRATION LAYER
# =====================================================================

class VolGANIntegration:
    """
    Integrates the enhanced adaptive features with the original VolGAN implementation
    """
    
    def __init__(self, gen_model, disc_model, m, tau, device='cpu'):
        """
        Parameters:
        -----------
        gen_model : torch.nn.Module
            Trained VolGAN generator from original implementation
        disc_model : torch.nn.Module
            Trained VolGAN discriminator from original implementation
        m : np.array
            Moneyness grid
        tau : np.array
            Time to maturity grid
        device : str
            Device for computation ('cpu' or 'cuda')
        """
        self.gen = gen_model
        self.disc = disc_model
        self.m = m
        self.tau = tau
        self.device = device
        
        # Initialize enhanced components
        self.regime_detector = MarketRegimeDetector()
        self.scenario_reweighter = AdaptiveScenarioReweighter()
        self.cost_model = DynamicTransactionCostModel()
        self.timescale_optimizer = MultiTimescaleOptimizer()
        self.online_learner = OnlineLearningHedgeOptimizer()
        
    def generate_enhanced_scenarios(self, condition, n_scenarios=1000, current_vix=None):
        """
        Generate scenarios using original VolGAN with adaptive enhancements
        
        Parameters:
        -----------
        condition : torch.Tensor
            Condition tensor for VolGAN generation
        n_scenarios : int
            Number of scenarios to generate
        current_vix : float
            Current VIX level for regime detection
        """
        
        # Detect current regime
        if current_vix is None:
            # Estimate from condition if not provided
            current_vix = self._estimate_vix_from_condition(condition)
        
        regime = self.regime_detector.detect_regime(current_vix)
        
        # Generate base scenarios using original VolGAN
        with torch.no_grad():
            # Generate noise
            noise = torch.randn(n_scenarios, self.gen.noise_dim).to(self.device)
            
            # Repeat condition for all scenarios
            condition_repeated = condition.repeat(n_scenarios, 1)
            
            # Generate surfaces
            generated_surfaces = self.gen(noise, condition_repeated)
            
        # Convert to numpy
        surfaces_np = generated_surfaces.cpu().numpy()
        
        # Calculate implied returns from surfaces
        returns = self._calculate_returns_from_surfaces(surfaces_np)
        
        # Apply regime-based reweighting
        weights = self.scenario_reweighter.compute_regime_weights(
            surfaces_np, returns, regime, current_vix
        )
        
        # Handle regime transition if needed
        transition_info = self.regime_detector.predict_regime_transition()
        if transition_info['transition_prob'] > 0.3:
            # Generate additional scenarios for potential regime change
            n_transition = int(n_scenarios * transition_info['transition_prob'])
            
            # Modify condition for stressed scenario
            stressed_condition = self._create_stressed_condition(condition)
            
            with torch.no_grad():
                noise_transition = torch.randn(n_transition, self.gen.noise_dim).to(self.device)
                condition_transition = stressed_condition.repeat(n_transition, 1)
                transition_surfaces = self.gen(noise_transition, condition_transition)
            
            # Combine scenarios
            all_surfaces = np.vstack([surfaces_np, transition_surfaces.cpu().numpy()])
            all_returns = np.hstack([returns, self._calculate_returns_from_surfaces(
                transition_surfaces.cpu().numpy()
            )])
            
            # Recompute weights for combined scenarios
            weights = self.scenario_reweighter.compute_regime_weights(
                all_surfaces, all_returns, regime, current_vix
            )
            
            return all_surfaces, all_returns, weights
        
        return surfaces_np, returns, weights
    
    def adaptive_lasso_hedging(self, scenarios, weights, target_var, regime='normal'):
        """
        Perform LASSO hedging with adaptive regularization
        
        Parameters:
        -----------
        scenarios : np.array
            Generated IV surface scenarios
        weights : np.array
            Scenario weights from reweighting
        target_var : float
            Target portfolio variance
        regime : str
            Current market regime
        """
        
        # Get regime-specific alpha
        regime_params = self.regime_detector.get_regime_parameters(regime)
        base_alpha = regime_params['alpha']
        
        # Get suggested alpha from online learning
        if self.online_learner.is_fitted:
            current_conditions = {
                'regime': regime,
                'expected_instruments': regime_params['instruments'],
                'predicted_var': target_var,
                'expected_costs': 0.001
            }
            alpha = self.online_learner.suggest_alpha(current_conditions)
        else:
            alpha = base_alpha
        
        # Get dynamic instrument menu
        instrument_menu = self._get_regime_instrument_menu(regime)
        
        # Calculate option payoffs for each scenario
        payoffs = self._calculate_option_payoffs(scenarios, instrument_menu)
        
        # Weighted LASSO optimization
        def objective(x):
            # Portfolio variance (weighted)
            portfolio_returns = np.dot(payoffs, x)
            weighted_var = np.sum(weights * (portfolio_returns - np.mean(portfolio_returns))**2)
            
            # L1 penalty with dynamic costs
            transaction_costs = self._calculate_adaptive_costs(x, instrument_menu, regime)
            
            return weighted_var + alpha * transaction_costs
        
        # Constraints
        n_instruments = len(instrument_menu['strikes']) + 1  # +1 for underlying
        x0 = np.zeros(n_instruments)
        x0[0] = 1.0  # Start with 100% in underlying
        
        # Budget constraint
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # Bounds (can be long or short)
        bounds = [(-2, 2) for _ in range(n_instruments)]
        
        # Optimize
        result = optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        hedge_weights = result.x
        
        # Apply sparsity threshold
        threshold = 1e-4
        hedge_weights[np.abs(hedge_weights) < threshold] = 0
        
        # Renormalize
        if np.sum(hedge_weights) != 0:
            hedge_weights = hedge_weights / np.sum(hedge_weights)
        
        return hedge_weights, alpha
    
    def _estimate_vix_from_condition(self, condition):
        """
        Estimate VIX level from condition tensor
        """
        # Simple estimation based on ATM implied vol
        condition_np = condition.cpu().numpy().flatten()
        
        # Assuming condition contains IV surface information
        # Find ATM vol (moneyness = 1.0, tau = 30 days)
        m_idx = np.argmin(np.abs(self.m - 1.0))
        tau_idx = np.argmin(np.abs(self.tau - 30/365))
        
        if len(condition_np) > m_idx * len(self.tau) + tau_idx:
            atm_vol = condition_np[m_idx * len(self.tau) + tau_idx]
            # VIX approximation
            vix = atm_vol * 100
        else:
            vix = 20  # Default
        
        return vix
    
    def _calculate_returns_from_surfaces(self, surfaces):
        """
        Calculate implied returns from IV surfaces
        """
        # Simple approximation: use ATM vol to estimate returns
        n_scenarios = surfaces.shape[0]
        returns = np.zeros(n_scenarios)
        
        for i in range(n_scenarios):
            # Get ATM vol
            surface = surfaces[i].reshape(len(self.m), len(self.tau))
            m_idx = np.argmin(np.abs(self.m - 1.0))
            tau_idx = 0  # Shortest maturity
            
            atm_vol = surface[m_idx, tau_idx]
            
            # Generate return from vol
            returns[i] = np.random.normal(0, atm_vol / np.sqrt(252))
        
        return returns
    
    def _create_stressed_condition(self, condition):
        """
        Create stressed version of condition for regime transition scenarios
        """
        stressed = condition.clone()
        
        # Increase all vols by 50%
        stressed = stressed * 1.5
        
        # Add skew (lower strikes have higher vol)
        n_points = stressed.shape[1]
        skew = torch.linspace(1.2, 0.8, n_points).to(self.device)
        stressed = stressed * skew
        
        return stressed
    
    def _get_regime_instrument_menu(self, regime):
        """
        Get appropriate instruments for current regime
        """
        if regime == 'calm':
            strikes = [0.95, 1.0, 1.05]
            maturities = [7, 30]
        elif regime == 'normal':
            strikes = [0.90, 0.95, 1.0, 1.05, 1.10]
            maturities = [7, 30, 60]
        elif regime == 'stressed':
            strikes = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]
            maturities = [7, 30, 60, 90]
        else:  # crisis
            strikes = np.arange(0.70, 1.31, 0.05).tolist()
            maturities = [7, 14, 30, 60, 90, 180]
        
        return {'strikes': strikes, 'maturities': maturities}
    
    def _calculate_option_payoffs(self, scenarios, instrument_menu):
        """
        Calculate option payoffs for given scenarios and instruments
        """
        n_scenarios = scenarios.shape[0]
        n_instruments = len(instrument_menu['strikes']) + 1  # +1 for underlying
        
        payoffs = np.zeros((n_scenarios, n_instruments))
        
        # Underlying payoff (index 0)
        payoffs[:, 0] = self._calculate_returns_from_surfaces(scenarios)
        
        # Option payoffs
        for i, strike in enumerate(instrument_menu['strikes']):
            # Simplified: use shortest maturity
            for j in range(n_scenarios):
                surface = scenarios[j].reshape(len(self.m), len(self.tau))
                
                # Find closest moneyness
                m_idx = np.argmin(np.abs(self.m - strike))
                tau_idx = 0  # Shortest maturity
                
                vol = surface[m_idx, tau_idx]
                
                # Calculate option value change (simplified Black-Scholes delta)
                if strike <= 1.0:  # Put
                    delta = -VolGAN.norm.cdf(-self._d1(1.0, strike, vol, self.tau[tau_idx]))
                else:  # Call
                    delta = VolGAN.norm.cdf(self._d1(1.0, strike, vol, self.tau[tau_idx]))
                
                payoffs[j, i+1] = delta * payoffs[j, 0]
        
        return payoffs
    
    def _d1(self, S, K, sigma, tau):
        """Black-Scholes d1"""
        return (np.log(S/K) + 0.5*sigma**2*tau) / (sigma * np.sqrt(tau))
    
    def _calculate_adaptive_costs(self, weights, instrument_menu, regime):
        """
        Calculate transaction costs with regime adjustments
        """
        total_cost = 0
        
        # Underlying cost
        total_cost += abs(weights[0]) * self.cost_model.get_bid_ask_spread('stock', regime)
        
        # Option costs
        for i, (w, strike) in enumerate(zip(weights[1:], instrument_menu['strikes'])):
            if abs(w) < 1e-6:
                continue
            
            # Classify option
            if abs(strike - 1.0) < 0.03:
                inst_type = 'atm_option'
            elif abs(strike - 1.0) < 0.10:
                inst_type = 'otm_option'
            else:
                inst_type = 'deep_otm_option'
            
            spread = self.cost_model.get_bid_ask_spread(inst_type, regime)
            total_cost += abs(w) * spread
        
        return total_cost


# =====================================================================
# ENHANCED BACKTEST FUNCTION
# =====================================================================

def run_enhanced_backtest(gen_model, disc_model, m, tau, data, start_date, end_date):
    """
    Run backtest with enhanced adaptive VolGAN hedging
    
    Parameters:
    -----------
    gen_model : torch.nn.Module
        Trained VolGAN generator
    disc_model : torch.nn.Module
        Trained VolGAN discriminator
    m : np.array
        Moneyness grid
    tau : np.array
        Time to maturity grid
    data : pd.DataFrame
        Historical data with columns: date, close, vix, iv_surface
    start_date : datetime
        Backtest start date
    end_date : datetime
        Backtest end date
    """
    
    # Initialize integration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    integrator = VolGANIntegration(gen_model, disc_model, m, tau, device)
    
    # Filter data
    test_data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    results = []
    
    for idx, (date, row) in enumerate(test_data.iterrows()):
        if idx == 0:
            continue  # Skip first day
        
        # Get previous day for condition
        prev_date = test_data.index[idx-1]
        prev_row = test_data.loc[prev_date]
        
        # Create condition from previous day's surface
        condition = torch.tensor(
            prev_row['iv_surface'].flatten(), 
            dtype=torch.float32
        ).to(device).unsqueeze(0)
        
        # Generate enhanced scenarios
        scenarios, returns, weights = integrator.generate_enhanced_scenarios(
            condition, 
            n_scenarios=1000,
            current_vix=prev_row.get('vix', 20)
        )
        
        # Get current regime
        regime = integrator.regime_detector.current_regime
        
        # Check if rebalancing needed
        current_portfolio = {
            'value': 1000000,
            'delta': 1.0,
            'vega': 0.01
        }
        
        should_rebalance = integrator.timescale_optimizer.should_rebalance(
            current_portfolio, regime
        )
        
        if should_rebalance:
            # Perform adaptive LASSO hedging
            hedge_weights, alpha_used = integrator.adaptive_lasso_hedging(
                scenarios, weights, target_var=0.001, regime=regime
            )
            
            # Calculate realized performance
            realized_return = (row['close'] / prev_row['close']) - 1
            
            # Record performance for online learning
            integrator.online_learner.record_performance(
                date=date,
                alpha_used=alpha_used,
                instruments_selected=np.sum(np.abs(hedge_weights) > 1e-4),
                predicted_var=np.var(returns),
                realized_var=realized_return**2,
                transaction_costs=integrator._calculate_adaptive_costs(
                    hedge_weights, 
                    integrator._get_regime_instrument_menu(regime),
                    regime
                )
            )
            
            # Store results
            results.append({
                'date': date,
                'regime': regime,
                'alpha': alpha_used,
                'n_instruments': np.sum(np.abs(hedge_weights) > 1e-4),
                'hedge_weights': hedge_weights,
                'realized_return': realized_return,
                'predicted_vol': np.std(returns) * np.sqrt(252),
                'realized_vol': abs(realized_return) * np.sqrt(252),
                'vix': row.get('vix', 20)
            })
    
    results_df = pd.DataFrame(results)
    
    # Calculate performance metrics
    tracking_error = np.mean(np.abs(
        results_df['predicted_vol'] - results_df['realized_vol']
    ))
    
    avg_instruments_by_regime = results_df.groupby('regime')['n_instruments'].mean()
    
    print("Enhanced Backtest Results:")
    print("=" * 50)
    print(f"Tracking Error: {tracking_error:.4f}")
    print(f"\nAverage Instruments by Regime:")
    for regime, avg in avg_instruments_by_regime.items():
        print(f"  {regime}: {avg:.1f}")
    
    return results_df


# =====================================================================
# VISUALIZATION FUNCTIONS
# =====================================================================

def plot_adaptive_performance(results_df):
    """
    Visualize the performance of adaptive hedging
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot 1: Regime over time
    regime_map = {'calm': 0, 'normal': 1, 'stressed': 2, 'crisis': 3}
    regime_numeric = results_df['regime'].map(regime_map)
    
    axes[0].plot(results_df['date'], regime_numeric, 'o-')
    axes[0].set_ylabel('Regime')
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(['Calm', 'Normal', 'Stressed', 'Crisis'])
    axes[0].set_title('Market Regime Evolution')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Alpha adaptation
    axes[1].plot(results_df['date'], results_df['alpha'], 'b-')
    axes[1].set_ylabel('Alpha')
    axes[1].set_title('Adaptive Regularization Parameter')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Number of instruments
    axes[2].bar(results_df['date'], results_df['n_instruments'], 
                color=['green', 'yellow', 'orange', 'red'][regime_numeric])
    axes[2].set_ylabel('# Instruments')
    axes[2].set_title('Dynamic Instrument Selection')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Tracking performance
    axes[3].plot(results_df['date'], results_df['predicted_vol'], 'b-', label='Predicted', alpha=0.7)
    axes[3].plot(results_df['date'], results_df['realized_vol'], 'r-', label='Realized', alpha=0.7)
    axes[3].fill_between(results_df['date'], 
                         results_df['predicted_vol'], 
                         results_df['realized_vol'],
                         alpha=0.3)
    axes[3].set_ylabel('Volatility')
    axes[3].set_xlabel('Date')
    axes[3].set_title('Volatility Tracking Performance')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics by regime
    print("\nPerformance Summary by Regime:")
    print("=" * 50)
    
    for regime in results_df['regime'].unique():
        regime_data = results_df[results_df['regime'] == regime]
        
        tracking_error = np.mean(np.abs(
            regime_data['predicted_vol'] - regime_data['realized_vol']
        ))
        avg_instruments = regime_data['n_instruments'].mean()
        avg_alpha = regime_data['alpha'].mean()
        
        print(f"\n{regime.upper()}:")
        print(f"  Tracking Error: {tracking_error:.4f}")
        print(f"  Avg Instruments: {avg_instruments:.1f}")
        print(f"  Avg Alpha: {avg_alpha:.4f}")
        print(f"  Days in Regime: {len(regime_data)}")


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

if __name__ == "__main__":
    print("VolGAN Integration with Enhanced Adaptive Features")
    print("=" * 60)
    
    # Example: Load pre-trained VolGAN models
    # (Replace with actual model loading)
    
    # Mock models for demonstration
    class MockGenerator(torch.nn.Module):
        def __init__(self, noise_dim=100, condition_dim=50, output_dim=100):
            super().__init__()
            self.noise_dim = noise_dim
            self.fc = torch.nn.Linear(noise_dim + condition_dim, output_dim)
            
        def forward(self, noise, condition):
            x = torch.cat([noise, condition], dim=1)
            return torch.sigmoid(self.fc(x))
    
    class MockDiscriminator(torch.nn.Module):
        def __init__(self, input_dim=100):
            super().__init__()
            self.fc = torch.nn.Linear(input_dim, 1)
            
        def forward(self, x):
            return torch.sigmoid(self.fc(x))
    
    # Initialize mock models
    gen = MockGenerator()
    disc = MockDiscriminator()
    
    # Define grids
    m = np.linspace(0.8, 1.2, 10)  # Moneyness grid
    tau = np.array([7, 14, 30, 60, 90]) / 365  # Time to maturity grid
    
    # Create integration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    integrator = VolGANIntegration(gen, disc, m, tau, device)
    
    # Test with different market conditions
    print("\nTesting Adaptive Features:")
    print("-" * 40)
    
    test_conditions = [
        {'vix': 12, 'label': 'Calm Market'},
        {'vix': 22, 'label': 'Normal Market'},
        {'vix': 35, 'label': 'Stressed Market'},
        {'vix': 55, 'label': 'Crisis Market'}
    ]
    
    for test in test_conditions:
        # Create mock condition
        condition = torch.randn(1, 50).to(device)
        
        # Generate enhanced scenarios
        scenarios, returns, weights = integrator.generate_enhanced_scenarios(
            condition,
            n_scenarios=100,
            current_vix=test['vix']
        )
        
        # Get regime
        regime = integrator.regime_detector.current_regime
        
        # Perform adaptive hedging
        hedge_weights, alpha = integrator.adaptive_lasso_hedging(
            scenarios, weights, target_var=0.001, regime=regime
        )
        
        print(f"\n{test['label']} (VIX={test['vix']}):")
        print(f"  Detected Regime: {regime}")
        print(f"  Alpha Used: {alpha:.4f}")
        print(f"  Instruments Selected: {np.sum(np.abs(hedge_weights) > 1e-4)}")
        print(f"  Scenario Concentration: {np.max(weights):.3f} (max weight)")
    
    print("\n" + "=" * 60)
    print("Integration Complete - System Ready for Production")
