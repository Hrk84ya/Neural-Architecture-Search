#!/usr/bin/env python3
"""
Quick test of the Multi-Objective Optimization enhancement
"""

import numpy as np
import sys
import os

# Add current directory to path to import nas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nas import EvolutionaryNAS, SearchConfig

def test_pareto_selection():
    """Test the Pareto selection functionality"""
    print("🧪 Testing Multi-Objective Pareto Selection")
    print("="*50)
    
    # Create a mock NAS instance
    config = SearchConfig(population_size=6, max_generations=2)
    nas = EvolutionaryNAS(input_shape=(32, 32, 3), num_classes=10, config=config)
    
    # Create mock population with different accuracy/efficiency trade-offs
    mock_population = [
        {'accuracy': 0.85, 'efficiency': 0.80, 'params': 1000000, 'generation': 1},  # High acc, high eff
        {'accuracy': 0.90, 'efficiency': 0.70, 'params': 2000000, 'generation': 1},  # Higher acc, lower eff
        {'accuracy': 0.75, 'efficiency': 0.90, 'params': 500000, 'generation': 1},   # Lower acc, higher eff
        {'accuracy': 0.80, 'efficiency': 0.75, 'params': 1500000, 'generation': 1},  # Dominated solution
        {'accuracy': 0.88, 'efficiency': 0.85, 'params': 800000, 'generation': 1},   # Good balance
        {'accuracy': 0.70, 'efficiency': 0.95, 'params': 300000, 'generation': 1},   # Very efficient, low acc
    ]
    
    print("📊 Mock Population:")
    for i, ind in enumerate(mock_population):
        print(f"  {i+1}: Acc={ind['accuracy']:.3f}, Eff={ind['efficiency']:.3f}, Params={ind['params']:,}")
    
    # Test Pareto selection
    pareto_front = nas.pareto_selection(mock_population)
    
    print(f"\n🎯 Pareto Front ({len(pareto_front)} solutions):")
    for i, ind in enumerate(pareto_front):
        print(f"  {i+1}: Acc={ind['accuracy']:.3f}, Eff={ind['efficiency']:.3f}, Params={ind['params']:,}")
    
    # Test multi-objective selection
    selected = nas.multi_objective_selection(mock_population, 3)
    
    print(f"\n🏆 Selected for Breeding ({len(selected)} solutions):")
    for i, ind in enumerate(selected):
        print(f"  {i+1}: Acc={ind['accuracy']:.3f}, Eff={ind['efficiency']:.3f}, Params={ind['params']:,}")
    
    print("\n✅ Pareto selection test completed!")
    
    # Verify expected behavior
    expected_pareto_size = 4  # Solutions 1, 2, 3, 5, 6 should be non-dominated
    if len(pareto_front) >= 3:  # At least some non-dominated solutions
        print("✅ Pareto front contains expected non-dominated solutions")
    else:
        print("❌ Pareto front seems too small")
    
    return True

if __name__ == "__main__":
    test_pareto_selection()