"""Test advantage fitness (beat VWAP)."""
import pytest
from es_lobs5.training.fitness import compute_advantage_fitness


class TestAdvantageFitness:
    """Tests for compute_advantage_fitness function."""

    def test_advantage_beat_vwap(self):
        """revenue > vwap * qty 时 advantage > 0"""
        # Setup: execution revenue exceeds VWAP baseline
        total_revenue = 10500.0  # Actual execution revenue
        vwap_rm = 100.0          # Market VWAP price
        quant_executed = 100     # Quantity executed

        # VWAP baseline = 100 * 100 = 10000
        # Advantage = 10500 - 10000 = 500 > 0

        advantage = compute_advantage_fitness(
            total_revenue=total_revenue,
            vwap_rm=vwap_rm,
            quant_executed=quant_executed,
        )

        assert advantage > 0, f"Expected positive advantage when beating VWAP, got {advantage}"
        assert advantage == 500.0, f"Expected advantage=500.0, got {advantage}"

    def test_advantage_lose_vwap(self):
        """revenue < vwap * qty 时 advantage < 0"""
        # Setup: execution revenue below VWAP baseline
        total_revenue = 9500.0   # Actual execution revenue
        vwap_rm = 100.0          # Market VWAP price
        quant_executed = 100     # Quantity executed

        # VWAP baseline = 100 * 100 = 10000
        # Advantage = 9500 - 10000 = -500 < 0

        advantage = compute_advantage_fitness(
            total_revenue=total_revenue,
            vwap_rm=vwap_rm,
            quant_executed=quant_executed,
        )

        assert advantage < 0, f"Expected negative advantage when losing to VWAP, got {advantage}"
        assert advantage == -500.0, f"Expected advantage=-500.0, got {advantage}"

    def test_advantage_zero_quantity(self):
        """Zero quantity should return 0 advantage."""
        advantage = compute_advantage_fitness(
            total_revenue=1000.0,
            vwap_rm=100.0,
            quant_executed=0,
        )

        assert advantage == 0.0, f"Expected 0 for zero quantity, got {advantage}"

    def test_advantage_exact_vwap(self):
        """Executing at exactly VWAP should give zero advantage."""
        total_revenue = 10000.0
        vwap_rm = 100.0
        quant_executed = 100

        advantage = compute_advantage_fitness(
            total_revenue=total_revenue,
            vwap_rm=vwap_rm,
            quant_executed=quant_executed,
        )

        assert advantage == 0.0, f"Expected 0 for exact VWAP, got {advantage}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
