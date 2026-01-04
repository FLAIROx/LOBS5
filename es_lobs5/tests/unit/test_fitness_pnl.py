"""Test PnL fitness function."""
import pytest
import jax.numpy as jnp

from es_lobs5.training.fitness import (
    compute_pnl_fitness,
    compute_execution_fitness,
    compute_advantage_fitness,
    compute_normalized_pnl_fitness,
)


class TestPnLFitness:
    """Tests for PnL fitness functions."""

    def test_pnl_fitness_basic(self):
        """Test that fitness equals total_revenue."""
        total_revenue = 1000.0
        fitness = compute_pnl_fitness(total_revenue)
        assert fitness == total_revenue

        # Test with zero revenue
        assert compute_pnl_fitness(0.0) == 0.0

        # Test with float precision
        assert compute_pnl_fitness(123.456) == 123.456

    def test_pnl_fitness_positive(self):
        """Test that profitable trades give positive fitness."""
        # Profitable trade: positive revenue
        profitable_revenue = 5000.0
        fitness = compute_pnl_fitness(profitable_revenue)
        assert fitness > 0, "Profitable trade should have positive fitness"

        # More profit = higher fitness
        higher_revenue = 10000.0
        higher_fitness = compute_pnl_fitness(higher_revenue)
        assert higher_fitness > fitness, "Higher revenue should give higher fitness"

    def test_pnl_fitness_negative(self):
        """Test that loss trades give negative fitness."""
        # Loss trade: negative revenue
        loss_revenue = -2000.0
        fitness = compute_pnl_fitness(loss_revenue)
        assert fitness < 0, "Loss trade should have negative fitness"

        # Bigger loss = lower (more negative) fitness
        bigger_loss = -5000.0
        bigger_loss_fitness = compute_pnl_fitness(bigger_loss)
        assert bigger_loss_fitness < fitness, "Bigger loss should give lower fitness"


class TestExecutionFitness:
    """Tests for execution quality fitness."""

    def test_execution_fitness_no_penalty(self):
        """Test execution fitness without slippage/vwap penalties."""
        revenue = 1000.0
        fitness = compute_execution_fitness(revenue)
        assert fitness == revenue

    def test_execution_fitness_with_slippage(self):
        """Test execution fitness with slippage penalty."""
        revenue = 1000.0
        slippage = 10.0
        slippage_weight = 0.5
        fitness = compute_execution_fitness(
            revenue,
            slippage_rm=slippage,
            slippage_weight=slippage_weight
        )
        expected = revenue - slippage_weight * abs(slippage)
        assert float(fitness) == pytest.approx(expected)


class TestAdvantageFitness:
    """Tests for advantage over VWAP fitness."""

    def test_advantage_beat_vwap(self):
        """Test advantage when beating VWAP."""
        total_revenue = 1100.0
        vwap = 100.0
        qty = 10
        # Expected: 1100 - 100*10 = 100 (beat VWAP by 100)
        fitness = compute_advantage_fitness(total_revenue, vwap, qty)
        assert fitness == 100.0

    def test_advantage_underperform_vwap(self):
        """Test advantage when underperforming VWAP."""
        total_revenue = 900.0
        vwap = 100.0
        qty = 10
        # Expected: 900 - 100*10 = -100 (underperformed by 100)
        fitness = compute_advantage_fitness(total_revenue, vwap, qty)
        assert fitness == -100.0

    def test_advantage_zero_quantity(self):
        """Test advantage with zero quantity executed."""
        fitness = compute_advantage_fitness(1000.0, 100.0, 0)
        assert fitness == 0.0


class TestNormalizedPnLFitness:
    """Tests for normalized PnL fitness."""

    def test_normalized_positive(self):
        """Test normalized fitness with profit."""
        # Sold 10 shares at avg 105 (expected at 100)
        # Revenue = 1050, expected = 1000, advantage = 50
        fitness = compute_normalized_pnl_fitness(
            total_revenue=1050.0,
            task_size=10,
            init_price=100.0,
            scale=100.0
        )
        assert fitness == pytest.approx(0.5)

    def test_normalized_negative(self):
        """Test normalized fitness with loss."""
        fitness = compute_normalized_pnl_fitness(
            total_revenue=950.0,
            task_size=10,
            init_price=100.0,
            scale=100.0
        )
        assert fitness == pytest.approx(-0.5)

    def test_normalized_zero_expected(self):
        """Test normalized fitness when expected revenue is zero."""
        fitness = compute_normalized_pnl_fitness(
            total_revenue=100.0,
            task_size=0,
            init_price=100.0
        )
        assert fitness == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
