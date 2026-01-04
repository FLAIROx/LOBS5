"""Test execution fitness (combined metrics)."""
import pytest
import jax.numpy as jnp

from es_lobs5.training.fitness import compute_execution_fitness


class TestExecutionFitness:
    """Tests for execution quality fitness."""

    def test_execution_fitness_slippage(self):
        """Test: fitness = revenue - slippage_cost.

        When slippage_weight > 0, the fitness should be reduced by:
            fitness = revenue - slippage_weight * |slippage_rm|
        """
        revenue = 1000.0
        slippage = 50.0  # Slippage in price units
        slippage_weight = 1.0

        # Compute fitness with slippage penalty
        fitness = compute_execution_fitness(
            total_revenue=revenue,
            slippage_rm=slippage,
            slippage_weight=slippage_weight,
        )

        # Expected: 1000 - 1.0 * |50| = 950
        expected_fitness = revenue - slippage_weight * abs(slippage)
        assert float(fitness) == pytest.approx(expected_fitness)
        assert float(fitness) == pytest.approx(950.0)

        # Test with different slippage weight
        slippage_weight_half = 0.5
        fitness_half = compute_execution_fitness(
            total_revenue=revenue,
            slippage_rm=slippage,
            slippage_weight=slippage_weight_half,
        )
        # Expected: 1000 - 0.5 * |50| = 975
        assert float(fitness_half) == pytest.approx(975.0)

        # Test with negative slippage (absolute value should be taken)
        negative_slippage = -30.0
        fitness_neg = compute_execution_fitness(
            total_revenue=revenue,
            slippage_rm=negative_slippage,
            slippage_weight=slippage_weight,
        )
        # Expected: 1000 - 1.0 * |-30| = 970
        assert float(fitness_neg) == pytest.approx(970.0)

        # Test with zero slippage weight (no penalty)
        fitness_no_penalty = compute_execution_fitness(
            total_revenue=revenue,
            slippage_rm=slippage,
            slippage_weight=0.0,
        )
        # Expected: revenue unchanged
        assert float(fitness_no_penalty) == pytest.approx(revenue)

    def test_execution_fitness_incomplete(self):
        """Test: incomplete orders have penalty.

        An incomplete order is when quant_executed < task_size.
        The penalty comes from VWAP deviation on partial execution.

        For incomplete orders:
        - Lower quant_executed means less opportunity for VWAP penalty
        - But the trader may have suboptimal execution quality

        Test verifies that incomplete execution with VWAP deviation
        results in appropriate fitness reduction.
        """
        # Scenario: Task is to sell 100 shares
        task_size = 100
        init_price = 100.0

        # Case 1: Complete execution at VWAP (no penalty)
        complete_revenue = 10000.0  # 100 shares * 100 price
        complete_quant = 100
        vwap = 100.0

        fitness_complete = compute_execution_fitness(
            total_revenue=complete_revenue,
            vwap_rm=vwap,
            init_price=init_price,
            quant_executed=complete_quant,
            vwap_weight=1.0,
        )
        # No deviation from VWAP, so fitness = revenue
        assert float(fitness_complete) == pytest.approx(complete_revenue)

        # Case 2: Incomplete execution (only 50 shares executed)
        incomplete_quant = 50
        incomplete_revenue = 4800.0  # 50 shares * 96 avg price (below VWAP)
        avg_price_incomplete = incomplete_revenue / incomplete_quant  # 96.0

        fitness_incomplete = compute_execution_fitness(
            total_revenue=incomplete_revenue,
            vwap_rm=vwap,
            init_price=init_price,
            quant_executed=incomplete_quant,
            vwap_weight=1.0,
        )

        # VWAP deviation penalty: |96 - 100| * 50 = 200
        vwap_deviation = abs(avg_price_incomplete - vwap)
        expected_penalty = vwap_deviation * incomplete_quant
        expected_fitness = incomplete_revenue - expected_penalty
        # 4800 - 200 = 4600
        assert float(fitness_incomplete) == pytest.approx(expected_fitness)
        assert float(fitness_incomplete) == pytest.approx(4600.0)

        # Case 3: Zero execution (no penalty applied since quant=0)
        zero_revenue = 0.0
        zero_quant = 0

        fitness_zero = compute_execution_fitness(
            total_revenue=zero_revenue,
            vwap_rm=vwap,
            init_price=init_price,
            quant_executed=zero_quant,
            vwap_weight=1.0,
        )
        # No execution, no VWAP penalty (avoids division by zero)
        assert float(fitness_zero) == pytest.approx(0.0)

        # Case 4: Incomplete but good price (above VWAP for selling)
        good_incomplete_revenue = 5200.0  # 50 shares * 104 avg price
        avg_price_good = good_incomplete_revenue / incomplete_quant  # 104.0

        fitness_good_incomplete = compute_execution_fitness(
            total_revenue=good_incomplete_revenue,
            vwap_rm=vwap,
            init_price=init_price,
            quant_executed=incomplete_quant,
            vwap_weight=1.0,
        )

        # VWAP deviation penalty: |104 - 100| * 50 = 200
        # Note: VWAP penalty is symmetric (above or below VWAP both get penalized)
        vwap_deviation_good = abs(avg_price_good - vwap)
        expected_penalty_good = vwap_deviation_good * incomplete_quant
        expected_fitness_good = good_incomplete_revenue - expected_penalty_good
        # 5200 - 200 = 5000
        assert float(fitness_good_incomplete) == pytest.approx(expected_fitness_good)
        assert float(fitness_good_incomplete) == pytest.approx(5000.0)


class TestCombinedPenalties:
    """Tests for combined slippage and VWAP penalties."""

    def test_combined_slippage_and_vwap(self):
        """Test fitness with both slippage and VWAP penalties."""
        revenue = 10000.0
        slippage = 20.0
        slippage_weight = 0.5
        vwap = 100.0
        quant_executed = 100
        vwap_weight = 0.1

        # Average price = 10000 / 100 = 100 (matches VWAP)
        fitness = compute_execution_fitness(
            total_revenue=revenue,
            slippage_rm=slippage,
            vwap_rm=vwap,
            quant_executed=quant_executed,
            slippage_weight=slippage_weight,
            vwap_weight=vwap_weight,
        )

        # Slippage penalty: 0.5 * 20 = 10
        # VWAP penalty: |100 - 100| * 0.1 * 100 = 0
        # Fitness: 10000 - 10 - 0 = 9990
        expected = revenue - slippage_weight * slippage
        assert float(fitness) == pytest.approx(expected)
        assert float(fitness) == pytest.approx(9990.0)

    def test_combined_with_vwap_deviation(self):
        """Test combined penalties with VWAP deviation."""
        revenue = 9500.0  # 100 shares at avg 95
        slippage = 10.0
        slippage_weight = 1.0
        vwap = 100.0
        quant_executed = 100
        vwap_weight = 0.5

        fitness = compute_execution_fitness(
            total_revenue=revenue,
            slippage_rm=slippage,
            vwap_rm=vwap,
            quant_executed=quant_executed,
            slippage_weight=slippage_weight,
            vwap_weight=vwap_weight,
        )

        # Average price = 9500 / 100 = 95
        avg_price = revenue / quant_executed
        # Slippage penalty: 1.0 * 10 = 10
        slippage_penalty = slippage_weight * abs(slippage)
        # VWAP penalty: |95 - 100| * 100 * 0.5 = 5 * 100 * 0.5 = 250
        vwap_penalty = vwap_weight * abs(avg_price - vwap) * quant_executed
        # Fitness: 9500 - 10 - 250 = 9240
        expected = revenue - slippage_penalty - vwap_penalty
        assert float(fitness) == pytest.approx(expected)
        assert float(fitness) == pytest.approx(9240.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
