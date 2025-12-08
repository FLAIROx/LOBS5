"""
JaxLOB Environment Wrapper for ES Training.

Wraps JaxLOB ExecutionEnv for Evolution Strategies parallel evaluation.

Architecture:
    - Frozen World Model: Predicts background market order flow
    - Policy Agent (ES trained): Generates trading actions as message tokens
    - JaxLOB: Simulates order book execution
    - Fitness: total_revenue (PnL)
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
from functools import partial

# JaxLOB imports - these require gymnax_exchange to be installed
# Import lazily to avoid import errors when not using RL mode


class RolloutState(NamedTuple):
    """State for a single rollout episode."""
    env_state: any  # JaxLOB EnvState
    hidden_world: jnp.ndarray  # World model hidden state (for RNN mode)
    hidden_policy: jnp.ndarray  # Policy hidden state (for RNN mode)
    step: int  # Current step in episode
    done: bool  # Episode terminated


class JaxLOBESWrapper:
    """
    Wraps JaxLOB for ES parallel evaluation.

    This wrapper handles:
    1. World model inference (frozen, no ES noise)
    2. Policy inference (with ES perturbation)
    3. Combining messages for JaxLOB execution
    4. Extracting PnL as fitness

    Usage:
        wrapper = JaxLOBESWrapper(env_config)
        fitness = wrapper.evaluate_policy(
            policy_params, world_model_params,
            noiser, iterinfo, data
        )
    """

    def __init__(
        self,
        alphatrade_path: str,
        task: str = 'sell',
        window_index: int = -1,
        action_type: str = 'delta',
        task_size: int = 500,
        reward_lambda: float = 0.0,
        gamma: float = 0.0,
    ):
        """
        Initialize JaxLOB wrapper.

        Args:
            alphatrade_path: Path to AlphaTrade data directory
            task: 'sell' or 'buy'
            window_index: Data window index (-1 for random)
            action_type: 'delta' or absolute
            task_size: Number of shares to execute
            reward_lambda: Reward shaping parameter
            gamma: Discount factor (unused in ES)
        """
        self.alphatrade_path = alphatrade_path
        self.task = task
        self.window_index = window_index
        self.action_type = action_type
        self.task_size = task_size
        self.reward_lambda = reward_lambda
        self.gamma = gamma

        # Lazy import JaxLOB
        self._env = None
        self._env_params = None

    def _get_env(self):
        """Lazy load JaxLOB environment."""
        if self._env is None:
            from gymnax_exchange.jaxen.exec_env import ExecutionEnv
            self._env = ExecutionEnv(
                self.alphatrade_path,
                self.task,
                self.window_index,
                self.action_type,
                self.task_size,
                self.reward_lambda,
                self.gamma,
            )
            self._env_params = self._env.default_params
        return self._env, self._env_params

    def reset(self, key: jnp.ndarray) -> Tuple[jnp.ndarray, any]:
        """
        Reset environment.

        Args:
            key: JAX random key

        Returns:
            (observation, env_state)
        """
        env, params = self._get_env()
        return env.reset_env(key, params)

    def step(
        self,
        key: jnp.ndarray,
        state: any,
        action: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, any, float, bool, dict]:
        """
        Step environment with action.

        Args:
            key: JAX random key
            state: Current EnvState
            action: Action array (policy output)

        Returns:
            (observation, new_state, reward, done, info)
        """
        env, params = self._get_env()
        return env.step_env(key, state, action, params)

    def get_pnl(self, state: any) -> float:
        """
        Extract PnL (total_revenue) from state.

        This is the fitness signal for ES training.

        Args:
            state: JaxLOB EnvState

        Returns:
            Total revenue (higher is better)
        """
        return state.total_revenue

    def get_execution_metrics(self, state: any) -> dict:
        """
        Extract all execution metrics from state.

        Args:
            state: JaxLOB EnvState

        Returns:
            Dict with execution metrics
        """
        return {
            'total_revenue': state.total_revenue,
            'quant_executed': state.quant_executed,
            'task_to_execute': state.task_to_execute,
            'slippage_rm': state.slippage_rm,
            'price_adv_rm': state.price_adv_rm,
            'price_drift_rm': state.price_drift_rm,
            'vwap_rm': state.vwap_rm,
        }

    @partial(jax.jit, static_argnums=(0,))
    def rollout_episode(
        self,
        key: jnp.ndarray,
        policy_forward_fn,
        policy_params: dict,
        world_forward_fn,
        world_params: dict,
        noiser,
        iterinfo: Tuple[int, int],
    ) -> float:
        """
        Run a complete episode and return fitness.

        Args:
            key: JAX random key
            policy_forward_fn: Policy model forward function
            policy_params: Policy parameters (will be perturbed)
            world_forward_fn: World model forward function
            world_params: World model parameters (frozen)
            noiser: ES noiser for policy perturbation
            iterinfo: (epoch, thread_id) for noise generation

        Returns:
            Fitness score (total_revenue)
        """
        env, params = self._get_env()

        key, reset_key = jax.random.split(key)
        obs, state = env.reset_env(reset_key, params)

        def step_fn(carry, _):
            key, state, obs = carry
            key, step_key, action_key = jax.random.split(key, 3)

            # World model: predict background order flow (no noise)
            # world_messages = world_forward_fn(world_params, obs)

            # Policy: generate action with ES perturbation
            # policy_messages = policy_forward_fn(
            #     policy_params, obs, noiser, iterinfo
            # )

            # For now, use a simple action (to be replaced with actual model)
            action = jnp.zeros(2)  # Placeholder

            obs, state, reward, done, info = env.step_env(
                step_key, state, action, params
            )

            return (key, state, obs), (reward, done)

        # Run episode
        max_steps = state.max_steps_in_episode
        (_, final_state, _), (rewards, dones) = jax.lax.scan(
            step_fn,
            (key, state, obs),
            None,
            length=max_steps,
        )

        return self.get_pnl(final_state)


def create_jaxlob_wrapper(config: dict) -> JaxLOBESWrapper:
    """
    Create JaxLOB wrapper from config dict.

    Args:
        config: Dict with keys:
            - alphatrade_path: str
            - task: str ('sell' or 'buy')
            - window_index: int
            - action_type: str
            - task_size: int

    Returns:
        JaxLOBESWrapper instance
    """
    return JaxLOBESWrapper(
        alphatrade_path=config.get('alphatrade_path', '/path/to/alphatrade'),
        task=config.get('task', 'sell'),
        window_index=config.get('window_index', -1),
        action_type=config.get('action_type', 'delta'),
        task_size=config.get('task_size', 500),
    )
