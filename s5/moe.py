"""Mixture of Experts (MoE) module for S5 SSM.

Implements dispatch-combine routing with SwiGLU expert FFNs,
following GLaM/Switch Transformer routing with GLM-4.7 style shared experts.

Architecture:
  x -> MoERouter -> dispatch -> Expert FFN (SwiGLU) -> combine -> output
  x -> SharedExpert (SwiGLU) -> added to routed output

All MoE params use bfloat16 for memory efficiency (SSM params stay fp32).
"""
import math
import jax
import jax.numpy as jnp
from flax import linen as nn


class MoERouter(nn.Module):
    """Top-k router with load balance and z-loss auxiliaries.

    Computes gate logits, selects top-k experts per token, returns
    normalized routing weights. Auxiliary losses are sow'd into
    'intermediates' for extraction in train_step.

    Args:
        num_experts: total number of routed experts (E)
        top_k: number of experts selected per token (K)
        lb_weight: load balance loss coefficient (alpha)
        z_loss_weight: router z-loss coefficient
    """
    num_experts: int
    top_k: int
    lb_weight: float = 0.01
    z_loss_weight: float = 0.001

    @nn.compact
    def __call__(self, x):
        """Route tokens to experts.

        Args:
            x: (L, D) input tokens

        Returns:
            top_k_indices: (L, K) expert indices per token
            top_k_weights: (L, K) normalized routing weights
        """
        L, D = x.shape
        E = self.num_experts
        K = self.top_k

        # Gate logits: (L, E)
        gate_logits = nn.Dense(
            E, use_bias=False, dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16, name='gate',
        )(x.astype(jnp.bfloat16)).astype(jnp.float32)

        # Router z-loss: mean(logsumexp(logits)^2) — stabilizes training
        log_z = jax.nn.logsumexp(gate_logits, axis=-1)  # (L,)
        z_loss = self.z_loss_weight * jnp.mean(log_z ** 2)
        self.sow('intermediates', 'moe_aux_loss', z_loss)

        # Top-k selection
        top_k_logits, top_k_indices = jax.lax.top_k(gate_logits, K)  # (L, K)
        top_k_weights = jax.nn.softmax(top_k_logits, axis=-1)  # (L, K)

        # Load balance loss: alpha * E^2 * mean(f_i * P_i)
        # f_i = fraction of tokens routed to expert i
        # P_i = mean routing probability for expert i
        expert_mask = jax.nn.one_hot(top_k_indices, E)  # (L, K, E)
        f_i = jnp.mean(jnp.sum(expert_mask, axis=1), axis=0)  # (E,)
        router_probs = jax.nn.softmax(gate_logits, axis=-1)  # (L, E)
        p_i = jnp.mean(router_probs, axis=0)  # (E,)
        lb_loss = self.lb_weight * (E ** 2) * jnp.mean(f_i * p_i)
        self.sow('intermediates', 'moe_aux_loss', lb_loss)

        return top_k_indices, top_k_weights


class MoEFFN(nn.Module):
    """Mixture of Experts FFN with dispatch-combine routing and SwiGLU experts.

    Uses dispatch-combine (NOT dense matmul) for efficiency with many experts.
    Expert weights are 3D tensors computed via batched einsum.

    Args:
        d_model: model dimension (D)
        d_ff: FFN hidden dimension (H)
        num_experts: number of routed experts (E)
        top_k: experts per token (K)
        num_shared_experts: number of shared experts (applied to all tokens)
        capacity_factor: capacity factor for expert buffer sizing
        lb_weight: load balance loss weight
        z_loss_weight: router z-loss weight
    """
    d_model: int
    d_ff: int
    num_experts: int = 128
    top_k: int = 8
    num_shared_experts: int = 1
    capacity_factor: float = 1.25
    lb_weight: float = 0.01
    z_loss_weight: float = 0.001

    def setup(self):
        self.router = MoERouter(
            num_experts=self.num_experts,
            top_k=self.top_k,
            lb_weight=self.lb_weight,
            z_loss_weight=self.z_loss_weight,
        )

        # Expert weights as 3D tensors (E, D, H) / (E, H, D) in bfloat16
        E, D, H = self.num_experts, self.d_model, self.d_ff
        self.wi_gate = self.param(
            'wi_gate',
            nn.initializers.lecun_normal(dtype=jnp.bfloat16),
            (E, D, H), jnp.bfloat16,
        )
        self.wi_value = self.param(
            'wi_value',
            nn.initializers.lecun_normal(dtype=jnp.bfloat16),
            (E, D, H), jnp.bfloat16,
        )
        self.wo = self.param(
            'wo',
            nn.initializers.lecun_normal(dtype=jnp.bfloat16),
            (E, H, D), jnp.bfloat16,
        )

        if self.num_shared_experts > 0:
            self.shared_experts = [
                SharedExpert(
                    d_model=self.d_model, d_ff=self.d_ff,
                    name=f'SharedExpert_{i}',
                )
                for i in range(self.num_shared_experts)
            ]

    def __call__(self, x):
        """Dispatch-combine MoE forward pass.

        Args:
            x: (L, D) input tokens

        Returns:
            output: (L, D) MoE output
        """
        L, D = x.shape
        E = self.num_experts
        K = self.top_k

        # Capacity per expert
        C = math.ceil(L * K / E * self.capacity_factor)
        C = max(C, 1)

        # Route
        top_k_indices, top_k_weights = self.router(x)  # (L, K), (L, K)

        # Build dispatch table
        # Flatten (L, K) -> (L*K,) assignments
        flat_expert_ids = top_k_indices.reshape(-1)        # (L*K,)
        flat_token_ids = jnp.repeat(jnp.arange(L), K)     # (L*K,)
        flat_weights = top_k_weights.reshape(-1)           # (L*K,)

        # Sort by expert ID to group assignments per expert
        sort_order = jnp.argsort(flat_expert_ids, stable=True)
        sorted_expert_ids = flat_expert_ids[sort_order]
        sorted_token_ids = flat_token_ids[sort_order]
        sorted_weights = flat_weights[sort_order]

        # Compute per-expert slot positions via segment cumcount
        # Detect expert boundaries: where expert ID changes
        changed = jnp.concatenate([
            jnp.array([True]),
            sorted_expert_ids[1:] != sorted_expert_ids[:-1],
        ])
        # Cumulative index within each expert segment
        # boundary_positions[i] = index of the last boundary at or before i
        boundary_idx = jnp.where(changed, jnp.arange(L * K), 0)
        boundary_idx = jax.lax.cummax(boundary_idx, axis=0)
        cumcount = jnp.arange(L * K) - boundary_idx  # slot position within expert

        # Only keep assignments within capacity
        valid = cumcount < C

        # Build dispatch indices: (E, C) -> which token fills each slot
        dispatch_indices = jnp.zeros((E * C,), dtype=jnp.int32)
        dispatch_weights = jnp.zeros((E * C,), dtype=jnp.float32)

        # Flat index into (E, C) array
        flat_ec_idx = sorted_expert_ids * C + cumcount
        # Clamp invalid indices to 0 (will be masked by weight=0)
        flat_ec_idx = jnp.where(valid, flat_ec_idx, 0)

        dispatch_indices = dispatch_indices.at[flat_ec_idx].set(
            jnp.where(valid, sorted_token_ids, 0),
            mode='drop',
        )
        dispatch_weights = dispatch_weights.at[flat_ec_idx].set(
            jnp.where(valid, sorted_weights, 0.0),
            mode='drop',
        )

        dispatch_indices = dispatch_indices.reshape(E, C)
        dispatch_weights = dispatch_weights.reshape(E, C)

        # Gather expert inputs: (E, C, D)
        x_bf16 = x.astype(jnp.bfloat16)
        expert_inputs = x_bf16[dispatch_indices]  # (E, C, D)

        # Expert computation: batched SwiGLU via einsum
        gate = jnp.einsum('ecd,edh->ech', expert_inputs, self.wi_gate)
        value = jnp.einsum('ecd,edh->ech', expert_inputs, self.wi_value)
        hidden = jax.nn.silu(gate) * value  # SwiGLU activation
        expert_out = jnp.einsum('ech,ehd->ecd', hidden, self.wo)  # (E, C, D)

        # Combine: scatter-add weighted outputs back to token positions
        weighted_out = expert_out * dispatch_weights[:, :, None]
        output = jnp.zeros((L, D), dtype=jnp.bfloat16)
        flat_indices = dispatch_indices.reshape(-1)    # (E*C,)
        flat_out = weighted_out.reshape(-1, D)         # (E*C, D)
        output = output.at[flat_indices].add(flat_out)

        # Add shared expert outputs (all tokens pass through all shared experts)
        if self.num_shared_experts > 0:
            shared_out = sum(se(x) for se in self.shared_experts)
            output = output + shared_out.astype(jnp.bfloat16)

        return output.astype(x.dtype)


class SharedExpert(nn.Module):
    """Single SwiGLU FFN applied to all tokens (no routing).

    Args:
        d_model: model dimension
        d_ff: FFN hidden dimension
    """
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        """SwiGLU FFN forward.

        Args:
            x: (L, D) input
        Returns:
            output: (L, D)
        """
        x_bf16 = x.astype(jnp.bfloat16)
        gate = nn.Dense(
            self.d_ff, use_bias=False, dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16, name='wi_gate',
        )(x_bf16)
        value = nn.Dense(
            self.d_ff, use_bias=False, dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16, name='wi_value',
        )(x_bf16)
        hidden = jax.nn.silu(gate) * value
        output = nn.Dense(
            self.d_model, use_bias=False, dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16, name='wo',
        )(hidden)
        return output.astype(x.dtype)
