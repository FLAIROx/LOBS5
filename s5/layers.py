from flax import linen as nn
import jax


class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
            mlp_ratio   (float32):  expansion ratio for MLP block (d_ff = mlp_ratio * d_model)
                                    Only used for swiglu/mlp_gelu activations
    """
    ssm: nn.Module
    dropout: float
    d_model: int
    activation: str = "gelu"
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    mlp_ratio: float = 4.0  # Transformer-style MLP expansion ratio

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm(step_rescale=self.step_rescale)

        # Legacy GLU activations (D -> D, no expansion)
        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)

        # NEW: Transformer-style MLP (D -> d_ff -> D)
        elif self.activation in ["swiglu"]:
            # SwiGLU: used in LLaMA, Qwen, etc.
            # d_ff = mlp_ratio * d_model, but for SwiGLU we use 2/3 of that
            # to keep param count similar (since we have 3 projections)
            d_ff = int(self.d_model * self.mlp_ratio * 2 / 3)
            self.up_gate = nn.Dense(d_ff)   # gate projection
            self.up_proj = nn.Dense(d_ff)   # value projection
            self.down_proj = nn.Dense(self.d_model)  # down projection

        elif self.activation in ["mlp_gelu"]:
            # Standard Transformer MLP: D -> 4D -> D
            d_ff = int(self.d_model * self.mlp_ratio)
            self.up_proj = nn.Dense(d_ff)
            self.down_proj = nn.Dense(self.d_model)

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name='batch')
        else:
            self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def _apply_activation(self, x):
        """Apply activation/MLP block after SSM."""
        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x1))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(nn.gelu(x))

        # NEW: Transformer-style MLP activations
        elif self.activation in ["swiglu"]:
            # SwiGLU: silu(gate) * value, then down-project
            # Like LLaMA/Qwen: x = down(silu(gate(x)) * up(x))
            gate = jax.nn.silu(self.up_gate(x))  # D -> d_ff, with SiLU
            value = self.up_proj(x)              # D -> d_ff
            x = self.drop(gate * value)          # element-wise gating
            x = self.down_proj(x)                # d_ff -> D
            x = self.drop(x)

        elif self.activation in ["mlp_gelu"]:
            # Standard Transformer MLP: up -> gelu -> down
            x = self.up_proj(x)      # D -> d_ff
            x = nn.gelu(x)
            x = self.drop(x)
            x = self.down_proj(x)    # d_ff -> D
            x = self.drop(x)

        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))
        return x

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        if self.prenorm:
            x = self.norm(x)

        x = self.seq(x)
        x = self._apply_activation(x)

        x = skip + x
        if not self.prenorm:
            x = self.norm(x)

        return x

    def __call_rnn__(self,hidden, x,d):
            """
            Compute the LxH output of S5 layer given an LxH input.
            Args:
                hidden : hidden state (P,)
                x (float32): input sequence (L, d_model)
                d (bool): reset signal (L,)
            Returns:
                output sequence (float32): (L, d_model)
            """
            skip = x
            if self.prenorm:
                x = self.norm(x)

            hidden,x = self.seq.__call_rnn__(hidden,x,d)
            x = self._apply_activation(x)

            x = skip + x
            if not self.prenorm:
                x = self.norm(x)

            return hidden, x
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return jax.numpy.zeros((batch_size,1, hidden_size), dtype=jax.numpy.complex64)