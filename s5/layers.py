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
    # MoE parameters
    use_moe: bool = False
    num_experts: int = 128
    top_k: int = 8
    d_ff: int = 1024
    num_shared_experts: int = 1
    moe_capacity_factor: float = 1.25
    moe_lb_weight: float = 0.01
    moe_z_loss_weight: float = 0.001

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        # Detect if the SSM is a TransformerBlock (has its own Pre-LN + residual).
        # Unwrap nested functools.partial (e.g. LobBookModel wraps ssm in an extra partial).
        _cls = self.ssm
        while hasattr(_cls, 'func'):
            _cls = _cls.func
        self._is_transformer = getattr(_cls, 'is_transformer', False)

        self.seq = self.ssm(step_rescale=self.step_rescale)

        if not self._is_transformer:
            if self.activation in ["full_glu"]:
                self.out1 = nn.Dense(self.d_model)
                self.out2 = nn.Dense(self.d_model)
            elif self.activation in ["half_glu1", "half_glu2"]:
                self.out2 = nn.Dense(self.d_model)

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

            # MoE sub-layer (optional)
            if self.use_moe:
                from s5.moe import MoEFFN
                self.moe_norm = nn.LayerNorm()
                self.moe_ffn = MoEFFN(
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    num_shared_experts=self.num_shared_experts,
                    capacity_factor=self.moe_capacity_factor,
                    lb_weight=self.moe_lb_weight,
                    z_loss_weight=self.moe_z_loss_weight,
                )

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        # TransformerBlock has its own Pre-LN + residual — pass through directly
        if self._is_transformer:
            return self.seq(x)

        skip = x
        if self.prenorm:
            x = self.norm(x)
        
        #jax.debug.print("call x before ssm : {}",x)
        x = self.seq(x)
        #jax.debug.print("call x_m after ssm : {}",x)
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
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))
        
        #jax.debug.print("call x_m[0:5] after activation : {}",x[0:2][0][0:2])


        x = skip + x
        if not self.prenorm:
            x = self.norm(x)

        # MoE sub-layer: LayerNorm -> MoE FFN -> residual
        if self.use_moe:
            skip_moe = x
            x = self.moe_norm(x)
            x = self.moe_ffn(x)
            x = skip_moe + x

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
            # TransformerBlock: delegate to KV-cache inference
            if self._is_transformer:
                return self.seq.__call_rnn__(hidden, x, d)

            skip = x
            if self.prenorm:
                x = self.norm(x)

            hidden,x = self.seq.__call_rnn__(hidden,x,d)
            #hidden, x = jax.vmap(self.seq.__call_rnn__, in_axes=(None,1,None), out_axes=1)(hidden, x, d)


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
            else:
                raise NotImplementedError(
                    "Activation: {} not implemented".format(self.activation))

            x = skip + x
            if not self.prenorm:
                x = self.norm(x)

            # MoE sub-layer (token-wise, no hidden state)
            if self.use_moe:
                skip_moe = x
                x = self.moe_norm(x)
                x = self.moe_ffn(x)
                x = skip_moe + x

            return hidden, x
    @staticmethod
    def initialize_carry(batch_size, hidden_size,
                         is_transformer=False, transformer_config=None,
                         ssm_type='s5', **gdn_kwargs):
        if ssm_type in ('gdn', 'kda'):
            nh = gdn_kwargs['num_heads']
            hd = gdn_kwargs['head_dim']
            hvd = gdn_kwargs['head_v_dim']
            use_conv = gdn_kwargs.get('use_conv', True)
            conv_k = gdn_kwargs.get('conv_kernel_size', 4)
            S = jax.numpy.zeros((batch_size, 1, nh, hvd, hd), dtype=jax.numpy.float32)
            if use_conv:
                q_buf = jax.numpy.zeros((batch_size, conv_k - 1, nh * hd), dtype=jax.numpy.float32)
                k_buf = jax.numpy.zeros((batch_size, conv_k - 1, nh * hd), dtype=jax.numpy.float32)
                v_buf = jax.numpy.zeros((batch_size, conv_k - 1, nh * hvd), dtype=jax.numpy.float32)
                return (S, (q_buf, k_buf, v_buf))
            return S
        if is_transformer and transformer_config is not None:
            from s5.transformer import TransformerBlock
            cfg = transformer_config
            return TransformerBlock.initialize_cache(
                batch_size, cfg['n_heads'], cfg['head_dim'],
                cfg['max_cache_len'], cfg.get('dtype', jax.numpy.float32))
        return jax.numpy.zeros((batch_size,1, hidden_size), dtype=jax.numpy.complex64)