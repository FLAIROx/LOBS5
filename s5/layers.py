from flax import linen as nn
import jax
from typing import Any


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
    dtype: Any = jax.numpy.float32

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm(step_rescale=self.step_rescale)

        # GPT-style initialization for Dense layers
        # gpt_init = nn.initializers.normal(stddev=0.02)

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model, dtype=self.dtype)
            self.out2 = nn.Dense(self.d_model, dtype=self.dtype)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model, dtype=self.dtype)

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name='batch')
        else:
            self.norm = nn.LayerNorm(dtype=self.dtype)

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x, hidden_in=None, return_hidden=False):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
             hidden_in (complex64): optional initial hidden state (1, P)
             return_hidden (bool): whether to return final hidden state
        Returns:
            If return_hidden=False:
                output sequence (float32): (L, d_model)
            If return_hidden=True:
                (hidden_out, output): hidden_out is (1, P) complex64
        """
        #jax.debug.print("call x before prenorm : {}",x)

        skip = x
        if self.prenorm:
            x = self.norm(x)

        #jax.debug.print("call x before ssm : {}",x)
        if return_hidden:
            hidden_out, x = self.seq(x, hidden_in=hidden_in, return_hidden=True)
        else:
            x = self.seq(x, hidden_in=hidden_in, return_hidden=False)
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

        if return_hidden:
            return hidden_out, x
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
            #jax.debug.print("call_rnn x before prenorm : {}",x)

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

            return hidden, x
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return jax.numpy.zeros((batch_size,1, hidden_size), dtype=jax.numpy.complex64)