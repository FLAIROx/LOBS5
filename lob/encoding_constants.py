"""
Constants for LOB encoding that don't require JAX.
This module can be safely imported in worker processes without triggering JAX CUDA initialization.
"""
import numpy as np

# Special value constants
NA_VAL = -9999
HIDDEN_VAL = -20000
MASK_VAL = -10000
START_VAL = -30000


class VocabConstants:
    """Vocab constants that don't require JAX initialization."""
    MASK_TOK = 0
    HIDDEN_TOK = 1
    NA_TOK = 2
    START_TOK = 3


class Message_Tokenizer:
    """Message tokenizer constants - no JAX dependency."""

    FIELDS = (
        'event_type',
        'direction',
        'price',
        'size',
        'delta_t_s',
        'delta_t_ns',
        'time_s',
        'time_ns',
        # reference fields:
        'price_ref',
        'size_ref',
        'time_s_ref',
        'time_ns_ref',
    )
    N_NEW_FIELDS = 8
    N_REF_FIELDS = 4
    # note: list comps only work inside function for class variables
    FIELD_I = (lambda fields=FIELDS:{
        f: i for i, f in enumerate(fields)
    })()
    # Updated: size and size_ref now use 2 tokens each (base-100 encoding)
    #          event_type, direction, price(2), size(2), delta_t_s, delta_t_ns(3), time_s(2), time_ns(3), price_ref(2), size_ref(2), time_s_ref(2), time_ns_ref(3)
    TOK_LENS = np.array((1, 1, 2, 2, 1, 3, 2, 3, 2, 2, 2, 3))
    TOK_DELIM = np.cumsum(TOK_LENS[:-1])
    MSG_LEN = np.sum(TOK_LENS)
    # encoded message length: total length - length of reference fields
    NEW_MSG_LEN = MSG_LEN - \
        (lambda tl=TOK_LENS, fields=FIELDS: np.sum(tl[i] for i, f in enumerate(fields) if f.endswith('_ref')))()
    # fields in correct message order:
    FIELD_ENC_TYPES = {
        'event_type': 'event_type',
        'direction': 'direction',
        'price': 'price',
        'size': 'size',
        'delta_t_s': 'time',
        'delta_t_ns': 'time',
        'time_s': 'time',
        'time_ns': 'time',
        'price_ref': 'price',
        'size_ref': 'size',
        'time_s_ref': 'time',
        'time_ns_ref': 'time',
    }

    @staticmethod
    def get_field_from_idx(idx):
        """ Get the field of a given index (or indices) in a message
        """
        if isinstance(idx, int) or idx.ndim == 0:
            idx = np.array([idx])
        if np.any(idx > Message_Tokenizer.MSG_LEN - 1):
            raise ValueError("Index ({}) must be less than {}".format(idx, Message_Tokenizer.MSG_LEN))
        field_i = np.searchsorted(Message_Tokenizer.TOK_DELIM, idx, side='right')
        return [Message_Tokenizer.FIELDS[i] for i in field_i]
