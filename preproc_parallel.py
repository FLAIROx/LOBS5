#!/usr/bin/env python3
"""Parallel preprocessing script for GOOG 2022 data using multiprocessing."""

from __future__ import annotations
import argparse
from pathlib import Path
from glob import glob
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from decimal import Decimal
from functools import partial

# Import from original preproc
from lob.encoding import Vocab, Message_Tokenizer


def load_message_df(m_f: str) -> pd.DataFrame:
    cols = ['time', 'event_type', 'order_id', 'size', 'price', 'direction']
    messages = pd.read_csv(
        m_f,
        names=cols,
        usecols=cols,
        index_col=False,
        dtype={
            'time': str,
            'event_type': 'int32',
            'order_id': 'int32',
            'size': 'int32',
            'price': 'int32',
            'direction': 'int32'
        }
    )
    messages.time = messages.time.apply(lambda x: Decimal(x))
    return messages


def augment_book_state(book: pd.DataFrame, message: pd.DataFrame):
    best_ask = book.iloc[:, 0].replace({9999999999: np.nan}).ffill().astype(int)
    best_bid = book.iloc[:, 2].replace({-9999999999: np.nan}).ffill().astype(int)
    p_ref = ((best_ask + best_bid) / 2).round(-2).astype(int)
    mid_diff = p_ref.div(100).diff().fillna(0).astype(int)

    message.insert(0, 'time_s', message.time.astype(int))
    message.rename(columns={'time': 'time_ns'}, inplace=True)
    message.time_ns = ((message.time_ns % 1) * 1000000000).astype(int)

    times = message[['time_s', 'time_ns']]
    times = times.values.reshape(-1, 2)
    book = np.concatenate((mid_diff.values.reshape(-1, 1), times, book.values), axis=1)
    return book


def process_single_file(args):
    """Process a single message/book file pair."""
    m_f, b_f, save_dir, skip_existing = args

    v = Vocab()
    tok = Message_Tokenizer()

    m_path = save_dir + m_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
    b_path = save_dir + b_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'

    # Check if both already exist
    if skip_existing and Path(m_path).exists() and Path(b_path).exists():
        return f'skipped: {m_f}'

    try:
        messages = load_message_df(m_f)
        book = pd.read_csv(b_f, index_col=False, header=None)
    except pd.errors.EmptyDataError:
        return f'empty: {b_f}'
    except Exception as e:
        return f'error loading {m_f}: {e}'

    if len(messages) != len(book):
        return f'length mismatch: {m_f}'

    # Filter by time
    messages_filtered = messages.loc[(messages.time >= Decimal(34200)) & (messages.time < Decimal(57600))]
    book_filtered = book.loc[messages_filtered.index]

    # Process messages
    if not (skip_existing and Path(m_path).exists()):
        m_ = tok.preproc(messages_filtered.copy(), book_filtered.copy())
        np.save(m_path, m_)

    # Process orderbook with raw representation
    if not (skip_existing and Path(b_path).exists()):
        messages_for_book = messages.loc[messages.event_type.isin([1, 2, 3, 4])]
        messages_for_book = messages_for_book.loc[(messages_for_book.time >= Decimal(34200)) & (messages_for_book.time < Decimal(57600))]
        book_for_proc = book.loc[messages_for_book.index].copy()
        book_processed = augment_book_state(book_for_proc, messages_for_book.copy())
        np.save(b_path, book_processed, allow_pickle=True)

    return f'done: {m_f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--skip_existing", action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

    message_files = sorted(glob(args.data_dir + '*message*.csv'))
    book_files = sorted(glob(args.data_dir + '*orderbook*.csv'))

    print(f'Found {len(message_files)} message files')
    print(f'Found {len(book_files)} book files')

    if len(message_files) != len(book_files):
        print("Error: message and book file counts don't match!")
        return

    # Prepare arguments for parallel processing
    task_args = [
        (m_f, b_f, args.save_dir, args.skip_existing)
        for m_f, b_f in zip(message_files, book_files)
    ]

    num_workers = args.num_workers or min(cpu_count(), 32)
    print(f'Using {num_workers} workers')

    with Pool(num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_file, task_args)):
            print(f'[{i+1}/{len(task_args)}] {result}')

    print('DONE')


if __name__ == '__main__':
    main()
