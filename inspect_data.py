import numpy as np
import sys

# File path from the previous list_dir command
data_path = "/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc/GOOG_2023-01-03_34200000_57600000_orderbook_10_proc.npy"

try:
    print(f"Loading {data_path}...")
    data = np.load(data_path)
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    
    # Print first 5 rows
    print("\nFirst 5 rows (raw):")
    for i in range(5):
        print(f"Row {i}: {data[i]}")
        
    # Check for zeros in first few columns
    print("\nChecking for zeros in first 4 columns (likely AskP, AskV, BidP, BidV):")
    print(f"Col 0 (Avg): {np.mean(data[:100, 0])}")
    print(f"Col 1 (Avg): {np.mean(data[:100, 1])}")
    print(f"Col 2 (Avg): {np.mean(data[:100, 2])}")
    print(f"Col 3 (Avg): {np.mean(data[:100, 3])}")

    # LOBSTER format is usually: Ask Price 1, Ask Size 1, Bid Price 1, Bid Size 1, ...
    # But if it's preprocessed, it might be different.
    
except Exception as e:
    print(f"Error: {e}")
