import numpy as np

# File path from the previous list_dir command
data_path = "/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc/GOOG_2023-01-03_34200000_57600000_message_10_proc.npy"

try:
    print(f"Loading {data_path}...")
    data = np.load(data_path)
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    
    # Print first 5 rows
    print("\nFirst 5 rows (raw):")
    for i in range(5):
        print(f"Row {i}: {data[i]}")
        
    # Check if values look like tokens (small ints) or raw values (large ints/floats)
    print("\nValue Range Analysis:")
    for col in range(data.shape[1]):
        min_v = np.min(data[:, col])
        max_v = np.max(data[:, col])
        print(f"  Col {col}: Min={min_v}, Max={max_v}")

except Exception as e:
    print(f"Error: {e}")
