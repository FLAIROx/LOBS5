import numpy as np

# File path from the previous list_dir command
data_path = "/lus/lfs1aip2/home/s5e/kangli.s5e/JAN2023/GOOG_24tok_preproc/GOOG_2023-01-03_34200000_57600000_orderbook_10_proc.npy"

def verify_column(name, values, expected_type=None, min_val=None, max_val=None):
    print(f"\n--- Verifying {name} ---")
    print(f"Sample (first 5): {values[:5]}")
    mean_val = np.mean(values)
    min_v = np.min(values)
    max_v = np.max(values)
    print(f"Stats: Min={min_v}, Max={max_v}, Mean={mean_val:.2f}")
    
    if expected_type == "price":
        # Prices for GOOG should be large (e.g., ~900000 for $90.00 or $9000.00 depending on normalization)
        # Based on index 3 being 898500, it looks like raw price * 100 or something.
        if mean_val < 1000:
            print(f"[WARNING] {name} mean value {mean_val} seems too low for a price.")
    
    if expected_type == "size":
        # Sizes should be integers, usually > 0
        if np.any(values < 0):
             print(f"[WARNING] {name} contains negative values.")

try:
    print(f"Loading {data_path}...")
    data = np.load(data_path) # mmap_mode='r' not needed for small check, but helpful if huge
    
    N = data.shape[0]
    print(f"Total Rows: {N}")
    
    # Hypothesis: 
    # 0: mid_diff
    # 1: time_s
    # 2: time_ns
    # 3: Ask Price 1
    # 4: Ask Size 1
    # 5: Bid Price 1
    # 6: Bid Size 1
    
    # Check 1: Time Monotonicity
    time_s = data[:, 1]
    time_ns = data[:, 2]
    # Construct full time roughly
    full_time = time_s + time_ns / 1e9
    is_monotonic = np.all(np.diff(full_time) >= -1e-9) # Allow tiny tolerance
    print(f"\nTime Monotonicity Check: {is_monotonic}")
    if not is_monotonic:
        diffs = np.diff(full_time)
        bad_indices = np.where(diffs < 0)[0]
        print(f"  Time decreases at indices: {bad_indices[:5]}... (Total {len(bad_indices)})")

    # Check 2: Ask > Bid
    ask_p1 = data[:, 3]
    bid_p1 = data[:, 5]
    
    # Check if Ask > Bid usually
    spread = ask_p1 - bid_p1
    valid_spread = spread > 0
    percent_valid = np.mean(valid_spread) * 100
    print(f"\nAsk P1 > Bid P1 Check: {percent_valid:.2f}% of rows have Ask > Bid")
    print(f"  Average Spread: {np.mean(spread):.2f}")
    print(f"  Min Spread: {np.min(spread)}")
    print(f"  Max Spread: {np.max(spread)}")
    
    # Check 3: Volumes are positive
    ask_v1 = data[:, 4]
    bid_v1 = data[:, 6]
    print(f"\nAsk Size 1 Positive Check: {np.all(ask_v1 >= 0)} (Min: {np.min(ask_v1)})")
    print(f"Bid Size 1 Positive Check: {np.all(bid_v1 >= 0)} (Min: {np.min(bid_v1)})")

    # Validation Conclusion
    if percent_valid > 99.0 and np.mean(ask_p1) > 10000:
        print("\n>>> CONCLUSION: Hypothesis CONFIRMED.")
        print("    Index 3 is definitely ASK PRICE.")
        print("    Index 5 is definitely BID PRICE.")
    else:
        print("\n>>> CONCLUSION: Hypothesis FAILED or Uncertain.")

except Exception as e:
    print(f"Error: {e}")
