#!/bin/bash
# filepath: run_generation_tests.sh

# Array of (batch_size, n_samples) pairs
configs=(
    "16 32"
    "32 64"
    "64 128"
    "256 512"
    "512 1024"
)

# Run each configuration
for config in "${configs[@]}"; do
    read batch_size n_samples <<< "$config"
    
    echo "=========================================="
    echo "Running with batch_size=$batch_size, n_samples=$n_samples"
    echo "=========================================="
    
    python3 generate_data.py \
        --n_gen_msgs 500 \
        --stock GOOG \
        --n_samples=$n_samples \
        --batch_size=$batch_size
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: batch_size=$batch_size, n_samples=$n_samples"
    else
        echo "✗ Failed: batch_size=$batch_size, n_samples=$n_samples"
    fi
    echo ""
done

echo "=========================================="
echo "All tests completed!"
echo "=========================================="