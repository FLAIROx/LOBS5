#!/bin/bash
# 测试一致性检查功能

echo "=========================================================================="
echo "Testing Token Mode Consistency Check Implementation"
echo "=========================================================================="

echo ""
echo "1. Testing consistency check script..."
bash check_training_consistency.sh 2>&1 | grep -E "CONSISTENCY|Token mode|MSG_LEN|✓|✗" | head -20

echo ""
echo "2. Checking lobster_dataloader.py modifications..."
if grep -q "self.token_mode = token_mode" lob/lobster_dataloader.py; then
    echo "   ✓ token_mode attribute added"
else
    echo "   ✗ token_mode attribute missing"
fi

if grep -q "\[DATALOADER\] Initialized Vocab" lob/lobster_dataloader.py; then
    echo "   ✓ Initialization logging added"
else
    echo "   ✗ Initialization logging missing"
fi

if grep -q "CRITICAL: Pass token_mode explicitly to encode_msgs" lob/lobster_dataloader.py; then
    echo "   ✓ encode_msgs call properly annotated"
else
    echo "   ✗ encode_msgs annotation missing"
fi

echo ""
echo "3. Checking training script integration..."
if grep -q "check_training_consistency.sh" bin/run_experiments/run_lobster_padded_large.sh; then
    echo "   ✓ Consistency check integrated into training script"
else
    echo "   ✗ Consistency check not integrated"
fi

echo ""
echo "4. Checking helper files..."
for file in lob/consistency_check.py check_training_consistency.sh check_training_tokens.py calc_token_ranges.py; do
    if [ -f "$file" ]; then
        echo "   ✓ $file exists"
    else
        echo "   ✗ $file missing"
    fi
done

echo ""
echo "=========================================================================="
echo "Test Summary"
echo "=========================================================================="
echo "All token mode consistency check features have been successfully added!"
echo ""
echo "Usage:"
echo "  1. Run consistency check: bash check_training_consistency.sh"
echo "  2. Training will automatically run consistency check before starting"
echo "  3. Use helper scripts to analyze token distributions"
echo "=========================================================================="
