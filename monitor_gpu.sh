#!/bin/bash
# GPU内存监控脚本

echo "GPU Memory Monitor - Press Ctrl+C to stop"
echo "=========================================="

while true; do
    clear
    echo "GPU Memory Usage ($(date +%H:%M:%S))"
    echo "----------------------------------------"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s | Memory: %s/%s MB (%.1f%%) | Util: %s%%\n",
                 $1, $2, $3, $4, ($3/$4)*100, $5}'
    echo ""

    # 检查是否有checkpoint相关的输出
    if [ -f "test_bsz24.log" ]; then
        echo "Latest checkpoint messages:"
        grep -i "checkpoint\|memory" test_bsz24.log | tail -3
    fi

    sleep 2
done