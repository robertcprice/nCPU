#!/bin/bash
cd /root/kvrm-spnc
pkill -f train_mul 2>/dev/null || true
sleep 2

echo "Starting 3 PARALLEL MUL approaches (no sequential loops!)..."

nohup python3 train_mul_parallel.py --model v2 --batch-size 2048 > parallel_v2.log 2>&1 &
echo "V2 (outer product) started: PID $!"

nohup python3 train_mul_parallel.py --model transformer --batch-size 1024 > parallel_transformer.log 2>&1 &
echo "Transformer started: PID $!"

nohup python3 train_mul_parallel.py --model conv --batch-size 2048 > parallel_conv.log 2>&1 &
echo "Convolution started: PID $!"

sleep 5
echo ""
echo "Running processes:"
ps aux | grep train_mul | grep -v grep
