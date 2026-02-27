#!/bin/bash
cd /root/kvrm-spnc
pkill -f train_mul 2>/dev/null || true
sleep 2

echo "Starting 3 MUL approaches..."

nohup python3 train_mul_approach1_hybrid.py --batch-size 1024 > approach1.log 2>&1 &
echo "Approach 1 (Hybrid) started: PID $!"

nohup python3 train_mul_approach2_decomposed.py --model simpler --batch-size 2048 > approach2.log 2>&1 &
echo "Approach 2 (Decomposed) started: PID $!"

nohup python3 train_mul_approach3_lookup.py --model chunk --batch-size 2048 > approach3.log 2>&1 &
echo "Approach 3 (Lookup) started: PID $!"

sleep 3
echo ""
echo "Running processes:"
ps aux | grep train_mul | grep -v grep
