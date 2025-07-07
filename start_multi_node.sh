#!/bin/bash

# Multi-Node DDP Training Script
# This script demonstrates the CORRECT way to start multi-node training
#
# For proper DDP multi-node setup, you need EXACTLY:
# - 1 torchrun process per node
# - Each torchrun manages all GPUs on that node
#
# Setup: gpu-node1 (master, rank 0) + gpu-node (worker, rank 1)

echo "🔥 Starting Multi-Node DDP Training"
echo "=========================================="
echo "📍 Current node: $(hostname)"
echo "🌐 Master IP: 192.168.1.144 (gpu-node1)"
echo "🔢 Total nodes: 2"
echo "💻 GPUs per node: 1"
echo ""

# Configuration
MASTER_ADDR="192.168.1.144"  # gpu-node1 IP
MASTER_PORT="29503"
NNODES="2"
NPROC_PER_NODE="1"
EPOCHS="5"
BATCH_SIZE="8"
DATA_SIZE="100"

echo "⚠️  IMPORTANT: For multi-node DDP, you need to start this script manually on BOTH nodes:"
echo ""
echo "👉 On gpu-node1 (master):"
echo "   cd /home/sanzad/git/pytorch_app"
echo "   source venv/bin/activate"
echo "   ./start_multi_node.sh master"
echo ""
echo "👉 On gpu-node (worker):"
echo "   cd /home/sanzad/git/pytorch_app"
echo "   source venv/bin/activate"
echo "   ./start_multi_node.sh worker"
echo ""

if [ "$1" = "master" ]; then
    echo "🎯 Starting MASTER node (rank 0)"
    NODE_RANK=0
elif [ "$1" = "worker" ]; then
    echo "🎯 Starting WORKER node (rank 1)"
    NODE_RANK=1
else
    echo "❌ Usage: $0 [master|worker]"
    echo ""
    echo "Example commands:"
    echo "  ./start_multi_node.sh master   # Run on gpu-node1"
    echo "  ./start_multi_node.sh worker   # Run on gpu-node"
    exit 1
fi

echo "📝 Command: torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT hello_pytorch_ddp.py --epochs=$EPOCHS --batch-size=$BATCH_SIZE --data-size=$DATA_SIZE"
echo ""

# Start training
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    hello_pytorch_ddp.py \
    --epochs=$EPOCHS \
    --batch-size=$BATCH_SIZE \
    --data-size=$DATA_SIZE

echo ""
echo "🎉 Training completed on $(hostname)!" 