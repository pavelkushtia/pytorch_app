#!/bin/bash

# PyTorch DDP Setup Script
echo "🔥 Setting up PyTorch DDP (Distributed Data Parallel) environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if we're setting up for multi-node
MULTI_NODE=false
if [ "$1" = "--multi-node" ]; then
    MULTI_NODE=true
    echo "🌐 Setting up for multi-node training"
fi

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment 'venv' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf venv
    else
        echo "📦 Using existing virtual environment..."
        source venv/bin/activate
        echo "✅ Virtual environment activated!"
        echo "🎯 You can now run DDP training!"
        exit 0
    fi
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created successfully!"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated!"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully!"

# Test PyTorch installation
echo "🧪 Testing PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "❌ Error: PyTorch test failed"
    exit 1
fi

# Test distributed functionality
echo "🧪 Testing distributed functionality..."
python -c "import torch.distributed as dist; print('✅ Distributed module imported successfully')"

if [ $? -ne 0 ]; then
    echo "❌ Error: Distributed functionality test failed"
    exit 1
fi

# Make scripts executable
chmod +x hello_pytorch_ddp.py 2>/dev/null || true
chmod +x launch_ddp_training.py 2>/dev/null || true

echo ""
echo "🎉 DDP Setup completed successfully!"
echo ""

if [ "$MULTI_NODE" = true ]; then
    echo "🌐 Multi-node setup instructions:"
    echo "   1. Run this setup script on all nodes"
    echo "   2. Ensure SSH key-based authentication is set up between nodes"
    echo "   3. Test SSH connection: ssh <hostname> 'echo SSH works'"
    echo "   4. Check firewall allows communication on port 29500"
    echo "   5. Verify all nodes have the same PyTorch environment"
    echo ""
    echo "🚀 To start multi-node training:"
    echo "   python launch_ddp_training.py --mode multi-node --master-node gpu-node --worker-nodes gpu-node1"
    echo ""
else
    echo "🚀 To start single-node training:"
    echo "   python launch_ddp_training.py --mode single-node --gpus 2"
    echo ""
    echo "🌐 For multi-node training, run this script with --multi-node flag"
fi

echo "📋 Other useful commands:"
echo "   1. Activate environment: source venv/bin/activate"
echo "   2. Direct torchrun: torchrun --nproc_per_node=2 hello_pytorch_ddp.py"
echo "   3. Check GPU status: nvidia-smi"
echo "   4. Deactivate environment: deactivate"
echo "" 