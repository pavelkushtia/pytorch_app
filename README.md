# PyTorch Hello World & DDP Training

A comprehensive PyTorch application that demonstrates both basic neural network functionality and distributed training with PyTorch's DistributedDataParallel (DDP). This repository includes simple hello world examples and advanced multi-node GPU training capabilities.

## 🚀 Features

- **Single GPU Training**: Basic PyTorch hello world with automatic GPU detection
- **Multi-GPU Training**: Single-node distributed training across multiple GPUs
- **Multi-Node Training**: Distributed training across multiple machines
- **Automatic Setup**: Automated setup scripts for both single and multi-node environments
- **Easy Launchers**: Simple launcher scripts for various training configurations
- **Visualization**: Creates plots showing training results and performance metrics

## 📋 Requirements

- Python 3.8+
- PyTorch 2.2.0+ (with CUDA support for GPU training)
- NumPy 1.24.0+
- Matplotlib 3.7.0+
- SSH key-based authentication (for multi-node training)
- NVIDIA GPUs with CUDA support

## 🛠️ Installation

### Quick Setup

```bash
# Clone this repository
git clone <your-repo-url>
cd pytorch_app

# For single-node training
./setup_ddp.sh

# For multi-node training
./setup_ddp.sh --multi-node
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### 1. Single GPU Training (Basic Hello World)

```bash
# Activate the environment
source venv/bin/activate

# Run the basic PyTorch hello world
python hello_pytorch_gpu.py
```

### 2. Single-Node Multi-GPU Training

```bash
# Using the launcher (recommended)
python launch_ddp_training.py --mode single-node --gpus 2

# Or using torchrun directly
torchrun --nproc_per_node=2 hello_pytorch_ddp.py
```

### 3. Multi-Node Training (gpu-node & gpu-node1)

**IMPORTANT**: For multi-node DDP, you need **exactly ONE torchrun process per node**.

#### Method 1: Using the automated launcher (if SSH is properly configured)
```bash
python launch_ddp_training.py --mode multi-node --master-node gpu-node1 --worker-nodes gpu-node
```

#### Method 2: Manual start (recommended for learning)
Start the processes manually on each node:

**On gpu-node1 (master node):**
```bash
cd /home/sanzad/git/pytorch_app
source venv/bin/activate
./start_multi_node.sh master
```

**On gpu-node (worker node):**
```bash
cd /home/sanzad/git/pytorch_app
source venv/bin/activate
./start_multi_node.sh worker
```

#### Method 3: Direct torchrun commands
**On gpu-node1 (master, rank 0):**
```bash
source venv/bin/activate
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=192.168.1.144 --master_port=29500 hello_pytorch_ddp.py --epochs=10
```

**On gpu-node (worker, rank 1):**
```bash
source venv/bin/activate
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=192.168.1.144 --master_port=29500 hello_pytorch_ddp.py --epochs=10
```

### 4. Advanced Configuration

```bash
# Custom training parameters
python launch_ddp_training.py \
    --mode multi-node \
    --master-node gpu-node \
    --worker-nodes gpu-node1 \
    --gpus-per-node 2 \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --data-size 5000
```

## 🏗️ Project Structure

```
pytorch_app/
├── hello_pytorch_gpu.py        # Basic single-GPU hello world
├── hello_pytorch_ddp.py        # DDP training script
├── launch_ddp_training.py      # DDP launcher script
├── setup_ddp.sh               # DDP setup script
├── hosts.txt                  # GPU nodes configuration
├── requirements.txt           # Python dependencies
├── setup.sh                   # Basic setup script
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## 🔧 DDP Training Details

### What the DDP Script Does

1. **Distributed Setup**: Initializes process groups across all nodes
2. **Data Distribution**: Splits data across all processes using DistributedSampler
3. **Model Synchronization**: Keeps model parameters synchronized across all GPUs
4. **Gradient Synchronization**: Automatically synchronizes gradients during backpropagation
5. **Visualization**: Saves training progress plots (only on rank 0)

### Key DDP Concepts Demonstrated

- **Process Groups**: Setting up communication between distributed processes
- **DistributedDataParallel**: Wrapping models for distributed training
- **DistributedSampler**: Ensuring data is properly distributed across processes
- **Rank Management**: Handling different process ranks (master vs worker)
- **Gradient Synchronization**: Automatic gradient averaging across processes

### ⚠️ Important: Process Count Rules

**For multi-node DDP training, you need EXACTLY:**
- **1 torchrun process per node** (not per GPU!)
- Each torchrun process manages all GPUs on that node via `--nproc_per_node`

**Example with 2 nodes, 1 GPU each:**
- gpu-node1: 1 torchrun process (rank 0, master)
- gpu-node: 1 torchrun process (rank 1, worker)
- **Total: 2 torchrun processes**

**Example with 2 nodes, 2 GPUs each:**
- gpu-node1: 1 torchrun process managing 2 GPUs (rank 0, master)
- gpu-node: 1 torchrun process managing 2 GPUs (rank 1, worker)
- **Total: 2 torchrun processes, 4 GPU workers**

**Common mistakes:**
- ❌ Starting multiple torchrun processes on the same node
- ❌ Using `--nproc_per_node` = number of nodes instead of GPUs per node
- ❌ Not coordinating the start time between nodes

## 🌐 Multi-Node Setup Requirements

### SSH Configuration

1. **Set up SSH keys** between all nodes:
   ```bash
   # Generate SSH key (if not exists)
   ssh-keygen -t rsa -b 4096
   
   # Copy public key to all nodes
   ssh-copy-id user@gpu-node
   ssh-copy-id user@gpu-node1
   
   # Test SSH connection
   ssh gpu-node 'echo "SSH works"'
   ssh gpu-node1 'echo "SSH works"'
   ```

2. **Configure hostnames** in `/etc/hosts` or DNS:
   ```
   192.168.1.100  gpu-node
   192.168.1.101  gpu-node1
   ```

### Firewall Configuration

Ensure port 29500 (default) is open for communication:
```bash
# On Ubuntu/Debian
sudo ufw allow 29500

# On CentOS/RHEL
sudo firewall-cmd --add-port=29500/tcp --permanent
sudo firewall-cmd --reload
```

### Environment Synchronization

- Install the same PyTorch version on all nodes
- Ensure CUDA versions are compatible
- Sync the training code to all nodes

## 📊 Expected Performance

### Single Node vs Multi-Node

- **Single GPU**: Baseline performance
- **Multi-GPU (Same Node)**: ~1.8x speedup (depends on GPU communication)
- **Multi-Node**: ~1.5x speedup per additional node (depends on network bandwidth)

### Training Output Example

```
🚀 Starting DDP training with 2 processes
📊 Training parameters:
   - Epochs: 50
   - Batch size per GPU: 32
   - Learning rate: 0.01
   - Data size: 2000
   - Hidden size: 64
   - World size: 2

[Rank 0] Process initialized on gpu-node, local_rank: 0, world_size: 2
[Rank 1] Process initialized on gpu-node1, local_rank: 0, world_size: 2

🎯 Model architecture:
   - Input size: 1
   - Hidden size: 64
   - Output size: 1
   - Total parameters: 4481

🔄 Starting training on 2 GPUs...
Epoch [1/50] - Train Loss: 0.892156, Eval Loss: 0.887432
Epoch [2/50] - Train Loss: 0.456789, Eval Loss: 0.445123
...
✅ Training completed in 15.67 seconds
⚡ Average time per epoch: 0.31 seconds
```

## 🐛 Troubleshooting

### Common Issues

1. **SSH Connection Failed**:
   ```bash
   # Check SSH key authentication
   ssh -v gpu-node1
   
   # Verify SSH agent
   ssh-add -l
   ```

2. **CUDA Not Available**:
   ```bash
   # Check CUDA installation
   nvidia-smi
   
   # Verify PyTorch CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Port Already in Use**:
   ```bash
   # Use different port
   python launch_ddp_training.py --mode multi-node --port 29501
   ```

4. **Network Issues**:
   ```bash
   # Test network connectivity
   ping gpu-node1
   
   # Check firewall
   telnet gpu-node1 29500
   ```

### Debug Mode

Run with debug information:
```bash
export TORCH_DISTRIBUTED_DEBUG=INFO
python launch_ddp_training.py --mode multi-node
```

## 📚 Learning Resources

- **PyTorch DDP Tutorial**: [Official PyTorch DDP Guide](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- **Distributed Training**: [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- **torchrun**: [PyTorch Elastic Documentation](https://pytorch.org/docs/stable/elastic/run.html)

## 🚀 Next Steps

After running these examples, you can:

1. **Scale to More Nodes**: Add more GPU nodes to `hosts.txt`
2. **Custom Models**: Replace SimpleNet with your own models
3. **Real Datasets**: Use actual datasets instead of synthetic data
4. **Advanced Features**: Add checkpointing, logging, and monitoring
5. **Performance Optimization**: Experiment with different batch sizes and learning rates

## 🤝 Contributing

Feel free to submit issues and enhancement requests! Contributions are welcome for:
- Additional distributed training examples
- Performance optimizations
- Multi-node setup automation
- Advanced DDP features

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 