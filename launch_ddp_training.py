#!/usr/bin/env python3
"""
DDP Training Launcher

This script helps launch distributed training across multiple nodes.
It handles the configuration and setup for both single-node and multi-node training.

Usage:
    # Single node training (2 GPUs)
    python launch_ddp_training.py --mode single-node --gpus 2
    
    # Multi-node training (run on master node)
    python launch_ddp_training.py --mode multi-node --master-node gpu-node --worker-nodes gpu-node1
    
    # Multi-node training with custom configuration
    python launch_ddp_training.py --mode multi-node --master-node gpu-node --worker-nodes gpu-node1 --gpus-per-node 2 --port 29500
"""

import os
import sys
import subprocess
import argparse
import socket
import time
from typing import List, Optional


def check_ssh_connection(hostname: str, timeout: int = 5) -> bool:
    """Check if SSH connection to hostname is possible."""
    try:
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes', hostname, 'echo "SSH connection successful"'],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def get_local_ip() -> str:
    """Get the local IP address."""
    try:
        # Connect to a dummy IP to get the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'


def run_single_node_training(args):
    """Run single-node multi-GPU training."""
    print(f"🚀 Starting single-node training with {args.gpus} GPUs")
    
    # Build the torchrun command
    cmd = [
        'torchrun',
        f'--nproc_per_node={args.gpus}',
        'hello_pytorch_ddp.py',
        f'--epochs={args.epochs}',
        f'--batch-size={args.batch_size}',
        f'--lr={args.learning_rate}',
        f'--data-size={args.data_size}',
        f'--hidden-size={args.hidden_size}'
    ]
    
    print(f"📝 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Single-node training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1


def run_multi_node_training(args):
    """Run multi-node training."""
    master_node = args.master_node
    worker_nodes = args.worker_nodes
    
    print(f"🚀 Starting multi-node training")
    print(f"   Master node: {master_node}")
    print(f"   Worker nodes: {', '.join(worker_nodes)}")
    print(f"   GPUs per node: {args.gpus_per_node}")
    print(f"   Total GPUs: {args.gpus_per_node * (len(worker_nodes) + 1)}")
    
    # Check SSH connections to all nodes
    all_nodes = [master_node] + worker_nodes
    print(f"\n🔍 Checking SSH connections...")
    
    for node in all_nodes:
        if not check_ssh_connection(node):
            print(f"❌ Cannot connect to {node} via SSH")
            print(f"   Please ensure SSH key-based authentication is set up")
            return 1
        else:
            print(f"✅ SSH connection to {node} successful")
    
    # Get master node IP
    master_ip = get_local_ip()
    print(f"\n🌐 Master node IP: {master_ip}")
    
    # Build the base command
    base_cmd = [
        'torchrun',
        f'--nproc_per_node={args.gpus_per_node}',
        f'--nnodes={len(all_nodes)}',
        f'--master_addr={master_ip}',
        f'--master_port={args.port}',
        'hello_pytorch_ddp.py',
        f'--epochs={args.epochs}',
        f'--batch-size={args.batch_size}',
        f'--lr={args.learning_rate}',
        f'--data-size={args.data_size}',
        f'--hidden-size={args.hidden_size}'
    ]
    
    # Launch training on all nodes
    processes = []
    
    try:
        # Launch on worker nodes first
        for i, worker_node in enumerate(worker_nodes):
            node_rank = i + 1
            cmd = ['ssh', worker_node] + base_cmd + [f'--node_rank={node_rank}']
            
            print(f"\n🎯 Starting training on worker node {worker_node} (rank {node_rank})")
            print(f"   Command: {' '.join(cmd[2:])}")  # Skip ssh and hostname
            
            process = subprocess.Popen(cmd)
            processes.append((worker_node, process))
            time.sleep(2)  # Small delay between launches
        
        # Launch on master node (rank 0)
        master_cmd = base_cmd + ['--node_rank=0']
        print(f"\n🎯 Starting training on master node {master_node} (rank 0)")
        print(f"   Command: {' '.join(master_cmd)}")
        
        master_process = subprocess.Popen(master_cmd)
        processes.append((master_node, master_process))
        
        # Wait for all processes to complete
        print(f"\n⏳ Waiting for training to complete...")
        
        # Wait for master process (this will show the training progress)
        master_process.wait()
        
        # Wait for worker processes
        for node, process in processes[:-1]:  # Exclude master process
            process.wait()
            print(f"✅ Training completed on {node}")
        
        print(f"\n🎉 Multi-node training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        print(f"🧹 Cleaning up processes...")
        
        # Terminate all processes
        for node, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Process on {node} terminated")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️  Process on {node} killed")
            except Exception as e:
                print(f"❌ Error terminating process on {node}: {e}")
        
        return 1
    
    except Exception as e:
        print(f"❌ Multi-node training failed: {e}")
        return 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='DDP Training Launcher')
    
    # Mode selection
    parser.add_argument('--mode', choices=['single-node', 'multi-node'], required=True,
                        help='Training mode: single-node or multi-node')
    
    # Single-node options
    parser.add_argument('--gpus', type=int, default=2,
                        help='Number of GPUs for single-node training (default: 2)')
    
    # Multi-node options
    parser.add_argument('--master-node', type=str, default='gpu-node',
                        help='Master node hostname (default: gpu-node)')
    parser.add_argument('--worker-nodes', type=str, nargs='+', default=['gpu-node1'],
                        help='Worker node hostnames (default: gpu-node1)')
    parser.add_argument('--gpus-per-node', type=int, default=1,
                        help='Number of GPUs per node (default: 1)')
    parser.add_argument('--port', type=int, default=29500,
                        help='Master port (default: 29500)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size per GPU (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--data-size', type=int, default=2000,
                        help='Size of synthetic dataset (default: 2000)')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Hidden layer size (default: 64)')
    
    args = parser.parse_args()
    
    print("🔥 PyTorch DDP Training Launcher")
    print("=" * 50)
    
    # Check if training script exists
    if not os.path.exists('hello_pytorch_ddp.py'):
        print("❌ Error: hello_pytorch_ddp.py not found in current directory")
        return 1
    
    # Run training based on mode
    if args.mode == 'single-node':
        return run_single_node_training(args)
    else:
        return run_multi_node_training(args)


if __name__ == "__main__":
    sys.exit(main()) 