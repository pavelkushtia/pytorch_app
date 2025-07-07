#!/usr/bin/env python3
"""
PyTorch DDP Hello World

A simple distributed training example using PyTorch's DistributedDataParallel (DDP).
This script demonstrates how to train a simple neural network across multiple GPUs and nodes.

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=2 hello_pytorch_ddp.py
    
    # Multiple nodes
    # On node 0:
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=<node0_ip> --master_port=29500 hello_pytorch_ddp.py
    # On node 1:
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=<node0_ip> --master_port=29500 hello_pytorch_ddp.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import socket
from datetime import datetime


def setup_distributed():
    """Initialize the distributed process group."""
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    
    # Set the GPU for this process
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    # Get distributed info
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    
    print(f"[Rank {global_rank}] Process initialized on {socket.gethostname()}, "
          f"local_rank: {local_rank}, world_size: {world_size}")
    
    return local_rank, global_rank, world_size


def cleanup_distributed():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def create_synthetic_data(num_samples=1000, noise_std=0.5):
    """Create synthetic linear regression data."""
    torch.manual_seed(42)  # For reproducibility
    
    # Generate data: y = 2x + 1 + noise
    X = torch.linspace(-10, 10, num_samples).reshape(-1, 1)
    y = 2 * X + 1 + torch.randn(X.shape) * noise_std
    
    return X, y


class SimpleNet(nn.Module):
    """Simple neural network for linear regression."""
    
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0 and rank == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def save_visualization(model, X, y, device, rank, epoch):
    """Save visualization of the results (only on rank 0)."""
    if rank != 0:
        return
    
    model.eval()
    with torch.no_grad():
        # Generate predictions
        X_vis = torch.linspace(-10, 10, 200).reshape(-1, 1).to(device)
        y_pred = model(X_vis)
        
        # Move to CPU for plotting
        X_vis_cpu = X_vis.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()
        X_cpu = X.cpu().numpy()
        y_cpu = y.cpu().numpy()
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.scatter(X_cpu, y_cpu, alpha=0.6, label='Training Data', s=20)
        plt.plot(X_vis_cpu, y_pred_cpu, 'r-', linewidth=2, label='DDP Model Prediction')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'PyTorch DDP Hello World - Epoch {epoch}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        filename = f'pytorch_ddp_hello_world_epoch_{epoch}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Rank {rank}] Saved visualization: {filename}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='PyTorch DDP Hello World')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--data-size', type=int, default=2000, help='Size of synthetic dataset')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden layer size')
    args = parser.parse_args()
    
    # Setup distributed training
    local_rank, global_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    if global_rank == 0:
        print(f"🚀 Starting DDP training with {world_size} processes")
        print(f"📊 Training parameters:")
        print(f"   - Epochs: {args.epochs}")
        print(f"   - Batch size per GPU: {args.batch_size}")
        print(f"   - Learning rate: {args.lr}")
        print(f"   - Data size: {args.data_size}")
        print(f"   - Hidden size: {args.hidden_size}")
        print(f"   - World size: {world_size}")
    
    # Create synthetic data
    X, y = create_synthetic_data(args.data_size)
    
    # Create dataset and distributed sampler
    dataset = TensorDataset(X, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    # Create model and wrap with DDP
    model = SimpleNet(hidden_size=args.hidden_size).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if global_rank == 0:
        print(f"\n🎯 Model architecture:")
        print(f"   - Input size: 1")
        print(f"   - Hidden size: {args.hidden_size}")
        print(f"   - Output size: 1")
        print(f"   - Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    if global_rank == 0:
        print(f"\n🔄 Starting training on {world_size} GPUs...")
        start_time = time.time()
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device, global_rank)
        
        # Evaluate (optional - you can also create a separate validation set)
        eval_loss = evaluate_model(model, dataloader, criterion, device)
        
        # Print progress (only on rank 0)
        if global_rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] - "
                  f"Train Loss: {train_loss:.6f}, Eval Loss: {eval_loss:.6f}")
        
        # Save visualization periodically
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            save_visualization(model, X, y, device, global_rank, epoch + 1)
    
    if global_rank == 0:
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.2f} seconds")
        print(f"⚡ Average time per epoch: {total_time/args.epochs:.2f} seconds")
    
    # Print final model parameters (only on rank 0)
    if global_rank == 0:
        print(f"\n📊 Final model parameters:")
        for name, param in model.named_parameters():
            if param.numel() <= 10:  # Only print small parameters
                print(f"   {name}: {param.data.flatten()[:5].tolist()}")
    
    # Cleanup
    cleanup_distributed()
    
    if global_rank == 0:
        print(f"\n🎉 DDP training completed successfully!")
        print(f"📈 Check the generated PNG files for training visualizations")


if __name__ == "__main__":
    main() 