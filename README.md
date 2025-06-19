# PyTorch Hello World

A simple PyTorch application that demonstrates basic neural network functionality with automatic GPU acceleration. This example creates a simple neural network that learns to fit a linear relationship between input and output data.

## 🚀 Features

- **Automatic GPU Detection**: Automatically uses GPU if available, falls back to CPU
- **Simple Neural Network**: Demonstrates basic PyTorch concepts
- **Visualization**: Creates plots showing training results
- **Easy Setup**: Automated setup script included

## 📋 Requirements

- Python 3.8+
- PyTorch 2.2.0+ (with CUDA support if using GPU)
- NumPy 1.24.0+
- Matplotlib 3.7.0+

## 🛠️ Installation

### Option 1: Automated Setup (Recommended)
```bash
# Clone this repository
git clone <your-repo-url>
cd pytorch_app

# Run the automated setup script
./setup.sh
```

### Option 2: Manual Setup
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

### Quick Start
```bash
# Activate the environment (if not already activated)
source venv/bin/activate

# Run the PyTorch Hello World application
python hello_pytorch_gpu.py
```

### What the Script Does

1. **Data Generation**: Creates synthetic linear data with noise (y = 2x + 1 + noise)
2. **Model Definition**: Creates a simple neural network with one linear layer
3. **Training**: Trains the model for 100 epochs using SGD optimizer
4. **Visualization**: Saves a plot showing the original data and fitted line
5. **Results**: Displays learned model parameters

### Expected Output
```
Using device: cuda  # or cpu if no GPU available
Training the model on GPU...
Epoch [10/100], Loss: 0.2312
Epoch [20/100], Loss: 0.2254
...
Epoch [100/100], Loss: 0.2141

Model parameters:
linear.weight: 1.9987
linear.bias: 0.9681

Training completed on cuda! Check 'pytorch_hello_world_gpu.png' for the visualization.
```

## 🔧 Key PyTorch Concepts Demonstrated

- **Tensors**: Creating and manipulating PyTorch tensors
- **Neural Networks**: Defining models using `nn.Module`
- **Device Management**: Moving data between CPU and GPU
- **Loss Functions**: Using MSE loss for regression
- **Optimizers**: SGD optimization
- **Training Loops**: Forward pass, backward pass, and parameter updates
- **Model Evaluation**: Using `torch.no_grad()` for inference

## 📊 Understanding the Results

The model learns to approximate the function `y = 2x + 1` from noisy data:
- **Expected weight**: 2.0 (slope)
- **Expected bias**: 1.0 (y-intercept)
- **Learned weight**: ~1.9987 (very close!)
- **Learned bias**: ~0.9681 (close, considering noise)

## 🖥️ GPU vs CPU Performance

- **With GPU**: Faster training, better convergence
- **Without GPU**: Automatic fallback to CPU, still functional
- **Performance**: GPU typically 10-100x faster for neural networks

## 📁 Project Structure

```
pytorch_app/
├── hello_pytorch_gpu.py    # Main PyTorch application
├── requirements.txt        # Python dependencies
├── setup.sh               # Automated setup script
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA not available**: The script will automatically use CPU
2. **Import errors**: Make sure virtual environment is activated
3. **Permission denied**: Run `chmod +x setup.sh` to make setup script executable

### Getting Help

- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

## 📚 Next Steps

After running this example, you can:
- Modify the neural network architecture
- Try different optimizers (Adam, RMSprop)
- Experiment with different loss functions
- Add more layers to the network
- Try different datasets

## 🤝 Contributing

Feel free to submit issues and enhancement requests! 