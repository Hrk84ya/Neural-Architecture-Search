# Neural Architecture Search (NAS) with Evolutionary Algorithms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An implementation of Evolutionary Neural Architecture Search for automatically discovering optimal CNN architectures for image classification tasks, demonstrated on the CIFAR-10 dataset.

## Features

- 🧬 **Evolutionary Search**: Implements a population-based evolutionary algorithm to explore the architecture space
- 🏗️ **Modular Architecture Space**: Supports various CNN block types including:
  - Standard Convolutional Blocks
  - Depthwise Separable Convolutions
  - Residual Connections
  - Inception-style Blocks
- ⚡ **Efficient Training**:
  - Early Stopping
  - Learning Rate Reduction
  - Data Augmentation
- 📊 **Visualization**: Built-in plotting of search progress and architecture performance
- 🔄 **Reproducibility**: Seeding support for consistent results

## Requirements

- Python 3.8+
- TensorFlow 2.8.0+
- NumPy 1.19.2+
- Matplotlib 3.5.0+

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hrk84ya/Neural-Architecture-Search.git
   cd Neural-Architecture-Search

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the NAS search:
```bash
python nas.py
```

2. The script will:
    - Load and preprocess the CIFAR-10 dataset
    - Run the evolutionary search
    - Save the best architecture and model weights
    - Generate performance plots

## Configuration

The search can be configured by modifying the SearchConfig class parameters in nas.py:

```bash
config = SearchConfig(
    population_size=6,      # Number of architectures in each generation
    max_generations=5,      # Maximum number of generations
    mutation_rate=0.3,      # Probability of mutation
    epochs_per_eval=10,     # Training epochs per architecture
    batch_size=128,         # Batch size for training
    early_stopping_patience=5,
    reduce_lr_patience=3
)
```

## Output
The script will generate:

- best_architecture.json: The best found architecture specification
- nas_best_model.h5: Trained weights of the best model
- Performance plots showing the evolution of the search

## Results
The search process will display:

- Progress of each generation
- Best validation accuracy achieved
- Model architecture summary
- Performance metrics on the validation set

## Next Steps
- Extended Training: Train the best model for more epochs on the full dataset
- Evaluation: Test the model on the test set
- Transfer Learning: Use the discovered architecture for other image classification tasks
- Deployment: Convert to TensorFlow Lite for mobile deployment

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
