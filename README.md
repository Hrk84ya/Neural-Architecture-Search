# Neural Architecture Search (NAS) with Professional GUI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional web-based Neural Architecture Search implementation with evolutionary algorithms for automatically discovering optimal CNN architectures. Features a modern, interactive GUI for real-time monitoring and control.

## 🎯 Features

### Core NAS Engine
- 🧬 **Evolutionary Search**: Population-based algorithm exploring architecture space
- 🏗️ **Modular Architecture Space**: Conv, Separable, Residual, Inception blocks
- ⚡ **Efficient Training**: Early stopping, learning rate reduction, data augmentation
- 🔄 **Reproducibility**: Seeding support for consistent results

### Professional Web GUI
- 🌐 **Modern Interface**: Professional blue gradient design with glass morphism
- 📊 **Real-time Monitoring**: Live progress tracking and status updates
- 📋 **Interactive Logs**: Terminal-style logs with color coding
- 📈 **Visualizations**: Dynamic charts showing search evolution
- ⚙️ **Easy Configuration**: Intuitive parameter controls with validation
- 💾 **Export Results**: Save architectures and search results

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch GUI:**
   ```bash
   python run.py
   ```
   
3. **Open browser:** http://localhost:5001

## 📁 Project Structure

```
NAS/
├── gui.py              # Main web application
├── nas.py              # Core NAS implementation
├── run.py              # Simple launcher
├── static/
│   ├── style.css       # Professional styling
│   └── app.js          # Interactive JavaScript
├── templates/
│   └── index.html      # Web interface
└── requirements.txt    # Dependencies
```

## 🎛️ GUI Features

### Configuration Panel
- **Dataset Settings**: Training/validation sample sizes
- **Evolution Parameters**: Population size, generations, mutation rate
- **Training Settings**: Epochs, batch size, early stopping

### Results Panel
- **Search Logs**: Real-time progress with timestamps
- **Best Architecture**: JSON display of discovered architecture
- **Visualization**: Interactive charts showing search evolution

### Controls
- **Start/Stop Search**: Control search execution
- **Export Results**: Save architectures and results
- **Real-time Updates**: Live status and progress tracking

## ⚙️ Configuration

Configure search parameters through the web interface or modify defaults in `gui.py`:

```python
config = SearchConfig(
    population_size=4,      # Architectures per generation
    max_generations=3,      # Evolution generations
    mutation_rate=0.3,      # Mutation probability
    epochs_per_eval=5,      # Training epochs per architecture
    batch_size=64,          # Training batch size
    early_stopping_patience=3
)
```

## 📊 Output

- **Real-time GUI**: Live monitoring and control
- **Architecture JSON**: Best discovered architecture
- **Search Results**: Complete evolution history
- **Visualizations**: Performance charts and evolution plots
- **Export Files**: Downloadable results in JSON format

## 🔧 Advanced Usage

### Command Line (Core NAS only)
```bash
python nas.py
```

### Direct GUI Launch
```bash
python gui.py
```

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.8.0+
- Flask 2.0+
- NumPy 1.19.2+
- Matplotlib 3.5.0+

## 📈 Results

The GUI provides:
- Real-time search progress
- Best validation accuracy tracking
- Architecture evolution visualization
- Complete search history
- Exportable results

## 🚀 Next Steps

- **Extended Training**: Train best model for more epochs
- **Evaluation**: Test on full test set
- **Transfer Learning**: Apply to other datasets
- **Deployment**: Convert to TensorFlow Lite
- **Scaling**: Distributed search across multiple GPUs

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
