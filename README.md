# Earthquake Forecasting System with Shared LSTM Architecture

A comprehensive earthquake forecasting system that uses **adaptive quadtree binning** and **shared LSTM neural networks** to predict earthquake frequency and magnitude across spatial regions.

## Features

### Quadtree Spatial Binning

- **Adaptive binning** based on earthquake density
- **Non-overlapping bins** ensuring complete coverage
- **Custom spatial boundaries** for focused regions
- **Automatic merging** of low-event-count bins
- **L-shaped and irregular bin shapes** supported

### Deep Learning Models

- **Shared LSTM** - Single model across all spatial bins with dual output heads
- **Attention LSTM** - Enhanced with attention mechanisms
- **Dual targets** - Predicts both frequency and magnitude simultaneously
- **10-year lookback** window for temporal patterns
- **Early stopping** and learning rate scheduling

### Comprehensive Evaluation

- **Performance metrics**: MAE, MSE, RMSE, WMAPE, Forecast Accuracy
- **Model comparison** between Simple and Attention LSTM
- **Training progress visualization** with loss curves and learning rates
- **Overfitting detection** indicators

## System Architecture

```
eq-forecast/
├── src/                          # Core source code
│   ├── binning/                  # Quadtree implementation
│   ├── preprocessing/            # Earthquake data processing
│   └── models/                   # LSTM model definitions
├── data/                         # Trained models & results
│   └── results/                  # Trained models + plots
├── notebooks/                    # Jupyter notebooks for analysis
├── main.py                       # Main system entry point
├── requirements.txt              # Python dependencies
├── config_example.json          # Configuration template
└── README.md                     # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the System

#### Preprocessing (Earthquake Data + Quadtree Binning)

```bash
python main.py --mode preprocess --input_data data/eq_catalog.csv
```

#### Training (Shared LSTM Models)

```bash
python main.py --mode train --model_type both --input_data output/prep/processed_earthquake_catalog_lstm_ready.csv
```

#### Testing (Model Evaluation)

```bash
python main.py --mode test --model_type both --input_data output/prep/processed_earthquake_catalog_lstm_ready.csv
```

#### Full Pipeline (All Steps)

```bash
python main.py --mode full_pipeline --model_type both --input_data data/eq_catalog.csv
```

## Available Modes

| Mode             | Description                              | Input Data                |
| ---------------- | ---------------------------------------- | ------------------------- |
| `preprocess`     | Filter earthquakes, create quadtree bins | Raw earthquake catalog    |
| `train`          | Train LSTM models                        | Processed earthquake data |
| `test`           | Evaluate model performance               | Processed earthquake data |
| `validate`       | Validate on validation set               | Processed earthquake data |
| `evaluate`       | Comprehensive evaluation                 | Processed earthquake data |
| `full_pipeline`  | Run all steps sequentially               | Raw earthquake catalog    |
| `compare_models` | Compare Simple vs Attention LSTM         | Processed earthquake data |

## Model Types

| Type        | Description             | Use Case             |
| ----------- | ----------------------- | -------------------- |
| `simple`    | Shared LSTM baseline    | Basic forecasting    |
| `attention` | Attention-enhanced LSTM | Advanced forecasting |
| `both`      | Train and compare both  | Performance analysis |

## Current System Status

### What's Working:

- **Shared LSTM architecture** with dual output heads
- **Quadtree binning** with custom boundaries
- **Model training** with resume capability
- **Performance evaluation** and comparison
- **Training visualization** and progress tracking

### Key Files:

- **Main Script**: `main.py` - System entry point
- **Models**: `data/results/*.pth` (trained models)
- **Plots**: `data/results/*.png` (training progress + comparisons)
- **Configuration**: `config_example.json` - System parameters
- **Notebooks**: `notebooks/` - Analysis and exploration

## Configuration

### System Parameters

```json
{
  "preprocessing": {
    "min_depth_km": 70.0,
    "lookback_years": 10,
    "target_horizon": 1
  },
  "model_architecture": {
    "hidden_sizes": [120, 90, 30, 30],
    "dropout": 0.2,
    "activation": "sigmoid"
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "patience": 20
  }
}
```

### LSTM Parameters

- **Hidden layer sizes**: [120, 90, 30, 30]
- **Lookback window**: 10 years of historical data
- **Training epochs**: 100 maximum
- **Early stopping**: 20 epochs patience
- **Batch size**: 32
- **Learning rate**: 0.001

## Performance Metrics

The system evaluates models using:

- **MAE** - Mean Absolute Error
- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **WMAPE** - Weighted Mean Absolute Percentage Error
- **Forecast Accuracy** - 100% - WMAPE

## Outputs

### Training Results

- **Model files** (.pth) for each model type
- **Training progress plots** showing loss curves and learning rates
- **Model comparison plots** between Simple and Attention LSTM
- **Performance metrics** for each model

### Quadtree Visualization

- **Unmerged bins** showing initial quadtree structure
- **Merged bins** showing final spatial coverage
- **Event distribution** across bins

## Troubleshooting

### Common Issues:

1. **CUDA out of memory** - Reduce batch size or use CPU
2. **Model loading errors** - Check file paths in `data/results/`
3. **Plot generation fails** - Ensure matplotlib and seaborn are installed

### Resume Training:

The system automatically detects existing models and resumes training from where it left off. No need to retrain from scratch!

## Technical Details

### Data Processing

- **Shallow earthquake filtering** (<70km depth)
- **Z-score normalization** for LSTM input
- **10-year sliding windows** for temporal sequences
- **Spatial binning** using adaptive quadtree

### Model Architecture

- **Shared LSTM layers** across all spatial bins
- **Dual output heads** for frequency and magnitude
- **Dropout regularization** (20%) to prevent overfitting
- **Adam optimizer** with learning rate scheduling
- **Early stopping** based on validation loss

### Spatial Coverage

- **Complete coverage** of earthquake catalog extent
- **Non-overlapping bins** ensuring unique spatial regions
- **Adaptive sizing** based on event density
- **Custom boundaries** for focused analysis

## Contributing

This system is designed for earthquake forecasting research. Key areas for improvement:

- **Additional model architectures** (Transformers, Graph Neural Networks)
- **Enhanced spatial binning** algorithms
- **Real-time forecasting** capabilities
- **Multi-hazard integration**

## License

This project is for research and educational purposes. Please cite appropriately if used in academic work.

---

**Ready to forecast earthquakes? Start with preprocessing mode!**
