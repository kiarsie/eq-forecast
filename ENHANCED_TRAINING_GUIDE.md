# 🚀 Enhanced Earthquake Forecasting Training Guide

## 📋 **Problem Analysis & Solution**

### **🔍 What Was Happening:**

1. **Overfitting Fixed**: Your regularization (dropout + weight decay) successfully prevented memorization
2. **New Problem**: **Underfitting** - validation loss plateaued early at ~15-20 epochs
3. **Root Cause**: Model capacity too small + sparse data quality issues

### **✅ What We Fixed:**

#### **1. Balanced Model Architecture**

- **Before**: `hidden_sizes=(32, 32, 16, 16)` - Too restrictive
- **After**: `hidden_sizes=(64, 48, 24, 24)` - Balanced capacity
- **Why**: 2x larger than before, but not as large as original `(120, 90, 30, 30)`

#### **2. Improved Training Parameters**

- **Patience**: Increased from 8 → 15 epochs
- **Dropout**: Reduced from 0.3 → 0.2 (better capacity utilization)
- **Weight Decay**: Kept at 1e-4 (good regularization)

#### **3. Enhanced Data Processing**

- **Rolling Features**: 3, 5, and 7-year rolling averages
- **Trend Features**: Linear trends over lookback period
- **Better Normalization**: RobustScaler + log transformation for magnitudes
- **Feature Engineering**: Handles sparse earthquake data better

## 🏗️ **New Architecture**

### **Model Capacity:**

```python
hidden_sizes = (64, 48, 24, 24)  # Total: 160 neurons vs 96 before
```

- **Layer 1**: 64 → Captures basic temporal patterns
- **Layer 2**: 48 → Learns intermediate features
- **Layer 3**: 24 → Refines predictions
- **Layer 4**: 24 → Final output layer

### **Enhanced Features:**

```python
# Basic features (2)
- max_magnitude, frequency

# Rolling features (12)
- mag_rolling_3, mag_rolling_5, mag_rolling_7
- freq_rolling_3, freq_rolling_5, freq_rolling_7
- mag_std_3, mag_std_5, mag_std_7
- freq_std_3, freq_std_5, freq_std_7

# Trend features (2)
- mag_trend, freq_trend

# Total: 16 input features vs 2 before
```

## 🚀 **How to Use the Enhanced Training**

### **Option 1: Use Enhanced Trainer (Recommended)**

```bash
python train_enhanced.py \
    --data_path data/processed_earthquake_catalog_lstm_ready.csv \
    --save_dir data/results/enhanced \
    --model_types simple attention \
    --num_epochs 100 \
    --patience 15 \
    --batch_size 32
```

### **Option 2: Update Existing Code**

```python
from src.models.enhanced_trainer import EnhancedQuadtreeTrainer

trainer = EnhancedQuadtreeTrainer(
    data_path="data/processed_earthquake_catalog_lstm_ready.csv",
    save_dir="data/results/enhanced",
    model_types=['simple', 'attention'],
    hidden_sizes=(64, 48, 24, 24),  # Balanced capacity
    patience=15,  # Increased patience
    dropout=0.2,  # Reduced dropout
    add_rolling_features=True,  # Enable enhanced features
    rolling_windows=[3, 5, 7]
)

results = trainer.train_models()
```

## 📊 **Expected Improvements**

### **Training Behavior:**

- ✅ **No More Early Plateau**: Validation loss should improve beyond 15-20 epochs
- ✅ **Better Convergence**: Training and validation loss should be closer
- ✅ **Stable Learning**: Less erratic loss curves

### **Performance Metrics:**

- 📈 **Lower Validation Loss**: Should decrease more consistently
- 📈 **Better Generalization**: Gap between train/val loss should be smaller
- 📈 **Improved Predictions**: Better earthquake forecasting accuracy

## 🔧 **Fine-Tuning Options**

### **If Still Underfitting:**

```python
# Increase capacity further
hidden_sizes = (80, 60, 32, 32)  # 25% larger

# Reduce regularization
dropout = 0.1  # Less dropout
weight_decay = 5e-5  # Less weight decay
```

### **If Overfitting Returns:**

```python
# Reduce capacity
hidden_sizes = (48, 36, 20, 20)  # 25% smaller

# Increase regularization
dropout = 0.3  # More dropout
weight_decay = 2e-4  # More weight decay
```

### **Optimize Learning:**

```python
# Better learning rate scheduling
patience = 20  # More patience
batch_size = 16  # Smaller batches for better gradients
learning_rate = 0.0005  # Lower learning rate
```

## 📁 **File Structure**

```
eq-forecast/
├── src/models/
│   ├── enhanced_data_processor.py    # NEW: Enhanced data handling
│   ├── enhanced_trainer.py           # UPDATED: Balanced architecture
│   └── quadtree_trainer.py          # UPDATED: Balanced architecture
├── train_enhanced.py                 # NEW: Enhanced training script
├── ENHANCED_TRAINING_GUIDE.md       # This guide
└── data/
    └── results/
        └── enhanced/                 # NEW: Enhanced results directory
```

## 🎯 **Next Steps**

### **1. Test Enhanced Training**

```bash
python train_enhanced.py --data_path data/processed_earthquake_catalog_lstm_ready.csv
```

### **2. Monitor Training Progress**

- Watch for validation loss improvement beyond 15 epochs
- Check if train/val loss gap is reasonable
- Look for stable convergence patterns

### **3. Evaluate Results**

- Compare with previous training runs
- Check if validation loss plateau is resolved
- Analyze prediction accuracy improvements

### **4. Fine-tune if Needed**

- Adjust hidden sizes based on performance
- Tune patience and learning rate
- Experiment with different rolling window sizes

## 🚨 **Troubleshooting**

### **Common Issues:**

#### **Memory Errors:**

```python
# Reduce batch size
batch_size = 16  # or even 8

# Reduce rolling windows
rolling_windows = [3, 5]  # Remove largest window
```

#### **Training Too Slow:**

```python
# Increase batch size
batch_size = 64

# Reduce rolling windows
rolling_windows = [5]  # Single window
```

#### **Still Underfitting:**

```python
# Increase model capacity
hidden_sizes = (80, 60, 32, 32)

# Reduce dropout
dropout = 0.1
```

## 📈 **Success Metrics**

You'll know the enhanced training is working when:

1. **Validation Loss**: Continues decreasing beyond epoch 15
2. **Convergence**: Train and validation loss curves are closer together
3. **Stability**: Less erratic loss patterns during training
4. **Performance**: Better earthquake forecasting accuracy on test data

## 🎉 **Summary**

The enhanced training addresses your validation plateau issue by:

- **Balancing Model Capacity**: Not too small, not too large
- **Improving Data Quality**: Better features and normalization
- **Optimizing Training**: Better patience and learning parameters
- **Handling Sparse Data**: Rolling features for temporal patterns

This should resolve the underfitting while maintaining good generalization!
