# Cat/Dog Image Classification Project

A comprehensive deep learning project for classifying cat and dog images using PyTorch. This project implements a complete pipeline including dataset management, model architectures, training, inference, and evaluation.

## Project Structure

```
cat-dog-recog/
├── dataset.py          # Dataset management and loading
├── model.py           # Model architectures (LeNet, CustomCNN, ResNet, EfficientNet)
├── trainer.py         # Training loop with hyperparameter configuration
├── inference.py       # Inference engine for predictions
├── evaluate.py        # Comprehensive model evaluation
├── main.py           # Main entry point with CLI
├── code_by_bob.py    # Original TensorFlow implementation (reference)
├── pyproject.toml    # Project dependencies
└── README.md         # This file
```

## Features

### 1. Dataset Module (`dataset.py`)
- Downloads cat/dog dataset from Microsoft or Kaggle
- Automatic dataset organization into train/val/test splits
- Data augmentation and preprocessing
- PyTorch DataLoader creation

### 2. Model Module (`model.py`)
- Multiple model architectures:
  - **LeNet**: Lightweight CNN for quick experiments
  - **CustomCNN**: Custom CNN with batch normalization
  - **ResNet18/34/50**: Pretrained ResNet models with transfer learning
  - **EfficientNet-B0/B1/B2**: Efficient pretrained models
- Model factory for easy model creation
- Model information and comparison

### 3. Trainer Module (`trainer.py`)
- Complete training loop with validation
- Multiple optimizers (Adam, SGD, AdamW)
- Learning rate schedulers (Step, Plateau, Cosine)
- Early stopping and checkpoint saving
- Hyperparameter configuration class
- Training history tracking

### 4. Inference Module (`inference.py`)
- Single image and batch prediction
- Confidence scores and probability distributions
- Visualization of predictions
- Model loading utilities
- Test directory evaluation

### 5. Evaluation Module (`evaluate.py`)
- Accuracy evaluation with confusion matrix
- Computational efficiency analysis (FPS, memory usage)
- Robustness testing with noise injection
- Model comparison framework
- Comprehensive report generation with plots

## Installation

1. Ensure you have Python 3.13+ installed
2. Install UV package manager (if not already installed):
   ```bash
   pip install uv
   ```
3. Install project dependencies:
   ```bash
   uv sync
   ```

## Usage

### Basic Training
```bash
python main.py --mode train
```

### Inference with Trained Model
```bash
python main.py --mode inference --model-path checkpoints/best_model.pth
```

### Model Evaluation
```bash
python main.py --mode evaluate --model-path checkpoints/best_model.pth
```

### Model Comparison
```bash
python main.py --mode compare
```

### Complete Pipeline
```bash
python main.py --mode all
```

## Command Line Options

- `--mode`: Pipeline mode (`train`, `inference`, `evaluate`, `compare`, `all`)
- `--model-path`: Path to model checkpoint for inference/evaluation
- `--download-data`: Download dataset before training

## Example Workflow

1. **Download and prepare dataset**:
   ```python
   from dataset import DatasetManager
   manager = DatasetManager(data_dir="data")
   manager.download_dataset("microsoft")
   ```

2. **Create and train model**:
   ```python
   from model import ModelFactory
   from trainer import Trainer, HyperparameterConfig
   
   model = ModelFactory.create_model("custom_cnn", num_classes=2)
   trainer = Trainer(model)
   history = trainer.train(train_loader, val_loader, epochs=10)
   ```

3. **Make predictions**:
   ```python
   from inference import InferenceEngine
   engine = InferenceEngine(model)
   prediction, confidence = engine.predict("test_image.jpg", return_probabilities=True)
   ```

4. **Evaluate model**:
   ```python
   from evaluate import ModelEvaluator
   evaluator = ModelEvaluator(model)
   accuracy_results = evaluator.evaluate_accuracy(test_loader)
   efficiency_results = evaluator.evaluate_efficiency()
   ```

## Model Architectures

| Model | Parameters | Description | Best Use Case |
|-------|------------|-------------|---------------|
| LeNet | ~60K | Classic CNN architecture | Quick experiments, education |
| CustomCNN | ~1.2M | Custom CNN with batch norm | Balanced performance/speed |
| ResNet18 | ~11M | Pretrained on ImageNet | Good accuracy with transfer learning |
| EfficientNet-B0 | ~5M | Efficient pretrained model | Best accuracy with efficiency |

## Dataset

The project uses the Microsoft Cat vs Dog dataset (alternative to Kaggle dataset). The dataset is automatically downloaded and organized into:

```
data/
├── train/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

## Results

The evaluation module provides comprehensive metrics:
- **Accuracy**: Overall and per-class accuracy
- **Efficiency**: Inference time, FPS, memory usage
- **Robustness**: Performance under noise
- **Visualizations**: Confusion matrix, ROC curves, efficiency charts

## Dependencies

- PyTorch 2.9.1+
- torchvision 0.24.1+
- matplotlib 3.10.8+
- numpy 2.4.0+
- scikit-learn (for evaluation metrics)
- tqdm (for progress bars)
- Pillow (for image processing)

## License

This project is for educational purposes. The dataset is from Microsoft Research and should be used according to their terms.

## Acknowledgments

- Microsoft Research for the cat/dog dataset
- PyTorch team for the deep learning framework
- Original TensorFlow implementation by "Bob" for inspiration