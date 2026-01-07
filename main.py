"""
Main entry point for cat/dog image classification project.
Demonstrates usage of all modules: dataset, model, trainer, inference, and evaluate.
"""

import argparse
from math import ceil
from pathlib import Path
import os
from torchinfo import summary as model_summary

CPU_NUM = os.cpu_count() or 1
DATA_LOADER_WORKER_NUM = ceil(CPU_NUM / 4 * 3)


def train_pipeline():
    """Complete training pipeline."""
    print("CAT/DOG CLASSIFICATION - TRAINING PIPELINE")

    try:
        # Import modules
        from dataset import DatasetManager
        from model import ModelFactory
        from trainer import Trainer, HyperparameterConfig

        # Step 1: Setup dataset
        print("\n1. Setting up dataset...")
        dataset_manager = DatasetManager(data_dir="data")

        # Check if dataset exists, otherwise download
        if not (dataset_manager.data_dir / "train").exists():
            print("Dataset not found. Downloading...")
            dataset_manager.download_dataset("microsoft")
        else:
            print("Dataset found. Skipping download.")

        # Create dataloaders
        train_loader, val_loader = dataset_manager.create_dataloaders(
            batch_size=32,
            num_workers=DATA_LOADER_WORKER_NUM,
            augment=True
        )
        print(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")

        # Step 2: Create model
        print("\n2. Creating model...")
        model = ModelFactory.create_model(
            model_name="custom_cnn",
            num_classes=2
        )
        print("Model created: CustomCNN")
        model_summary(model)

        # Step 3: Setup trainer and hyperparameters
        print("\n3. Configuring training...")
        config = HyperparameterConfig(
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            optimizer='adam',
            criterion='cross_entropy',
            scheduler='plateau',
            patience=5
        )
        print(config)

        trainer = Trainer(model, checkpoint_dir="checkpoints")
        print(trainer.get_model_summary())

        # Step 4: Train model
        print("\n4. Starting training...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            optimizer_name=config.optimizer,
            criterion_name=config.criterion,
            scheduler_name=config.scheduler,
            patience=config.patience,
            save_best=True
        )

        print("\nTraining completed successfully!")
        print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")

        return trainer.best_model_path

    except Exception as e:
        print(f"\nError in training pipeline: {e}")
        return None


def inference_pipeline(model_path: Path):
    """Complete inference pipeline."""
    print("CAT/DOG CLASSIFICATION - INFERENCE PIPELINE")

    try:
        # Import modules
        from model import CustomCNN
        from inference import ModelLoader

        # Step 1: Load trained model
        print("\n1. Loading trained model...")
        engine = ModelLoader.load_model(
            checkpoint_path=model_path,
            model_class=CustomCNN,
        )

        # Step 2: Test on sample images
        print("\n2. Testing inference...")

        # Create a test image (red square for cat, blue square for dog)
        from PIL import Image

        # Create test images
        test_images = []
        for color, label in [('red', 'cat'), ('blue', 'dog')]:
            img = Image.new('RGB', (224, 224), color=color)
            test_images.append((img, label))

        # Make predictions
        print("\nSample predictions:")
        for img, expected_label in test_images:
            prediction, confidence = engine.predict(img, return_probabilities=True)
            print(f"  Expected: {expected_label:4s} | Predicted: {prediction:4s} | Confidence: {confidence:.2%}")

        # Step 3: Detailed prediction example
        print("\n3. Detailed prediction analysis:")
        detailed = engine.predict_with_confidence(test_images[0][0])
        print(f"  Predicted class: {detailed['predicted_class']}")
        print(f"  Confidence: {detailed['confidence']:.2%}")
        print("  Class probabilities:")
        for class_name, prob in detailed['class_probabilities'].items():
            print(f"    {class_name}: {prob:.2%}")

        return engine

    except Exception as e:
        print(f"\nError in inference pipeline: {e}")
        return None


def evaluation_pipeline(model_path: Path):
    """Complete evaluation pipeline."""
    print("CAT/DOG CLASSIFICATION - EVALUATION PIPELINE")

    try:
        # Import modules
        from dataset import DatasetManager
        from model import CustomCNN
        from evaluate import ModelEvaluator

        # Step 1: Load model
        print("\n1. Loading model for evaluation...")
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        model = CustomCNN(num_classes=2)
        model_summary(model)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Step 2: Create evaluator
        print("\n2. Creating evaluator...")
        evaluator = ModelEvaluator(model)

        # Step 3: Setup test dataset
        print("\n3. Preparing test dataset...")
        dataset_manager = DatasetManager(data_dir="data")

        if not (dataset_manager.data_dir / "test").exists():
            print("Test dataset not found. Using validation split...")
            _, test_loader = dataset_manager.create_dataloaders(
                batch_size=32,
                num_workers=DATA_LOADER_WORKER_NUM,
                augment=False
            )
        else:
            # Create test dataloader
            from dataset import CatDogDataset
            _, test_transform = dataset_manager.get_transforms(augment=False)
            test_dataset = CatDogDataset(
                root_dir=str(dataset_manager.data_dir),
                transform=test_transform,
                train=False
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=DATA_LOADER_WORKER_NUM
            )

        # Step 4: Evaluate accuracy
        print("\n4. Evaluating accuracy...")
        evaluator.evaluate_accuracy(test_loader, verbose=True)

        # Step 5: Evaluate efficiency
        print("\n5. Evaluating computational efficiency...")
        evaluator.evaluate_efficiency(verbose=True)

        # Step 6: Generate comprehensive report
        print("\n6. Generating evaluation report...")
        report = evaluator.generate_report(plot=True)

        print("\nEvaluation completed successfully!")

        return report

    except Exception as e:
        print(f"\nError in evaluation pipeline: {e}")
        return None


def compare_models_pipeline():
    """Compare different model architectures."""
    print("CAT/DOG CLASSIFICATION - MODEL COMPARISON")

    try:
        # Import modules
        from dataset import DatasetManager
        from model import ModelFactory
        from evaluate import ModelEvaluator

        # Step 1: Setup dataset
        print("\n1. Preparing dataset for comparison...")
        dataset_manager = DatasetManager(data_dir="data")

        if not (dataset_manager.data_dir / "train").exists():
            print("Dataset not found. Please run training pipeline first.")
            return

        # Create a small dataloader for quick comparison
        train_loader, _ = dataset_manager.create_dataloaders(
            batch_size=16,
            num_workers=1,
            augment=False
        )

        # Step 2: Create models to compare
        print("\n2. Creating models for comparison...")
        models_to_compare = {
            'LeNet': ModelFactory.create_model('lenet', num_classes=2),
            'CustomCNN': ModelFactory.create_model('custom_cnn', num_classes=2),
            'ResNet18': ModelFactory.create_model('resnet18', num_classes=2, pretrained=True),
            'EfficientNet-B0': ModelFactory.create_model('efficientnet_b0', num_classes=2, pretrained=True),
        }

        for (name, model) in models_to_compare.items():
            print(f"  - {name}")
            model_summary(model)

        # Step 3: Compare models
        print("\n3. Comparing models...")
        evaluator = ModelEvaluator(models_to_compare['CustomCNN'])
        comparison_results = evaluator.compare_models(
            models=models_to_compare,
            dataloader=train_loader,
            input_shape=(3, 224, 224)
        )

        print("\nModel comparison completed!")

        return comparison_results

    except Exception as e:
        print(f"\nError in model comparison pipeline: {e}")
        return None


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description='Cat/Dog Image Classification')
    parser.add_argument('--mode', choices=['train', 'inference', 'evaluate', 'compare', 'all'],
                       default='all', help='Pipeline mode to run')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint for inference/evaluation')
    parser.add_argument('--download-data', action='store_true', help='Download dataset before training')

    args = parser.parse_args()

    print("Cat/Dog Image Classification Project")
    print(f"DataLoader workers: {DATA_LOADER_WORKER_NUM}")

    if args.mode in ['train', 'all']:
        model_path = train_pipeline()
        if model_path and args.mode == 'all':
            args.model_path = str(model_path)

    if args.mode in ['inference', 'all'] and args.model_path:
        inference_pipeline(Path(args.model_path))

    if args.mode in ['evaluate', 'all'] and args.model_path:
        evaluation_pipeline(Path(args.model_path))

    if args.mode in ['compare', 'all']:
        compare_models_pipeline()


if __name__ == "__main__":
    main()
