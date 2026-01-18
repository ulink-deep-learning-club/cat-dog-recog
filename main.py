"""
Main entry point for cat/dog image classification project.
Demonstrates usage of all modules: dataset, model, trainer, inference, and evaluate.
"""

import argparse
from math import ceil
from pathlib import Path
import os
import torch
from torchinfo import summary as model_summary

CPU_NUM = os.cpu_count() or 1
DATA_LOADER_WORKER_NUM = ceil(CPU_NUM / 4 * 3)


def train_pipeline(
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    scheduler: str = "plateau",
    patience: int = 5,
    device: str = "auto",
    exp_name: str = None,
    model_name: str = "custom_cnn",
    augment: bool = True,
):
    """Complete training pipeline."""
    print("CAT/DOG CLASSIFICATION - TRAINING PIPELINE")

    try:
        # Import modules
        from dataset import DatasetManager
        from model import ModelFactory
        from trainer import Trainer, HyperparameterConfig
        from utils import auto_select_device

        # Device setup
        if device == "auto":
            device = str(auto_select_device())
        device_obj = torch.device(device)

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
            batch_size=batch_size, num_workers=DATA_LOADER_WORKER_NUM, augment=augment
        )
        print(
            f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches"
        )

        # Step 2: Create model
        print("\n2. Creating model...")
        model = ModelFactory.create_model(model_name=model_name, num_classes=2)
        print(f"Model created: {model_name}")
        model_summary(model)

        # Step 3: Setup trainer and hyperparameters
        print("\n3. Configuring training...")
        config = HyperparameterConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer,
            criterion="cross_entropy",
            scheduler=scheduler,
            patience=patience,
        )
        print(config)

        trainer = Trainer(model, exp_name=exp_name)
        trainer.save_config(config.to_dict())
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
            save_best=True,
        )

        print("\nTraining completed successfully!")
        print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
        print(f"Experiment directory: {trainer.exp_dir}")
        print(f"Best model: {trainer.best_model_path}")
        print(f"Training history: {trainer.history_dir / 'training_history.json'}")

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
        for color, label in [("red", "cat"), ("blue", "dog")]:
            img = Image.new("RGB", (224, 224), color=color)
            test_images.append((img, label))

        # Make predictions
        print("\nSample predictions:")
        for img, expected_label in test_images:
            prediction, confidence = engine.predict(img, return_probabilities=True)
            print(
                f"  Expected: {expected_label:4s} | Predicted: {prediction:4s} | Confidence: {confidence:.2%}"
            )

        # Step 3: Detailed prediction example
        print("\n3. Detailed prediction analysis:")
        detailed = engine.predict_with_confidence(test_images[0][0])
        print(f"  Predicted class: {detailed['predicted_class']}")
        print(f"  Confidence: {detailed['confidence']:.2%}")
        print("  Class probabilities:")
        for class_name, prob in detailed["class_probabilities"].items():
            print(f"    {class_name}: {prob:.2%}")

        return engine

    except Exception as e:
        print(f"\nError in inference pipeline: {e}")
        return None


def evaluation_pipeline(model_path: Path, batch_size: int = 32):
    """Complete evaluation pipeline."""
    print("CAT/DOG CLASSIFICATION - EVALUATION PIPELINE")

    try:
        # Import modules
        from dataset import DatasetManager
        from model import CustomCNN
        from evaluate import ModelEvaluator

        # Step 1: Load model
        print("\n1. Loading model for evaluation...")
        checkpoint = torch.load(model_path, map_location="cpu")
        model = CustomCNN(num_classes=2)
        model_summary(model)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Step 2: Create evaluator
        print("\n2. Creating evaluator...")
        evaluator = ModelEvaluator(model)

        # Step 3: Setup test dataset
        print("\n3. Preparing test dataset...")
        dataset_manager = DatasetManager(data_dir="data")

        if not (dataset_manager.data_dir / "test").exists():
            print("Test dataset not found. Using validation split...")
            _, test_loader = dataset_manager.create_dataloaders(
                batch_size=batch_size, num_workers=DATA_LOADER_WORKER_NUM, augment=False
            )
        else:
            # Create test dataloader
            from dataset import CatDogDataset

            _, test_transform = dataset_manager.get_transforms(augment=False)
            test_dataset = CatDogDataset(
                root_dir=str(dataset_manager.data_dir),
                transform=test_transform,
                train=False,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=DATA_LOADER_WORKER_NUM,
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
            batch_size=16, num_workers=1, augment=False
        )

        # Step 2: Create models to compare
        print("\n2. Creating models for comparison...")
        models_to_compare = {
            "LeNet": ModelFactory.create_model("lenet", num_classes=2),
            "CustomCNN": ModelFactory.create_model("custom_cnn", num_classes=2),
            "ResNet18": ModelFactory.create_model(
                "resnet18", num_classes=2, pretrained=True
            ),
            "EfficientNet-B0": ModelFactory.create_model(
                "efficientnet_b0", num_classes=2, pretrained=True
            ),
        }

        for name, model in models_to_compare.items():
            print(f"  - {name}")
            model_summary(model)

        # Step 3: Compare models
        print("\n3. Comparing models...")
        evaluator = ModelEvaluator(models_to_compare["CustomCNN"])
        comparison_results = evaluator.compare_models(
            models=models_to_compare, dataloader=train_loader, input_shape=(3, 224, 224)
        )

        print("\nModel comparison completed!")

        return comparison_results

    except Exception as e:
        print(f"\nError in model comparison pipeline: {e}")
        return None


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description="Cat/Dog Image Classification")

    # Global arguments
    parser.add_argument(
        "--mode",
        choices=["train", "inference", "evaluate", "compare", "all"],
        default="all",
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--download-data", action="store_true", help="Download dataset before training"
    )

    # Training arguments
    train_group = parser.add_argument_group("Training options")
    train_group.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    train_group.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    train_group.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="Optimizer",
    )
    train_group.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["step", "plateau", "cosine", None],
        help="LR scheduler",
    )
    train_group.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    train_group.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)"
    )
    train_group.add_argument(
        "--exp", type=str, help="Experiment name for this training run"
    )
    train_group.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["lenet", "custom_cnn", "resnet18", "efficientnet_b0"],
        help="Model architecture",
    )
    train_group.add_argument(
        "--no-augment", action="store_true", help="Disable data augmentation"
    )

    # Inference arguments
    inf_group = parser.add_argument_group("Inference options")
    inf_group.add_argument(
        "--image", type=str, help="Path to single image for inference"
    )
    inf_group.add_argument(
        "--image-dir", type=str, help="Directory of images for batch inference"
    )

    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation options")
    eval_group.add_argument(
        "--eval-batch-size", type=int, default=32, help="Batch size for evaluation"
    )

    # Model/experiment arguments
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model checkpoint for inference/evaluation",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Experiment name (e.g., exp1) for inference/evaluation",
    )

    args = parser.parse_args()

    print("Cat/Dog Image Classification Project")
    print(f"DataLoader workers: {DATA_LOADER_WORKER_NUM}")

    if args.mode in ["train", "all"]:
        model_path = train_pipeline(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            patience=args.patience,
            device=args.device,
            exp_name=args.exp,
            model_name=args.model,
            augment=not args.no_augment,
        )
        if model_path and args.mode == "all":
            args.model_path = str(model_path)
            args.exp_name = model_path.parent.parent.name

    if args.mode in ["inference", "all"] and (
        args.model_path or args.exp_name or args.image or args.image_dir
    ):
        from model import CustomCNN
        from inference import ModelLoader

        if args.exp_name:
            engine = ModelLoader.load_from_experiment(
                args.exp_name, CustomCNN, use_final=False
            )
        elif args.model_path:
            engine = ModelLoader.load_model(
                checkpoint_path=args.model_path,
                model_class=CustomCNN,
            )
        else:
            print("Error: Need --exp-name, --model-path, or --image for inference")
            return

        from PIL import Image

        if args.image:
            img = Image.open(args.image).convert("RGB")
            prediction, confidence = engine.predict(img, return_probabilities=True)
            print(f"Image: {args.image}")
            print(f"  Prediction: {prediction} | Confidence: {confidence:.2%}")
        elif args.image_dir:
            from pathlib import Path

            img_dir = Path(args.image_dir)
            image_paths = (
                list(img_dir.glob("*.jpg"))
                + list(img_dir.glob("*.jpeg"))
                + list(img_dir.glob("*.png"))
            )
            print(f"\nRunning inference on {len(image_paths)} images...")
            correct = 0
            total = 0
            for img_path in image_paths:
                pred, conf = engine.predict(img_path, return_probabilities=True)
                print(f"  {img_path.name}: {pred} ({conf:.2%})")
        else:
            test_images = []
            for color, label in [("red", "cat"), ("blue", "dog")]:
                img = Image.new("RGB", (224, 224), color=color)
                test_images.append((img, label))

            print("\nSample predictions:")
            for img, expected_label in test_images:
                prediction, confidence = engine.predict(img, return_probabilities=True)
                print(
                    f"  Expected: {expected_label:4s} | Predicted: {prediction:4s} | Confidence: {confidence:.2%}"
                )

    if args.mode in ["evaluate", "all"] and (args.model_path or args.exp_name):
        if args.exp_name:
            from model import CustomCNN
            from inference import ModelLoader
            from evaluate import ModelEvaluator
            from dataset import DatasetManager

            exp_dir = Path("runs") / args.exp_name
            checkpoint_path = exp_dir / "checkpoints" / "final_model.pth"
            if not checkpoint_path.exists():
                checkpoints = list((exp_dir / "checkpoints").glob("best_model*.pth"))
                checkpoint_path = max(
                    checkpoints,
                    key=lambda x: float(x.name.split("_acc")[1].replace(".pth", "")),
                )

            print(f"Evaluating experiment: {args.exp_name}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model = CustomCNN(num_classes=2)
            model_summary(model)
            model.load_state_dict(checkpoint["model_state_dict"])

            evaluator = ModelEvaluator(model)
            dataset_manager = DatasetManager(data_dir="data")

            from dataset import CatDogDataset

            _, test_transform = dataset_manager.get_transforms(augment=False)
            test_dataset = CatDogDataset(
                root_dir=str(dataset_manager.data_dir),
                transform=test_transform,
                train=False,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=DATA_LOADER_WORKER_NUM,
            )

            print("\nEvaluating accuracy...")
            evaluator.evaluate_accuracy(test_loader, verbose=True)
            print("\nEvaluating computational efficiency...")
            evaluator.evaluate_efficiency(verbose=True)
            print("\nGenerating evaluation report...")
            evaluator.generate_report(plot=True)
            print("\nEvaluation completed successfully!")
        else:
            evaluation_pipeline(Path(args.model_path), batch_size=args.eval_batch_size)

    if args.mode in ["compare", "all"]:
        compare_models_pipeline()


if __name__ == "__main__":
    main()
