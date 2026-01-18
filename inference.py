"""
Inference module for cat/dog image classification.
Handles model inference on single images or batches.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import json
import matplotlib.pyplot as plt
from utils import auto_select_device
import warnings

warnings.filterwarnings("ignore")


class InferenceEngine:
    """Engine for performing inference with trained models."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model: Trained PyTorch model
            device: Device to use for inference
            class_names: List of class names (e.g., ['cat', 'dog'])
        """
        self.model = model
        self.device = device or auto_select_device()
        self.model.to(self.device)
        self.model.eval()

        self.class_names = class_names or ["cat", "dog"]

        # Default transforms (should match training)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        print(f"Inference engine initialized on {self.device}")

    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for inference.

        Args:
            image: Image path, Path object, or PIL Image

        Returns:
            Preprocessed tensor
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Apply transforms
        image_tensor = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def predict(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        return_probabilities: bool = False,
    ) -> Union[str, Tuple[str, float]]:
        """
        Predict class for a single image.

        Args:
            image: Image to classify
            return_probabilities: Whether to return probability

        Returns:
            Class name (and probability if return_probabilities=True)
        """
        # Preprocess if not already a tensor
        if not isinstance(image, torch.Tensor):
            image_tensor = self.preprocess_image(image)
        else:
            image_tensor = image

        # Move to device
        image_tensor = image_tensor.to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

            # Get probabilities using softmax
            if outputs.shape[1] > 1:  # Multi-class classification
                probabilities = torch.softmax(outputs, dim=1)
            else:  # Binary classification
                probabilities = torch.sigmoid(outputs)
                probabilities = torch.cat([1 - probabilities, probabilities], dim=1)

            # Get predicted class
            _, predicted = torch.max(probabilities, 1)
            confidence = probabilities[0, predicted.item()].item()

            # Get class name
            class_name = self.class_names[predicted.item()]

        if return_probabilities:
            return class_name, confidence
        else:
            return class_name

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        return_probabilities: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Predict classes for a batch of images.

        Args:
            images: List of images to classify
            return_probabilities: Whether to return probabilities

        Returns:
            List of class names (and probabilities if return_probabilities=True)
        """
        # Preprocess all images
        image_tensors = []
        for image in images:
            if not isinstance(image, torch.Tensor):
                image_tensor = self.preprocess_image(image)
            else:
                image_tensor = image
            image_tensors.append(image_tensor)

        # Stack into batch
        batch_tensor = torch.cat(image_tensors, dim=0).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)

            # Get probabilities using softmax
            if outputs.shape[1] > 1:  # Multi-class classification
                probabilities = torch.softmax(outputs, dim=1)
            else:  # Binary classification
                probabilities = torch.sigmoid(outputs)
                probabilities = torch.cat([1 - probabilities, probabilities], dim=1)

            # Get predicted classes
            _, predicted = torch.max(probabilities, 1)
            confidences = (
                probabilities[torch.arange(len(predicted)), predicted].cpu().numpy()
            )

            # Get class names
            class_names = [self.class_names[p.item()] for p in predicted]

        if return_probabilities:
            return class_names, confidences.tolist()
        else:
            return class_names

    def predict_with_confidence(
        self, image: Union[str, Path, Image.Image]
    ) -> Dict[str, Union[str, float]]:
        """
        Predict with detailed confidence information.

        Args:
            image: Image to classify

        Returns:
            Dictionary with prediction details
        """
        # Preprocess
        if not isinstance(image, torch.Tensor):
            image_tensor = self.preprocess_image(image)
        else:
            image_tensor = image

        image_tensor = image_tensor.to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

            # Get probabilities
            if outputs.shape[1] > 1:  # Multi-class classification
                probabilities = torch.softmax(outputs, dim=1)[0]
            else:  # Binary classification
                probabilities = torch.sigmoid(outputs)[0]
                probabilities = torch.tensor(
                    [1 - probabilities.item(), probabilities.item()]
                )

            # Get all class probabilities
            class_probs = {}
            for i, class_name in enumerate(self.class_names):
                class_probs[class_name] = probabilities[i].item()

            # Get predicted class
            predicted_idx = torch.argmax(probabilities).item()
            predicted_class = self.class_names[predicted_idx]
            confidence = probabilities[predicted_idx].item()

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "all_classes": self.class_names,
        }

    def visualize_prediction(
        self,
        image_path: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> Image.Image:
        """
        Visualize prediction on an image.

        Args:
            image_path: Path to image
            save_path: Path to save visualization
            show: Whether to display the image

        Returns:
            Annotated PIL Image
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Get prediction
        prediction = self.predict_with_confidence(image_path)

        # Create annotated image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Try to load a font
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Add prediction text
        text = f"{prediction['predicted_class']}: {prediction['confidence']:.2%}"

        # Calculate text position (top left)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw background rectangle
        padding = 5
        draw.rectangle(
            [(10, 10), (10 + text_width + 2 * padding, 10 + text_height + 2 * padding)],
            fill="black",
        )

        # Draw text
        draw.text((10 + padding, 10 + padding), text, fill="white", font=font)

        # Add probability bars
        bar_height = 20
        bar_width = 100
        bar_spacing = 5

        for i, class_name in enumerate(self.class_names):
            prob = prediction["class_probabilities"][class_name]

            # Draw bar background
            y = 50 + i * (bar_height + bar_spacing)
            draw.rectangle(
                [(10, y), (10 + bar_width, y + bar_height)], fill="lightgray"
            )

            # Draw probability bar
            fill_width = int(bar_width * prob)
            draw.rectangle(
                [(10, y), (10 + fill_width, y + bar_height)],
                fill="green" if class_name == prediction["predicted_class"] else "blue",
            )

            # Draw text
            text = f"{class_name}: {prob:.2%}"
            draw.text((10 + bar_width + 10, y + 5), text, fill="black", font=font)

        # Save if requested
        if save_path:
            annotated_image.save(save_path)
            print(f"Saved visualization to {save_path}")

        # Show if requested
        if show:
            plt.figure(figsize=(10, 8))
            plt.imshow(annotated_image)
            plt.axis("off")
            plt.title(
                f"Prediction: {prediction['predicted_class']} ({prediction['confidence']:.2%})"
            )
            plt.show()

        return annotated_image

    def test_on_directory(
        self, directory: Union[str, Path], class_subdirs: bool = True
    ) -> Dict[str, float]:
        """
        Test model on all images in a directory.

        Args:
            directory: Directory containing images
            class_subdirs: Whether images are organized in class subdirectories

        Returns:
            Dictionary with accuracy and per-class metrics
        """
        directory = Path(directory)

        if not class_subdirs:
            # All images in one directory
            image_paths = (
                list(directory.glob("*.jpg"))
                + list(directory.glob("*.jpeg"))
                + list(directory.glob("*.png"))
            )

            # Without class labels, we can only get predictions
            predictions = []
            for img_path in image_paths:
                pred = self.predict(img_path)
                predictions.append((img_path.name, pred))

            return {"total_images": len(image_paths), "predictions": predictions}

        else:
            # Images organized by class
            results = {}
            total_correct = 0
            total_images = 0

            for class_idx, class_name in enumerate(self.class_names):
                class_dir = directory / class_name
                if not class_dir.exists():
                    continue

                image_paths = (
                    list(class_dir.glob("*.jpg"))
                    + list(class_dir.glob("*.jpeg"))
                    + list(class_dir.glob("*.png"))
                )

                if not image_paths:
                    continue

                correct = 0
                for img_path in image_paths:
                    pred = self.predict(img_path)
                    if pred == class_name:
                        correct += 1

                accuracy = correct / len(image_paths) if image_paths else 0
                results[class_name] = {
                    "total": len(image_paths),
                    "correct": correct,
                    "accuracy": accuracy,
                }

                total_correct += correct
                total_images += len(image_paths)

            if total_images > 0:
                results["overall"] = {
                    "total": total_images,
                    "correct": total_correct,
                    "accuracy": total_correct / total_images,
                }

            return results


class ModelLoader:
    """Utility class for loading trained models."""

    RUNS_DIR = Path("runs")

    @staticmethod
    def load_model(
        checkpoint_path: Union[str, Path],
        model_class: nn.Module,
        device: Optional[torch.device] = None,
    ) -> InferenceEngine:
        """
        Load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            model_class: Model class (e.g., CustomCNN)
            device: Device to load model on

        Returns:
            InferenceEngine with loaded model
        """
        device = device or auto_select_device()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model
        model = model_class(num_classes=2)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Create inference engine
        engine = InferenceEngine(model, device)

        print(f"Loaded model from {checkpoint_path}")
        print(
            f"Checkpoint info: Epoch {checkpoint.get('epoch', 'N/A')}, "
            f"Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%"
        )

        return engine

    @staticmethod
    def load_from_config(config_path: Union[str, Path]) -> InferenceEngine:
        """
        Load model from configuration file.

        Args:
            config_path: Path to configuration JSON file

        Returns:
            InferenceEngine with loaded model
        """
        config_path = Path(config_path)

        with open(config_path, "r") as f:
            config = json.load(f)

        # Import model class
        import importlib

        module = importlib.import_module(config["model_module"])
        model_class = getattr(module, config["model_class"])

        # Load model
        return ModelLoader.load_model(
            checkpoint_path=config["checkpoint_path"],
            model_class=model_class,
            device=torch.device(config.get("device", "cpu")),
        )

    @staticmethod
    def load_latest_model(
        model_class: nn.Module, device: Optional[torch.device] = None
    ) -> InferenceEngine:
        """
        Load the latest model from the most recent experiment.

        Args:
            model_class: Model class to instantiate
            device: Device to load model on

        Returns:
            InferenceEngine with loaded model
        """
        if not ModelLoader.RUNS_DIR.exists():
            raise FileNotFoundError(f"Runs directory not found: {ModelLoader.RUNS_DIR}")

        experiments = [
            d
            for d in ModelLoader.RUNS_DIR.iterdir()
            if d.is_dir() and d.name.startswith("exp")
        ]

        if not experiments:
            raise FileNotFoundError(f"No experiments found in {ModelLoader.RUNS_DIR}")

        latest_exp = max(experiments, key=lambda x: int(x.name.replace("exp", "")))
        checkpoint_path = latest_exp / "checkpoints" / "final_model.pth"

        if not checkpoint_path.exists():
            # Try to find best model
            checkpoints = list((latest_exp / "checkpoints").glob("best_model*.pth"))
            if not checkpoints:
                raise FileNotFoundError(
                    f"No checkpoints found in {latest_exp / 'checkpoints'}"
                )
            checkpoint_path = max(
                checkpoints,
                key=lambda x: int(
                    x.name.split("_acc")[0].replace("best_model_epoch", "")
                ),
            )

        print(f"Loading latest model from experiment: {latest_exp.name}")
        return ModelLoader.load_model(checkpoint_path, model_class, device)

    @staticmethod
    def load_from_experiment(
        exp_name: str,
        model_class: nn.Module,
        use_final: bool = True,
        device: Optional[torch.device] = None,
    ) -> InferenceEngine:
        """
        Load model from a specific experiment.

        Args:
            exp_name: Experiment name (e.g., 'exp1', 'my_experiment')
            model_class: Model class to instantiate
            use_final: If True, load final_model.pth, else load best model
            device: Device to load model on

        Returns:
            InferenceEngine with loaded model
        """
        exp_dir = ModelLoader.RUNS_DIR / exp_name

        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {exp_dir}")

        checkpoint_dir = exp_dir / "checkpoints"

        if use_final:
            checkpoint_path = checkpoint_dir / "final_model.pth"
        else:
            checkpoints = list(checkpoint_dir.glob("best_model*.pth"))
            if not checkpoints:
                raise FileNotFoundError(
                    f"No best model checkpoints found in {checkpoint_dir}"
                )
            checkpoint_path = max(
                checkpoints,
                key=lambda x: float(x.name.split("_acc")[1].replace(".pth", "")),
            )

        print(f"Loading model from experiment: {exp_name}")
        return ModelLoader.load_model(checkpoint_path, model_class, device)


def test_inference():
    """Test the inference module."""
    print("Testing inference module...")

    # Create a simple model
    from model import CustomCNN

    model = CustomCNN(num_classes=2)

    # Create inference engine
    engine = InferenceEngine(model)

    # Create a dummy image
    dummy_image = Image.new("RGB", (224, 224), color="red")

    # Test single prediction
    print("\nTesting single image prediction:")
    prediction = engine.predict(dummy_image, return_probabilities=True)
    print(f"Prediction: {prediction}")

    # Test batch prediction
    print("\nTesting batch prediction:")
    batch_predictions = engine.predict_batch(
        [dummy_image, dummy_image], return_probabilities=True
    )
    print(f"Batch predictions: {batch_predictions}")

    # Test detailed prediction
    print("\nTesting detailed prediction:")
    detailed = engine.predict_with_confidence(dummy_image)
    print(f"Detailed prediction: {detailed}")

    print("\nInference tests completed successfully!")


if __name__ == "__main__":
    test_inference()
