"""
Evaluation module for cat/dog image classification.
Evaluates model performance including accuracy and computational efficiency.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils import auto_select_device
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluates model performance on various metrics."""

    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None):
        """
        Args:
            model: PyTorch model to evaluate
            device: Device to use for evaluation
        """
        self.model = model
        self.device = device or auto_select_device()
        self.model.to(self.device)
        self.model.eval()

        # Results storage
        self.results = {}

        print(f"Evaluator initialized on {self.device}")

    def evaluate_accuracy(self,
                         dataloader: DataLoader,
                         verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate model accuracy on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            verbose: Whether to print detailed results

        Returns:
            Dictionary with accuracy metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating", leave=False) if verbose else dataloader
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Get probabilities
                if outputs.shape[1] > 1:  # Multi-class
                    probabilities = torch.softmax(outputs, dim=1)
                else:  # Binary
                    probabilities = torch.sigmoid(outputs)
                    probabilities = torch.cat([1 - probabilities, probabilities], dim=1)

                # Get predictions
                _, predicted = torch.max(outputs, 1)

                # Update statistics
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Calculate metrics
        accuracy = 100. * correct / total if total > 0 else 0

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)

        # Calculate additional metrics
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'error_rate': 100. - accuracy
        }

        # Calculate per-class accuracy if we have class information
        if len(np.unique(all_targets)) > 1:
            # Confusion matrix
            cm = confusion_matrix(all_targets, all_predictions)
            metrics['confusion_matrix'] = cm.tolist()

            # Per-class accuracy
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            metrics['per_class_accuracy'] = per_class_acc.tolist()

            # Classification report
            report = classification_report(all_targets, all_predictions,
                                          output_dict=True)
            metrics['classification_report'] = report

            # ROC AUC for binary classification
            if len(np.unique(all_targets)) == 2:
                fpr, tpr, _ = roc_curve(all_targets, all_probabilities[:, 1])
                roc_auc = auc(fpr, tpr)
                metrics['roc_auc'] = roc_auc
                metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

        if verbose:
            print("\nAccuracy Evaluation:")
            print(f"  Correct: {correct}/{total}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Error Rate: {100. - accuracy:.2f}%")

            if 'per_class_accuracy' in metrics:
                print("\nPer-class Accuracy:")
                for i, acc in enumerate(metrics['per_class_accuracy']):
                    print(f"  Class {i}: {acc:.2%}")

            if 'roc_auc' in metrics:
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")

        self.results['accuracy'] = metrics
        return metrics

    def evaluate_efficiency(self,
                          input_shape: Tuple[int, int, int] = (3, 224, 224),
                          num_iterations: int = 100,
                          batch_size: int = 1,
                          verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate computational efficiency of the model.

        Args:
            input_shape: Shape of input tensor (channels, height, width)
            num_iterations: Number of iterations for timing
            batch_size: Batch size for inference
            verbose: Whether to print detailed results

        Returns:
            Dictionary with efficiency metrics
        """
        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)

        # Measure inference time
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)

        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_iteration = total_time / num_iterations
        fps = 1.0 / avg_time_per_iteration if avg_time_per_iteration > 0 else 0

        # Calculate memory usage (approximate)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = self.model(dummy_input)
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        else:
            memory_used = 0

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        metrics = {
            'total_time_seconds': total_time,
            'avg_inference_time_ms': avg_time_per_iteration * 1000,
            'fps': fps,
            'memory_used_mb': memory_used,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameters_millions': total_params / 1e6,
            'input_shape': input_shape,
            'batch_size': batch_size,
            'iterations': num_iterations
        }

        if verbose:
            print("\nEfficiency Evaluation:")
            print(f"  Average Inference Time: {avg_time_per_iteration * 1000:.2f} ms")
            print(f"  FPS: {fps:.2f}")
            print(f"  Total Parameters: {total_params:,}")
            print(f"  Trainable Parameters: {trainable_params:,}")
            print(f"  Parameters (Millions): {total_params / 1e6:.2f}M")

            if self.device.type == 'cuda':
                print(f"  GPU Memory Used: {memory_used:.2f} MB")

        self.results['efficiency'] = metrics
        return metrics

    def evaluate_robustness(self,
                          dataloader: DataLoader,
                          noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3],
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate model robustness to noise.

        Args:
            dataloader: DataLoader for evaluation
            noise_levels: List of noise levels to test
            verbose: Whether to print detailed results

        Returns:
            Dictionary with robustness metrics
        """
        self.model.eval()

        results = {}
        base_accuracy = None

        for noise_level in noise_levels:
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # Add Gaussian noise
                    if noise_level > 0:
                        noise = torch.randn_like(inputs) * noise_level
                        noisy_inputs = torch.clamp(inputs + noise, 0, 1)
                    else:
                        noisy_inputs = inputs

                    # Forward pass
                    outputs = self.model(noisy_inputs)
                    _, predicted = torch.max(outputs, 1)

                    # Update statistics
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total if total > 0 else 0

            results[f'noise_{noise_level}'] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }

            if noise_level == 0.0:
                base_accuracy = accuracy

            if verbose:
                print(f"  Noise Level {noise_level}: Accuracy = {accuracy:.2f}%")

        # Calculate robustness score (average accuracy drop)
        if base_accuracy is not None and len(noise_levels) > 1:
            accuracy_drops = []
            for noise_level in noise_levels[1:]:  # Skip noise level 0
                accuracy = results[f'noise_{noise_level}']['accuracy']
                accuracy_drops.append(base_accuracy - accuracy)

            avg_accuracy_drop = np.mean(accuracy_drops) if accuracy_drops else 0
            robustness_score = 100 - avg_accuracy_drop

            results['robustness_metrics'] = {
                'base_accuracy': base_accuracy,
                'avg_accuracy_drop': avg_accuracy_drop,
                'robustness_score': robustness_score
            }

            if verbose:
                print("\nRobustness Summary:")
                print(f"  Base Accuracy: {base_accuracy:.2f}%")
                print(f"  Average Accuracy Drop: {avg_accuracy_drop:.2f}%")
                print(f"  Robustness Score: {robustness_score:.2f}")

        self.results['robustness'] = results
        return results

    def generate_report(self,
                       output_dir: str = "evaluation_results",
                       plot: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            output_dir: Directory to save report
            plot: Whether to generate plots

        Returns:
            Complete evaluation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Generate timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save results as JSON
        report_path = output_dir / f"evaluation_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nEvaluation report saved to: {report_path}")

        # Generate plots if requested
        if plot and self.results:
            self._generate_plots(output_dir, timestamp)

        return self.results

    def _generate_plots(self, output_dir: Path, timestamp: str):
        """Generate visualization plots."""

        # Accuracy metrics plot
        if 'accuracy' in self.results:
            acc_results = self.results['accuracy']

            # Confusion matrix heatmap
            if 'confusion_matrix' in acc_results:
                cm = np.array(acc_results['confusion_matrix'])
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(output_dir / f"confusion_matrix_{timestamp}.png")
                plt.close()

            # ROC curve for binary classification
            if 'roc_curve' in acc_results:
                roc_data = acc_results['roc_curve']
                roc_auc = acc_results.get('roc_auc', 0)

                plt.figure(figsize=(8, 6))
                plt.plot(roc_data['fpr'], roc_data['tpr'],
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / f"roc_curve_{timestamp}.png")
                plt.close()

        # Efficiency comparison plot
        if 'efficiency' in self.results:
            eff_results = self.results['efficiency']

            # Create a simple bar chart for key metrics
            metrics_to_plot = {
                'Inference Time (ms)': eff_results['avg_inference_time_ms'],
                'FPS': eff_results['fps'],
                'Parameters (M)': eff_results['parameters_millions']
            }

            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(metrics_to_plot)), list(metrics_to_plot.values()))
            plt.xticks(range(len(metrics_to_plot)), list(metrics_to_plot.keys()))
            plt.title('Model Efficiency Metrics')
            plt.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, value in zip(bars, metrics_to_plot.values()):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(output_dir / f"efficiency_metrics_{timestamp}.png")
            plt.close()

        print(f"Plots saved to {output_dir}")

    def compare_models(self,
                      models: Dict[str, nn.Module],
                      dataloader: DataLoader,
                      input_shape: Tuple[int, int, int] = (3, 224, 224)) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on accuracy and efficiency.

        Args:
            models: Dictionary of model names to models
            dataloader: DataLoader for accuracy evaluation
            input_shape: Input shape for efficiency evaluation

        Returns:
            Dictionary with comparison results
        """
        comparison_results = {}

        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")

            # Create evaluator for this model
            evaluator = ModelEvaluator(model, self.device)

            # Evaluate accuracy
            accuracy_results = evaluator.evaluate_accuracy(dataloader, verbose=False)

            # Evaluate efficiency
            efficiency_results = evaluator.evaluate_efficiency(
                input_shape=input_shape, verbose=False
            )

            # Store results
            comparison_results[model_name] = {
                'accuracy': accuracy_results['accuracy'],
                'inference_time_ms': efficiency_results['avg_inference_time_ms'],
                'fps': efficiency_results['fps'],
                'parameters_millions': efficiency_results['parameters_millions'],
                'memory_used_mb': efficiency_results['memory_used_mb']
            }

            print(f"  Accuracy: {accuracy_results['accuracy']:.2f}%")
            print(f"  Inference Time: {efficiency_results['avg_inference_time_ms']:.2f} ms")
            print(f"  Parameters: {efficiency_results['parameters_millions']:.2f}M")

        # Sort by accuracy
        sorted_by_accuracy = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Accuracy':<12} {'Time (ms)':<12} {'Params (M)':<12}")
        print(f"{'-'*60}")

        for model_name, metrics in sorted_by_accuracy:
            print(f"{model_name:<20} {metrics['accuracy']:<11.2f}% "
                  f"{metrics['inference_time_ms']:<11.2f} {metrics['parameters_millions']:<11.2f}")

        return comparison_results


def test_evaluator():
    """Test the evaluator module."""
    print("Testing evaluator module...")

    # Create a simple model
    from model import CustomCNN
    model = CustomCNN(num_classes=2)

    # Create evaluator
    evaluator = ModelEvaluator(model)

    # Create dummy dataloader
    from torch.utils.data import TensorDataset, DataLoader
    dummy_data = torch.randn(50, 3, 224, 224)
    dummy_labels = torch.randint(0, 2, (50,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=False)

    # Test accuracy evaluation
    print("\nTesting accuracy evaluation...")
    evaluator.evaluate_accuracy(dummy_loader, verbose=True)

    # Test efficiency evaluation
    print("\nTesting efficiency evaluation...")
    evaluator.evaluate_efficiency(verbose=True)

    # Test robustness evaluation
    print("\nTesting robustness evaluation...")
    evaluator.evaluate_robustness(dummy_loader, verbose=True)

    # Generate report
    print("\nGenerating evaluation report...")
    report = evaluator.generate_report(plot=False)

    print("\nEvaluation tests completed successfully!")
    return report


if __name__ == "__main__":
    test_evaluator()
