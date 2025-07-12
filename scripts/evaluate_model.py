#!/usr/bin/env python3
"""
Model evaluation script for sentiment analysis models.

This script provides comprehensive evaluation capabilities including various metrics,
confusion matrices, classification reports, and visualization tools for model assessment.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize model evaluator.
        
        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer (defaults to model_path)
            device: Device to use for inference
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.num_labels = len(self.id2label)
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model has {self.num_labels} labels: {list(self.id2label.values())}")
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def evaluate_dataset(self, dataset: Dataset, output_dir: str) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Dataset to evaluate
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Extract texts and labels
        texts = dataset["text"] if "text" in dataset.column_names else [
            self.tokenizer.decode(ids, skip_special_tokens=True) 
            for ids in dataset["input_ids"]
        ]
        true_labels = np.array(dataset["labels"])
        
        # Make predictions
        logger.info(f"Making predictions on {len(texts)} samples...")
        predictions, probabilities = self.predict_batch(texts)
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_labels, predictions, probabilities)
        
        # Generate detailed reports
        self.generate_classification_report(true_labels, predictions, output_dir)
        self.plot_confusion_matrix(true_labels, predictions, output_dir)
        self.plot_roc_curves(true_labels, probabilities, output_dir)
        self.plot_precision_recall_curves(true_labels, probabilities, output_dir)
        self.analyze_predictions(texts, true_labels, predictions, probabilities, output_dir)
        
        return metrics
    
    def calculate_metrics(self, true_labels: np.ndarray, predictions: np.ndarray,
                         probabilities: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            true_labels: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        f1_micro = f1_score(true_labels, predictions, average='micro')
        
        # Per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_micro': float(f1_micro),
        }
        
        # Add per-class metrics
        for i in range(self.num_labels):
            label_name = self.id2label.get(i, f"class_{i}")
            metrics[f'precision_{label_name}'] = float(precision[i])
            metrics[f'recall_{label_name}'] = float(recall[i])
            metrics[f'f1_{label_name}'] = float(f1_per_class[i])
            metrics[f'support_{label_name}'] = int(support[i])
        
        # ROC AUC (for binary and multiclass)
        try:
            if self.num_labels == 2:
                roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
                metrics['roc_auc'] = float(roc_auc)
            else:
                roc_auc_ovr = roc_auc_score(true_labels, probabilities, 
                                          multi_class='ovr', average='macro')
                roc_auc_ovo = roc_auc_score(true_labels, probabilities, 
                                          multi_class='ovo', average='macro')
                metrics['roc_auc_ovr'] = float(roc_auc_ovr)
                metrics['roc_auc_ovo'] = float(roc_auc_ovo)
        except ValueError as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
        
        return metrics
    
    def generate_classification_report(self, true_labels: np.ndarray, 
                                     predictions: np.ndarray, output_dir: str):
        """Generate and save classification report."""
        # Create classification report
        target_names = [self.id2label.get(i, f"class_{i}") for i in range(self.num_labels)]
        report = classification_report(
            true_labels, predictions, target_names=target_names, output_dict=True
        )
        
        # Save as JSON
        report_path = os.path.join(output_dir, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as text
        report_text = classification_report(
            true_labels, predictions, target_names=target_names
        )
        report_text_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_text_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Classification report saved to {report_path}")
    
    def plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray,
                            output_dir: str, normalize: bool = True):
        """Plot and save confusion matrix."""
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_to_plot = cm_normalized
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            cm_to_plot = cm
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Create plot
        plt.figure(figsize=(10, 8))
        labels = [self.id2label.get(i, f"class_{i}") for i in range(self.num_labels)]
        
        sns.heatmap(
            cm_to_plot,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Percentage' if normalize else 'Count'}
        )
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save plot
        cm_path = os.path.join(output_dir, f"confusion_matrix{'_normalized' if normalize else ''}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save raw confusion matrix
        cm_data_path = os.path.join(output_dir, "confusion_matrix_data.json")
        cm_data = {
            'matrix': cm.tolist(),
            'labels': labels,
            'true_labels': true_labels.tolist(),
            'predictions': predictions.tolist()
        }
        with open(cm_data_path, 'w') as f:
            json.dump(cm_data, f, indent=2)
        
        logger.info(f"Confusion matrix saved to {cm_path}")
    
    def plot_roc_curves(self, true_labels: np.ndarray, probabilities: np.ndarray,
                       output_dir: str):
        """Plot ROC curves for each class."""
        plt.figure(figsize=(12, 8))
        
        if self.num_labels == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
            roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'ROC Curve (AUC = {roc_auc:.3f})')
        else:
            # Multiclass classification (one-vs-rest)
            from sklearn.preprocessing import label_binarize
            
            # Binarize labels
            y_bin = label_binarize(true_labels, classes=range(self.num_labels))
            
            for i in range(self.num_labels):
                fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
                roc_auc = roc_auc_score(y_bin[:, i], probabilities[:, i])
                label_name = self.id2label.get(i, f"class_{i}")
                
                plt.plot(fpr, tpr, linewidth=2,
                        label=f'{label_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        roc_path = os.path.join(output_dir, "roc_curves.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {roc_path}")
    
    def plot_precision_recall_curves(self, true_labels: np.ndarray, 
                                   probabilities: np.ndarray, output_dir: str):
        """Plot precision-recall curves for each class."""
        plt.figure(figsize=(12, 8))
        
        if self.num_labels == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(true_labels, probabilities[:, 1])
            
            plt.plot(recall, precision, linewidth=2, label='Precision-Recall Curve')
        else:
            # Multiclass classification (one-vs-rest)
            from sklearn.preprocessing import label_binarize
            
            # Binarize labels
            y_bin = label_binarize(true_labels, classes=range(self.num_labels))
            
            for i in range(self.num_labels):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], probabilities[:, i])
                label_name = self.id2label.get(i, f"class_{i}")
                
                plt.plot(recall, precision, linewidth=2, label=f'{label_name}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        pr_path = os.path.join(output_dir, "precision_recall_curves.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-recall curves saved to {pr_path}")
    
    def analyze_predictions(self, texts: List[str], true_labels: np.ndarray,
                          predictions: np.ndarray, probabilities: np.ndarray,
                          output_dir: str, num_examples: int = 50):
        """Analyze individual predictions and save examples."""
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'text': texts,
            'true_label': [self.id2label.get(label, f"class_{label}") for label in true_labels],
            'predicted_label': [self.id2label.get(pred, f"class_{pred}") for pred in predictions],
            'confidence': np.max(probabilities, axis=1),
            'correct': true_labels == predictions
        })
        
        # Add probability scores for each class
        for i in range(self.num_labels):
            label_name = self.id2label.get(i, f"class_{i}")
            df[f'prob_{label_name}'] = probabilities[:, i]
        
        # Save full predictions
        predictions_path = os.path.join(output_dir, "detailed_predictions.csv")
        df.to_csv(predictions_path, index=False)
        
        # Analyze correct vs incorrect predictions
        correct_df = df[df['correct'] == True]
        incorrect_df = df[df['correct'] == False]
        
        # High-confidence correct predictions
        high_conf_correct = correct_df.nlargest(num_examples, 'confidence')
        high_conf_correct_path = os.path.join(output_dir, "high_confidence_correct.csv")
        high_conf_correct.to_csv(high_conf_correct_path, index=False)
        
        # High-confidence incorrect predictions (potential model issues)
        high_conf_incorrect = incorrect_df.nlargest(num_examples, 'confidence')
        high_conf_incorrect_path = os.path.join(output_dir, "high_confidence_incorrect.csv")
        high_conf_incorrect.to_csv(high_conf_incorrect_path, index=False)
        
        # Low-confidence predictions (uncertain cases)
        low_conf_all = df.nsmallest(num_examples, 'confidence')
        low_conf_path = os.path.join(output_dir, "low_confidence_predictions.csv")
        low_conf_all.to_csv(low_conf_path, index=False)
        
        # Summary statistics
        summary = {
            'total_samples': len(df),
            'correct_predictions': len(correct_df),
            'incorrect_predictions': len(incorrect_df),
            'accuracy': len(correct_df) / len(df),
            'average_confidence': float(df['confidence'].mean()),
            'average_confidence_correct': float(correct_df['confidence'].mean()),
            'average_confidence_incorrect': float(incorrect_df['confidence'].mean()),
            'confidence_std': float(df['confidence'].std())
        }
        
        summary_path = os.path.join(output_dir, "prediction_analysis_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Prediction analysis saved to {output_dir}")
        logger.info(f"Accuracy: {summary['accuracy']:.4f}")
        logger.info(f"Average confidence: {summary['average_confidence']:.4f}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis model")
    
    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--tokenizer_path", type=str,
                       help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--data_path", type=str,
                       help="Path to evaluation dataset or data directory")
    parser.add_argument("--dataset_split", type=str, default="test",
                       choices=["train", "validation", "test"],
                       help="Dataset split to evaluate")
    
    # Evaluation options
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for evaluation results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str,
                       help="Device to use (cuda/cpu)")
    
    # Analysis options
    parser.add_argument("--num_examples", type=int, default=50,
                       help="Number of examples for detailed analysis")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save detailed predictions")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device
    )
    
    # Load evaluation dataset
    if args.data_path:
        if os.path.isfile(args.data_path):
            # Single dataset file
            dataset = load_from_disk(args.data_path)
        else:
            # Directory with multiple splits
            dataset_path = os.path.join(args.data_path, f"{args.dataset_split}_dataset")
            if not os.path.exists(dataset_path):
                raise ValueError(f"Dataset split {args.dataset_split} not found at {dataset_path}")
            dataset = load_from_disk(dataset_path)
        
        logger.info(f"Loaded {args.dataset_split} dataset: {len(dataset)} samples")
        
        # Evaluate model
        metrics = evaluator.evaluate_dataset(dataset, args.output_dir)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print key metrics
        logger.info("Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        if 'roc_auc' in metrics:
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        elif 'roc_auc_ovr' in metrics:
            logger.info(f"ROC AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
    
    else:
        logger.info("No dataset provided. Model loaded and ready for evaluation.")
        logger.info("Use the ModelEvaluator class methods for custom evaluation.")
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()