#!/usr/bin/env python3
"""
Fine-tuning script for BERT-based sentiment analysis models.

This script provides comprehensive fine-tuning capabilities using the HuggingFace
Trainer API with support for distributed training, custom datasets, and advanced
training techniques.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentTrainer:
    """Custom trainer for sentiment analysis fine-tuning."""
    
    def __init__(self, model_name: str, num_labels: int, output_dir: str,
                 learning_rate: float = 2e-5, warmup_ratio: float = 0.1,
                 weight_decay: float = 0.01, use_wandb: bool = False):
        """
        Initialize sentiment trainer.
        
        Args:
            model_name: Pre-trained model name
            num_labels: Number of sentiment classes
            output_dir: Directory to save model and logs
            learning_rate: Learning rate for optimization
            warmup_ratio: Warmup ratio for learning rate scheduler
            weight_decay: Weight decay for regularization
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.use_wandb = use_wandb
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        )
        
        # Data collator
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=True
        )
        
        logger.info(f"Initialized trainer for {model_name} with {num_labels} labels")
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions from trainer
            
        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        precision, recall, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
        }
        
        # Add per-class metrics
        for i in range(len(precision)):
            metrics[f'precision_class_{i}'] = precision[i]
            metrics[f'recall_class_{i}'] = recall[i]
            metrics[f'f1_class_{i}'] = f1_per_class[i]
        
        return metrics
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Dataset,
                      training_args: TrainingArguments) -> Trainer:
        """
        Create HuggingFace Trainer instance.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments
            
        Returns:
            Configured Trainer instance
        """
        # Callbacks
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        if self.use_wandb:
            callbacks.append(WandbCallback())
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        return trainer
    
    def fine_tune(self, train_dataset: Dataset, eval_dataset: Dataset,
                 num_train_epochs: int = 3, per_device_train_batch_size: int = 16,
                 per_device_eval_batch_size: int = 32, gradient_accumulation_steps: int = 1,
                 eval_strategy: str = "steps", eval_steps: int = 500,
                 save_strategy: str = "steps", save_steps: int = 500,
                 logging_steps: int = 100, load_best_model_at_end: bool = True,
                 metric_for_best_model: str = "eval_f1_weighted",
                 greater_is_better: bool = True, dataloader_num_workers: int = 4,
                 fp16: bool = False, gradient_checkpointing: bool = False,
                 **kwargs) -> Trainer:
        """
        Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Training batch size per device
            per_device_eval_batch_size: Evaluation batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            eval_strategy: Evaluation strategy
            eval_steps: Evaluation frequency
            save_strategy: Save strategy
            save_steps: Save frequency
            logging_steps: Logging frequency
            load_best_model_at_end: Whether to load best model at end
            metric_for_best_model: Metric to use for model selection
            greater_is_better: Whether higher metric is better
            dataloader_num_workers: Number of dataloader workers
            fp16: Whether to use mixed precision
            gradient_checkpointing: Whether to use gradient checkpointing
            **kwargs: Additional training arguments
            
        Returns:
            Trained Trainer instance
        """
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=logging_steps,
            evaluation_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            dataloader_num_workers=dataloader_num_workers,
            fp16=fp16,
            gradient_checkpointing=gradient_checkpointing,
            report_to="wandb" if self.use_wandb else None,
            run_name=f"sentiment-{self.model_name.split('/')[-1]}" if self.use_wandb else None,
            **kwargs
        )
        
        # Create trainer
        trainer = self.create_trainer(train_dataset, eval_dataset, training_args)
        
        # Log training info
        logger.info(f"Starting fine-tuning with {len(train_dataset)} training samples")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        logger.info(f"Training for {num_train_epochs} epochs")
        logger.info(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
        
        # Train model
        trainer.train()
        
        # Save final model
        trainer.save_model()
        trainer.save_state()
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")
        
        return trainer


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CustomTrainer(Trainer):
    """Custom trainer with focal loss support."""
    
    def __init__(self, *args, use_focal_loss: bool = False, 
                 focal_alpha: float = 1.0, focal_gamma: float = 2.0, **kwargs):
        """Initialize custom trainer."""
        super().__init__(*args, **kwargs)
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if self.use_focal_loss and labels is not None:
            loss = self.focal_loss(outputs.logits, labels)
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss


def load_dataset_splits(data_dir: str) -> Dict[str, Dataset]:
    """Load train/validation/test datasets."""
    datasets = {}
    
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(data_dir, f"{split}_dataset")
        if os.path.exists(split_path):
            datasets[split] = load_from_disk(split_path)
            logger.info(f"Loaded {split} dataset: {len(datasets[split])} samples")
        else:
            logger.warning(f"Dataset split {split} not found at {split_path}")
    
    return datasets


def save_training_config(args: argparse.Namespace, output_dir: str):
    """Save training configuration."""
    config = vars(args)
    config_path = os.path.join(output_dir, "training_config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Training config saved to {config_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sentiment analysis")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="Pre-trained model name")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed datasets")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for model and logs")
    parser.add_argument("--num_labels", type=int, default=3,
                       help="Number of sentiment classes")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32,
                       help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for regularization")
    
    # Training strategy
    parser.add_argument("--eval_strategy", type=str, default="steps",
                       choices=["no", "steps", "epoch"],
                       help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation frequency (steps)")
    parser.add_argument("--save_strategy", type=str, default="steps",
                       choices=["no", "steps", "epoch"],
                       help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save frequency (steps)")
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Logging frequency")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_f1_weighted",
                       help="Metric for model selection")
    
    # Advanced training options
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Use gradient checkpointing to save memory")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--use_focal_loss", action="store_true",
                       help="Use focal loss for class imbalance")
    parser.add_argument("--focal_alpha", type=float, default=1.0,
                       help="Focal loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal loss gamma parameter")
    
    # Distributed training
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    
    # Monitoring and logging
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="sentiment-analysis",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str,
                       help="Weights & Biases run name")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize Weights & Biases
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"sentiment-{args.model_name.split('/')[-1]}",
            config=vars(args)
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save training configuration
    save_training_config(args, args.output_dir)
    
    # Load datasets
    logger.info(f"Loading datasets from {args.data_dir}")
    datasets = load_dataset_splits(args.data_dir)
    
    if "train" not in datasets:
        raise ValueError("Training dataset not found")
    if "validation" not in datasets:
        raise ValueError("Validation dataset not found")
    
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    
    # Get number of labels from dataset if not provided
    if args.num_labels == 3:  # Default value
        unique_labels = set(train_dataset["labels"])
        args.num_labels = len(unique_labels)
        logger.info(f"Detected {args.num_labels} unique labels: {sorted(unique_labels)}")
    
    # Initialize trainer
    sentiment_trainer = SentimentTrainer(
        model_name=args.model_name,
        num_labels=args.num_labels,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        use_wandb=args.use_wandb
    )
    
    # Fine-tune model
    trainer = sentiment_trainer.fine_tune(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        metric_for_best_model=args.metric_for_best_model,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing
    )
    
    # Evaluate on test set if available
    if "test" in datasets:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(datasets["test"])
        
        # Save test results
        test_results_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Test results saved to {test_results_path}")
        logger.info(f"Test accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
        logger.info(f"Test F1 (weighted): {test_results.get('eval_f1_weighted', 'N/A'):.4f}")
    
    # Close wandb
    if args.use_wandb:
        wandb.finish()
    
    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()