#!/usr/bin/env python3
"""
Data preparation script for fine-tuning sentiment analysis models.

This script handles dataset loading, preprocessing, and augmentation for training
BERT-based sentiment analysis models. It supports various data formats and provides
comprehensive data augmentation techniques.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAugmenter:
    """Handles data augmentation for text datasets."""
    
    def __init__(self, augmentation_config: Dict[str, any]):
        """
        Initialize data augmenter.
        
        Args:
            augmentation_config: Configuration for augmentation techniques
        """
        self.config = augmentation_config
        self.augmenters = []
        
        if augmentation_config.get('synonym_replacement', False):
            self.augmenters.append(
                naw.SynonymAug(
                    aug_src='wordnet',
                    aug_p=augmentation_config.get('synonym_prob', 0.1),
                    aug_max=augmentation_config.get('synonym_max', 3)
                )
            )
        
        if augmentation_config.get('random_insertion', False):
            self.augmenters.append(
                naw.RandomWordAug(
                    action="insert",
                    aug_p=augmentation_config.get('insertion_prob', 0.1),
                    aug_max=augmentation_config.get('insertion_max', 2)
                )
            )
        
        if augmentation_config.get('random_swap', False):
            self.augmenters.append(
                naw.RandomWordAug(
                    action="swap",
                    aug_p=augmentation_config.get('swap_prob', 0.1),
                    aug_max=augmentation_config.get('swap_max', 2)
                )
            )
        
        if augmentation_config.get('random_deletion', False):
            self.augmenters.append(
                naw.RandomWordAug(
                    action="delete",
                    aug_p=augmentation_config.get('deletion_prob', 0.1),
                    aug_max=augmentation_config.get('deletion_max', 2)
                )
            )
    
    def augment_text(self, text: str, num_augmented: int = 1) -> List[str]:
        """
        Apply augmentation to a single text.
        
        Args:
            text: Input text to augment
            num_augmented: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        if not self.augmenters:
            return [text] * num_augmented
        
        augmented_texts = []
        for _ in range(num_augmented):
            # Randomly select an augmenter
            augmenter = random.choice(self.augmenters)
            try:
                augmented = augmenter.augment(text)
                if isinstance(augmented, list):
                    augmented_texts.extend(augmented)
                else:
                    augmented_texts.append(augmented)
            except Exception as e:
                logger.warning(f"Augmentation failed for text: {text[:50]}... Error: {e}")
                augmented_texts.append(text)
        
        return augmented_texts[:num_augmented]
    
    def augment_dataset(self, texts: List[str], labels: List[int], 
                       augmentation_factor: float = 0.5) -> Tuple[List[str], List[int]]:
        """
        Augment entire dataset.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            augmentation_factor: Fraction of data to augment
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        num_to_augment = int(len(texts) * augmentation_factor)
        indices_to_augment = random.sample(range(len(texts)), num_to_augment)
        
        logger.info(f"Augmenting {num_to_augment} samples out of {len(texts)}")
        
        for idx in tqdm(indices_to_augment, desc="Augmenting data"):
            original_text = texts[idx]
            original_label = labels[idx]
            
            # Generate one augmented version
            aug_texts = self.augment_text(original_text, num_augmented=1)
            
            for aug_text in aug_texts:
                if aug_text != original_text:  # Only add if actually different
                    augmented_texts.append(aug_text)
                    augmented_labels.append(original_label)
        
        return augmented_texts, augmented_labels


class DatasetProcessor:
    """Handles dataset loading and preprocessing."""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased", 
                 max_length: int = 512):
        """
        Initialize dataset processor.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def load_csv_dataset(self, file_path: str, text_column: str, 
                        label_column: str) -> Tuple[List[str], List[int]]:
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Tuple of (texts, labels)
        """
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV")
        
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].tolist()
        
        # Convert string labels to integers if needed
        if isinstance(labels[0], str):
            unique_labels = sorted(list(set(labels)))
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            labels = [label_to_id[label] for label in labels]
            logger.info(f"Label mapping: {label_to_id}")
        
        return texts, labels
    
    def load_json_dataset(self, file_path: str, text_key: str = "text", 
                         label_key: str = "label") -> Tuple[List[str], List[int]]:
        """
        Load dataset from JSON file.
        
        Args:
            file_path: Path to JSON file
            text_key: Key for text field
            label_key: Key for label field
            
        Returns:
            Tuple of (texts, labels)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # If data is a dict, assume it contains lists
            texts = data[text_key]
            labels = data[label_key]
        elif isinstance(data, list):
            # If data is a list of dicts
            texts = [item[text_key] for item in data]
            labels = [item[label_key] for item in data]
        else:
            raise ValueError("JSON format not supported")
        
        return texts, labels
    
    def load_huggingface_dataset(self, dataset_name: str, split: str = "train",
                                text_column: str = "text", 
                                label_column: str = "label") -> Tuple[List[str], List[int]]:
        """
        Load dataset from HuggingFace Hub.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace Hub
            split: Dataset split to load
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Tuple of (texts, labels)
        """
        dataset = load_dataset(dataset_name, split=split)
        
        texts = dataset[text_column]
        labels = dataset[label_column]
        
        return texts, labels
    
    def tokenize_data(self, texts: List[str], labels: List[int]) -> Dataset:
        """
        Tokenize texts and create HuggingFace Dataset.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            
        Returns:
            HuggingFace Dataset object
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset = Dataset.from_dict({
            "text": texts,
            "labels": labels
        })
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    def split_dataset(self, texts: List[str], labels: List[int], 
                     train_size: float = 0.8, val_size: float = 0.1,
                     test_size: float = 0.1, random_state: int = 42) -> Dict[str, Dataset]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            train_size: Fraction for training set
            val_size: Fraction for validation set
            test_size: Fraction for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train/val/test datasets
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-7:
            raise ValueError("Split sizes must sum to 1.0")
        
        # First split: train + val vs test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Second split: train vs val
        val_ratio = val_size / (train_size + val_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=val_ratio, 
            random_state=random_state, stratify=train_val_labels
        )
        
        # Create datasets
        datasets = {
            "train": self.tokenize_data(train_texts, train_labels),
            "validation": self.tokenize_data(val_texts, val_labels),
            "test": self.tokenize_data(test_texts, test_labels)
        }
        
        logger.info(f"Dataset splits - Train: {len(train_texts)}, "
                   f"Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        return datasets


def save_dataset_info(datasets: Dict[str, Dataset], output_dir: str):
    """Save dataset information and statistics."""
    info = {
        "num_labels": len(set(datasets["train"]["labels"])),
        "splits": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        "label_distribution": {}
    }
    
    # Calculate label distribution for each split
    for split_name, dataset in datasets.items():
        labels = dataset["labels"]
        unique, counts = np.unique(labels, return_counts=True)
        info["label_distribution"][split_name] = {
            str(label): int(count) for label, count in zip(unique, counts)
        }
    
    # Save info
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Dataset info saved to {info_path}")


def main():
    """Main function for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare training data for sentiment analysis")
    
    # Data source arguments
    parser.add_argument("--input_file", type=str, help="Path to input data file")
    parser.add_argument("--dataset_name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--data_format", choices=["csv", "json", "huggingface"], 
                       default="csv", help="Input data format")
    
    # Column/field names
    parser.add_argument("--text_column", type=str, default="text", 
                       help="Name of text column/field")
    parser.add_argument("--label_column", type=str, default="label", 
                       help="Name of label column/field")
    
    # Data processing
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                       help="Tokenizer to use")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Data splitting
    parser.add_argument("--train_size", type=float, default=0.8,
                       help="Training set size (fraction)")
    parser.add_argument("--val_size", type=float, default=0.1,
                       help="Validation set size (fraction)")
    parser.add_argument("--test_size", type=float, default=0.1,
                       help="Test set size (fraction)")
    
    # Data augmentation
    parser.add_argument("--augment", action="store_true",
                       help="Enable data augmentation")
    parser.add_argument("--augmentation_factor", type=float, default=0.3,
                       help="Fraction of data to augment")
    parser.add_argument("--synonym_replacement", action="store_true",
                       help="Enable synonym replacement augmentation")
    parser.add_argument("--random_insertion", action="store_true",
                       help="Enable random word insertion")
    parser.add_argument("--random_swap", action="store_true",
                       help="Enable random word swapping")
    parser.add_argument("--random_deletion", action="store_true",
                       help="Enable random word deletion")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed datasets")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor
    processor = DatasetProcessor(args.tokenizer_name, args.max_length)
    
    # Load data
    logger.info("Loading dataset...")
    if args.data_format == "csv":
        if not args.input_file:
            raise ValueError("--input_file required for CSV format")
        texts, labels = processor.load_csv_dataset(
            args.input_file, args.text_column, args.label_column
        )
    elif args.data_format == "json":
        if not args.input_file:
            raise ValueError("--input_file required for JSON format")
        texts, labels = processor.load_json_dataset(
            args.input_file, args.text_column, args.label_column
        )
    elif args.data_format == "huggingface":
        if not args.dataset_name:
            raise ValueError("--dataset_name required for HuggingFace format")
        texts, labels = processor.load_huggingface_dataset(
            args.dataset_name, "train", args.text_column, args.label_column
        )
    
    logger.info(f"Loaded {len(texts)} samples with {len(set(labels))} unique labels")
    
    # Apply data augmentation if enabled
    if args.augment:
        logger.info("Applying data augmentation...")
        aug_config = {
            "synonym_replacement": args.synonym_replacement,
            "random_insertion": args.random_insertion,
            "random_swap": args.random_swap,
            "random_deletion": args.random_deletion,
            "synonym_prob": 0.1,
            "insertion_prob": 0.1,
            "swap_prob": 0.1,
            "deletion_prob": 0.1
        }
        
        augmenter = DataAugmenter(aug_config)
        texts, labels = augmenter.augment_dataset(
            texts, labels, args.augmentation_factor
        )
        logger.info(f"Dataset size after augmentation: {len(texts)}")
    
    # Split dataset
    logger.info("Splitting dataset...")
    datasets = processor.split_dataset(
        texts, labels, args.train_size, args.val_size, args.test_size, args.random_seed
    )
    
    # Save datasets
    logger.info("Saving processed datasets...")
    for split_name, dataset in datasets.items():
        output_path = os.path.join(args.output_dir, f"{split_name}_dataset")
        dataset.save_to_disk(output_path)
        logger.info(f"Saved {split_name} dataset to {output_path}")
    
    # Save dataset info
    save_dataset_info(datasets, args.output_dir)
    
    # Save configuration
    config = vars(args)
    config_path = os.path.join(args.output_dir, "preparation_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Data preparation completed. Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()