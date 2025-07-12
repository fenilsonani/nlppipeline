#!/usr/bin/env python3
"""
Hyperparameter optimization script for sentiment analysis models.

This script provides automated hyperparameter tuning using various optimization
algorithms including grid search, random search, Bayesian optimization, and
Optuna-based optimization.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import torch
from datasets import Dataset, load_from_disk
from sklearn.model_selection import ParameterGrid, ParameterSampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterSearcher:
    """Hyperparameter optimization for sentiment analysis models."""
    
    def __init__(self, model_name: str, train_dataset: Dataset, eval_dataset: Dataset,
                 base_output_dir: str, num_labels: int, search_space: Dict[str, Any],
                 use_wandb: bool = False, wandb_project: str = "sentiment-hp-search"):
        """
        Initialize hyperparameter searcher.
        
        Args:
            model_name: Pre-trained model name
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            base_output_dir: Base directory for outputs
            num_labels: Number of sentiment classes
            search_space: Hyperparameter search space
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
        """
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.base_output_dir = base_output_dir
        self.num_labels = num_labels
        self.search_space = search_space
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create base output directory
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Results storage
        self.results = []
        self.best_params = None
        self.best_score = -float('inf')
        
        logger.info(f"Initialized hyperparameter searcher for {model_name}")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    def objective_function(self, params: Dict[str, Any], trial_id: Optional[str] = None) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            params: Hyperparameters to evaluate
            trial_id: Unique identifier for this trial
            
        Returns:
            Evaluation metric score
        """
        trial_id = trial_id or f"trial_{len(self.results)}"
        
        logger.info(f"Starting {trial_id} with params: {params}")
        
        # Set random seed for reproducibility
        set_seed(params.get('seed', 42))
        
        # Create trial output directory
        trial_dir = os.path.join(self.base_output_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        
        try:
            # Initialize model
            config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, config=config
            )
            
            # Create training arguments
            training_args = TrainingArguments(
                output_dir=trial_dir,
                num_train_epochs=params.get('num_train_epochs', 3),
                per_device_train_batch_size=params.get('per_device_train_batch_size', 16),
                per_device_eval_batch_size=params.get('per_device_eval_batch_size', 32),
                gradient_accumulation_steps=params.get('gradient_accumulation_steps', 1),
                learning_rate=params.get('learning_rate', 2e-5),
                weight_decay=params.get('weight_decay', 0.01),
                warmup_ratio=params.get('warmup_ratio', 0.1),
                logging_dir=os.path.join(trial_dir, "logs"),
                logging_steps=params.get('logging_steps', 100),
                evaluation_strategy="steps",
                eval_steps=params.get('eval_steps', 200),
                save_strategy="steps",
                save_steps=params.get('save_steps', 200),
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1_weighted",
                greater_is_better=True,
                dataloader_num_workers=params.get('dataloader_num_workers', 4),
                fp16=params.get('fp16', False),
                gradient_checkpointing=params.get('gradient_checkpointing', False),
                report_to="wandb" if self.use_wandb else None,
                run_name=f"{trial_id}" if self.use_wandb else None,
                disable_tqdm=True,  # Reduce output noise
            )
            
            # Initialize W&B for this trial
            if self.use_wandb:
                wandb.init(
                    project=self.wandb_project,
                    name=trial_id,
                    config=params,
                    reinit=True
                )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self._compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )
            
            # Train model
            trainer.train()
            
            # Evaluate model
            eval_results = trainer.evaluate()
            score = eval_results.get('eval_f1_weighted', 0.0)
            
            # Store results
            result = {
                'trial_id': trial_id,
                'params': params.copy(),
                'score': score,
                'eval_results': eval_results,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                
                # Save best model
                best_model_dir = os.path.join(self.base_output_dir, "best_model")
                trainer.save_model(best_model_dir)
                
                logger.info(f"New best score: {score:.4f} with params: {params}")
            
            # Clean up trial directory to save space (keep only best)
            if score < self.best_score:
                import shutil
                shutil.rmtree(trial_dir, ignore_errors=True)
            
            # Finish W&B run
            if self.use_wandb:
                wandb.log({"final_score": score})
                wandb.finish()
            
            logger.info(f"Completed {trial_id}: score = {score:.4f}")
            
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            
            # Finish W&B run on failure
            if self.use_wandb:
                wandb.finish()
                
            return -1.0  # Return very low score for failed trials
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import accuracy_score, f1_score
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1_weighted': f1_score(labels, predictions, average='weighted'),
            'f1_macro': f1_score(labels, predictions, average='macro'),
        }
    
    def grid_search(self, max_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform grid search over hyperparameters.
        
        Args:
            max_trials: Maximum number of trials to run
            
        Returns:
            Best parameters and results
        """
        logger.info("Starting grid search...")
        
        # Generate parameter grid
        param_grid = list(ParameterGrid(self.search_space))
        
        if max_trials and len(param_grid) > max_trials:
            # Randomly sample from grid if too large
            np.random.shuffle(param_grid)
            param_grid = param_grid[:max_trials]
        
        logger.info(f"Grid search will evaluate {len(param_grid)} parameter combinations")
        
        # Evaluate each parameter combination
        for i, params in enumerate(param_grid):
            trial_id = f"grid_trial_{i:03d}"
            self.objective_function(params, trial_id)
            
            # Save intermediate results
            self._save_results()
        
        return self._get_summary()
    
    def random_search(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Perform random search over hyperparameters.
        
        Args:
            n_trials: Number of random trials to run
            
        Returns:
            Best parameters and results
        """
        logger.info(f"Starting random search with {n_trials} trials...")
        
        # Generate random parameter combinations
        param_sampler = ParameterSampler(self.search_space, n_iter=n_trials, random_state=42)
        
        # Evaluate each parameter combination
        for i, params in enumerate(param_sampler):
            trial_id = f"random_trial_{i:03d}"
            self.objective_function(params, trial_id)
            
            # Save intermediate results
            self._save_results()
        
        return self._get_summary()
    
    def bayesian_search(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Perform Bayesian optimization using Optuna.
        
        Args:
            n_trials: Number of trials to run
            
        Returns:
            Best parameters and results
        """
        logger.info(f"Starting Bayesian optimization with {n_trials} trials...")
        
        def optuna_objective(trial):
            # Sample hyperparameters
            params = {}
            
            for param_name, param_config in self.search_space.items():
                if isinstance(param_config, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
                elif isinstance(param_config, dict):
                    if param_config.get('type') == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config.get('type') == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                else:
                    # Fixed parameter
                    params[param_name] = param_config
            
            # Evaluate parameters
            trial_id = f"bayesian_trial_{trial.number:03d}"
            score = self.objective_function(params, trial_id)
            
            # Save intermediate results
            self._save_results()
            
            return score
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        
        # Optimize
        study.optimize(optuna_objective, n_trials=n_trials)
        
        # Update best parameters from Optuna results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Save Optuna study
        study_path = os.path.join(self.base_output_dir, "optuna_study.pkl")
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        return self._get_summary()
    
    def _save_results(self):
        """Save current results to file."""
        results_path = os.path.join(self.base_output_dir, "search_results.json")
        
        with open(results_path, 'w') as f:
            json.dump({
                'results': self.results,
                'best_params': self.best_params,
                'best_score': self.best_score,
                'search_space': self.search_space
            }, f, indent=2)
    
    def _get_summary(self) -> Dict[str, Any]:
        """Get search summary."""
        summary = {
            'total_trials': len(self.results),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'search_space': self.search_space,
            'all_results': self.results
        }
        
        # Save final summary
        summary_path = os.path.join(self.base_output_dir, "search_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def create_search_space(args: argparse.Namespace) -> Dict[str, Any]:
    """Create hyperparameter search space from arguments."""
    search_space = {}
    
    # Learning rate
    if args.learning_rate_range:
        lr_low, lr_high = map(float, args.learning_rate_range.split(','))
        search_space['learning_rate'] = {'type': 'float', 'low': lr_low, 'high': lr_high, 'log': True}
    else:
        search_space['learning_rate'] = [1e-5, 2e-5, 3e-5, 5e-5]
    
    # Batch size
    if args.batch_size_options:
        search_space['per_device_train_batch_size'] = [int(x) for x in args.batch_size_options.split(',')]
    else:
        search_space['per_device_train_batch_size'] = [8, 16, 32]
    
    # Number of epochs
    if args.epochs_range:
        epoch_low, epoch_high = map(int, args.epochs_range.split(','))
        search_space['num_train_epochs'] = {'type': 'int', 'low': epoch_low, 'high': epoch_high}
    else:
        search_space['num_train_epochs'] = [2, 3, 4, 5]
    
    # Weight decay
    if args.weight_decay_range:
        wd_low, wd_high = map(float, args.weight_decay_range.split(','))
        search_space['weight_decay'] = {'type': 'float', 'low': wd_low, 'high': wd_high}
    else:
        search_space['weight_decay'] = [0.0, 0.01, 0.1]
    
    # Warmup ratio
    if args.warmup_ratio_range:
        wu_low, wu_high = map(float, args.warmup_ratio_range.split(','))
        search_space['warmup_ratio'] = {'type': 'float', 'low': wu_low, 'high': wu_high}
    else:
        search_space['warmup_ratio'] = [0.0, 0.1, 0.2]
    
    # Gradient accumulation steps
    if args.grad_accum_options:
        search_space['gradient_accumulation_steps'] = [int(x) for x in args.grad_accum_options.split(',')]
    else:
        search_space['gradient_accumulation_steps'] = [1, 2, 4]
    
    # Fixed parameters
    search_space['per_device_eval_batch_size'] = 32
    search_space['eval_steps'] = 200
    search_space['save_steps'] = 200
    search_space['logging_steps'] = 100
    search_space['dataloader_num_workers'] = 4
    search_space['fp16'] = args.fp16
    search_space['gradient_checkpointing'] = args.gradient_checkpointing
    search_space['seed'] = 42
    
    return search_space


def main():
    """Main hyperparameter search function."""
    parser = argparse.ArgumentParser(description="Hyperparameter search for sentiment analysis")
    
    # Model and data arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="Pre-trained model name")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed datasets")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for search results")
    parser.add_argument("--num_labels", type=int, default=3,
                       help="Number of sentiment classes")
    
    # Search configuration
    parser.add_argument("--search_method", type=str, default="bayesian",
                       choices=["grid", "random", "bayesian"],
                       help="Hyperparameter search method")
    parser.add_argument("--n_trials", type=int, default=50,
                       help="Number of trials for random/Bayesian search")
    parser.add_argument("--max_trials", type=int,
                       help="Maximum trials for grid search")
    
    # Hyperparameter search spaces
    parser.add_argument("--learning_rate_range", type=str,
                       help="Learning rate range as 'low,high' (e.g., '1e-5,5e-5')")
    parser.add_argument("--batch_size_options", type=str,
                       help="Batch size options as comma-separated (e.g., '8,16,32')")
    parser.add_argument("--epochs_range", type=str,
                       help="Epochs range as 'low,high' (e.g., '2,5')")
    parser.add_argument("--weight_decay_range", type=str,
                       help="Weight decay range as 'low,high' (e.g., '0.0,0.1')")
    parser.add_argument("--warmup_ratio_range", type=str,
                       help="Warmup ratio range as 'low,high' (e.g., '0.0,0.2')")
    parser.add_argument("--grad_accum_options", type=str,
                       help="Gradient accumulation options as comma-separated (e.g., '1,2,4')")
    
    # Training options
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Use gradient checkpointing")
    
    # Monitoring
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="sentiment-hp-search",
                       help="Weights & Biases project name")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load datasets
    logger.info(f"Loading datasets from {args.data_dir}")
    
    train_dataset_path = os.path.join(args.data_dir, "train_dataset")
    eval_dataset_path = os.path.join(args.data_dir, "validation_dataset")
    
    if not os.path.exists(train_dataset_path):
        raise ValueError(f"Training dataset not found at {train_dataset_path}")
    if not os.path.exists(eval_dataset_path):
        raise ValueError(f"Validation dataset not found at {eval_dataset_path}")
    
    train_dataset = load_from_disk(train_dataset_path)
    eval_dataset = load_from_disk(eval_dataset_path)
    
    # Get number of labels from dataset if not provided
    if args.num_labels == 3:  # Default value
        unique_labels = set(train_dataset["labels"])
        args.num_labels = len(unique_labels)
        logger.info(f"Detected {args.num_labels} unique labels: {sorted(unique_labels)}")
    
    # Create search space
    search_space = create_search_space(args)
    logger.info(f"Search space: {search_space}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize searcher
    searcher = HyperparameterSearcher(
        model_name=args.model_name,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        base_output_dir=args.output_dir,
        num_labels=args.num_labels,
        search_space=search_space,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Run search
    if args.search_method == "grid":
        results = searcher.grid_search(max_trials=args.max_trials)
    elif args.search_method == "random":
        results = searcher.random_search(n_trials=args.n_trials)
    elif args.search_method == "bayesian":
        results = searcher.bayesian_search(n_trials=args.n_trials)
    
    # Print results
    logger.info("Hyperparameter search completed!")
    logger.info(f"Best score: {results['best_score']:.4f}")
    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Total trials: {results['total_trials']}")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()