"""
Batch data loader for processing files with various formats.
"""
import os
import json
import csv
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Generator, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for data loader."""
    batch_size: int
    max_workers: int
    supported_formats: List[str]
    encoding: str
    error_handling: str  # 'skip', 'raise', 'log'
    
    @classmethod
    def from_env(cls) -> 'LoaderConfig':
        """Load configuration from environment variables."""
        return cls(
            batch_size=int(os.getenv('LOADER_BATCH_SIZE', '1000')),
            max_workers=int(os.getenv('LOADER_MAX_WORKERS', '4')),
            supported_formats=os.getenv('LOADER_FORMATS', 'csv,json,txt,parquet,yaml').split(','),
            encoding=os.getenv('LOADER_ENCODING', 'utf-8'),
            error_handling=os.getenv('LOADER_ERROR_HANDLING', 'log')
        )


class DataLoader:
    """Batch data loader with support for multiple file formats."""
    
    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        Initialize data loader.
        
        Args:
            config: Loader configuration. If None, loads from environment.
        """
        self.config = config or LoaderConfig.from_env()
        self.stats = {
            'files_processed': 0,
            'records_loaded': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def load_file(self, 
                  file_path: Union[str, Path],
                  file_format: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Load data from a single file.
        
        Args:
            file_path: Path to the file
            file_format: File format. If None, inferred from extension.
            
        Yields:
            Dictionary containing record data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self._handle_error(f"File not found: {file_path}")
            return
        
        # Infer format from extension if not provided
        if file_format is None:
            file_format = file_path.suffix.lstrip('.').lower()
        
        if file_format not in self.config.supported_formats:
            self._handle_error(f"Unsupported format: {file_format}")
            return
        
        try:
            logger.info(f"Loading file: {file_path} (format: {file_format})")
            
            if file_format == 'csv':
                yield from self._load_csv(file_path)
            elif file_format == 'json':
                yield from self._load_json(file_path)
            elif file_format == 'txt':
                yield from self._load_text(file_path)
            elif file_format == 'parquet':
                yield from self._load_parquet(file_path)
            elif file_format == 'yaml':
                yield from self._load_yaml(file_path)
            else:
                self._handle_error(f"Format not implemented: {file_format}")
                
        except Exception as e:
            self._handle_error(f"Error loading file {file_path}: {e}")
    
    def load_directory(self,
                      directory_path: Union[str, Path],
                      pattern: str = "*",
                      recursive: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Load data from all matching files in a directory.
        
        Args:
            directory_path: Path to directory
            pattern: File pattern to match (e.g., "*.csv")
            recursive: Whether to search recursively
            
        Yields:
            Dictionary containing record data with file metadata
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            self._handle_error(f"Directory not found: {directory_path}")
            return
        
        # Find matching files
        if recursive:
            files = list(directory_path.rglob(pattern))
        else:
            files = list(directory_path.glob(pattern))
        
        logger.info(f"Found {len(files)} files matching pattern: {pattern}")
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._load_file_batch, file_path): file_path
                for file_path in files
            }
            
            # Yield results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    batch_data = future.result()
                    for record in batch_data:
                        yield record
                except Exception as e:
                    self._handle_error(f"Error processing {file_path}: {e}")
    
    def _load_file_batch(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load file and return batch of records."""
        records = []
        for record in self.load_file(file_path):
            record['_source_file'] = str(file_path)
            record['_loaded_at'] = datetime.utcnow().isoformat()
            records.append(record)
        return records
    
    def _load_csv(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Load CSV file."""
        try:
            # Use pandas for efficient chunk processing
            for chunk in pd.read_csv(file_path, 
                                   chunksize=self.config.batch_size,
                                   encoding=self.config.encoding):
                for _, row in chunk.iterrows():
                    yield row.to_dict()
                    self.stats['records_loaded'] += 1
        except Exception as e:
            self._handle_error(f"CSV loading error: {e}")
    
    def _load_json(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Load JSON file (supports both single object and JSON lines)."""
        try:
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                # Try to load as single JSON object
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                            self.stats['records_loaded'] += 1
                    else:
                        yield data
                        self.stats['records_loaded'] += 1
                except json.JSONDecodeError:
                    # Try as JSON lines
                    f.seek(0)
                    for line_num, line in enumerate(f, 1):
                        try:
                            if line.strip():
                                yield json.loads(line)
                                self.stats['records_loaded'] += 1
                        except json.JSONDecodeError as e:
                            self._handle_error(f"JSON decode error at line {line_num}: {e}")
        except Exception as e:
            self._handle_error(f"JSON loading error: {e}")
    
    def _load_text(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Load text file line by line."""
        try:
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        yield {
                            'text': line.strip(),
                            'line_number': line_num
                        }
                        self.stats['records_loaded'] += 1
        except Exception as e:
            self._handle_error(f"Text loading error: {e}")
    
    def _load_parquet(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Load Parquet file."""
        try:
            df = pd.read_parquet(file_path)
            for _, row in df.iterrows():
                yield row.to_dict()
                self.stats['records_loaded'] += 1
        except Exception as e:
            self._handle_error(f"Parquet loading error: {e}")
    
    def _load_yaml(self, file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Load YAML file."""
        try:
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                data = yaml.safe_load_all(f)
                for doc in data:
                    if doc:
                        yield doc
                        self.stats['records_loaded'] += 1
        except Exception as e:
            self._handle_error(f"YAML loading error: {e}")
    
    def load_batch(self, 
                   sources: List[Union[str, Path]],
                   process_func: Optional[Callable[[Dict[str, Any]], Any]] = None) -> List[Any]:
        """
        Load and optionally process data from multiple sources.
        
        Args:
            sources: List of file or directory paths
            process_func: Optional function to process each record
            
        Returns:
            List of processed records
        """
        self.stats['start_time'] = datetime.utcnow()
        results = []
        
        try:
            for source in sources:
                source_path = Path(source)
                
                if source_path.is_file():
                    generator = self.load_file(source_path)
                    self.stats['files_processed'] += 1
                elif source_path.is_dir():
                    generator = self.load_directory(source_path)
                else:
                    self._handle_error(f"Invalid source: {source}")
                    continue
                
                # Process records in batches
                batch = []
                for record in generator:
                    if process_func:
                        record = process_func(record)
                    
                    batch.append(record)
                    
                    if len(batch) >= self.config.batch_size:
                        results.extend(batch)
                        batch = []
                
                # Add remaining records
                if batch:
                    results.extend(batch)
            
        finally:
            self.stats['end_time'] = datetime.utcnow()
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            logger.info(f"Loading completed: {self.stats['files_processed']} files, "
                       f"{self.stats['records_loaded']} records, "
                       f"{self.stats['errors']} errors, "
                       f"{duration:.2f} seconds")
        
        return results
    
    def _handle_error(self, error_msg: str):
        """Handle errors based on configuration."""
        self.stats['errors'] += 1
        
        if self.config.error_handling == 'raise':
            raise RuntimeError(error_msg)
        elif self.config.error_handling == 'log':
            logger.error(error_msg)
        elif self.config.error_handling == 'skip':
            logger.debug(error_msg)
    
    def save_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Save loader statistics as checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.stats, f)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load statistics from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                self.stats = pickle.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create loader
    loader = DataLoader()
    
    # Example: Load single file
    print("Loading single CSV file:")
    for record in loader.load_file("data/sample.csv"):
        print(record)
        break  # Just show first record
    
    # Example: Load directory
    print("\nLoading all JSON files from directory:")
    for record in loader.load_directory("data/", pattern="*.json"):
        print(record)
        break  # Just show first record
    
    # Example: Batch loading with processing
    def process_record(record):
        """Example processing function."""
        if 'text' in record:
            record['text_length'] = len(record['text'])
        return record
    
    print("\nBatch loading with processing:")
    results = loader.load_batch(
        sources=["data/file1.json", "data/texts/"],
        process_func=process_record
    )
    print(f"Loaded {len(results)} records")