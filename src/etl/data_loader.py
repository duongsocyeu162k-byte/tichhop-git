"""
Data Loader Module
=================

Handles data extraction from various sources including CSV files,
databases, and APIs.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class to handle data loading from multiple sources.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the DataLoader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_sources = self.config.get('data_sources', {})
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            return {}
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_glassdoor_data(self) -> pd.DataFrame:
        """Load Glassdoor data with specific configuration."""
        config = self.data_sources.get('glassdoor', {})
        file_path = config.get('file_path', './data/DataAnalyst.csv')
        
        return self.load_csv(
            file_path,
            encoding=config.get('encoding', 'utf-8'),
            delimiter=config.get('delimiter', ',')
        )
    
    def load_monster_data(self) -> pd.DataFrame:
        """Load Monster.com data with specific configuration."""
        config = self.data_sources.get('monster', {})
        file_path = config.get('file_path', './data/monster_com-job_sample.csv')
        
        return self.load_csv(
            file_path,
            encoding=config.get('encoding', 'utf-8'),
            delimiter=config.get('delimiter', ',')
        )
    
    def load_naukri_data(self) -> pd.DataFrame:
        """Load Naukri.com data with specific configuration."""
        config = self.data_sources.get('naukri', {})
        file_path = config.get('file_path', './data/naukri_com-job_sample.csv')
        
        return self.load_csv(
            file_path,
            encoding=config.get('encoding', 'utf-8'),
            delimiter=config.get('delimiter', ',')
        )
    
    def load_all_sources(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from all configured sources.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with source names as keys
        """
        data = {}
        
        try:
            data['glassdoor'] = self.load_glassdoor_data()
            logger.info(f"Loaded Glassdoor data: {len(data['glassdoor'])} rows")
        except Exception as e:
            logger.error(f"Error loading Glassdoor data: {e}")
            data['glassdoor'] = pd.DataFrame()
        
        try:
            data['monster'] = self.load_monster_data()
            logger.info(f"Loaded Monster data: {len(data['monster'])} rows")
        except Exception as e:
            logger.error(f"Error loading Monster data: {e}")
            data['monster'] = pd.DataFrame()
        
        try:
            data['naukri'] = self.load_naukri_data()
            logger.info(f"Loaded Naukri data: {len(data['naukri'])} rows")
        except Exception as e:
            logger.error(f"Error loading Naukri data: {e}")
            data['naukri'] = pd.DataFrame()
        
        return data
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Get summary statistics for loaded data.
        
        Args:
            data: Dictionary of DataFrames
            
        Returns:
            Dict[str, Dict]: Summary statistics
        """
        summary = {}
        
        for source, df in data.items():
            if not df.empty:
                summary[source] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'missing_values': df.isnull().sum().to_dict(),
                    'data_types': df.dtypes.to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum()
                }
            else:
                summary[source] = {
                    'rows': 0,
                    'columns': 0,
                    'missing_values': {},
                    'data_types': {},
                    'memory_usage': 0
                }
        
        return summary


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load all data sources
    all_data = loader.load_all_sources()
    
    # Print summary
    summary = loader.get_data_summary(all_data)
    for source, stats in summary.items():
        print(f"\n{source.upper()} Data Summary:")
        print(f"  Rows: {stats['rows']}")
        print(f"  Columns: {stats['columns']}")
        print(f"  Memory Usage: {stats['memory_usage']:,} bytes")
