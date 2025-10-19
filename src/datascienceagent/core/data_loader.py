"""
Data Loading Module - Comprehensive CSV Data Loading

This module provides robust data loading functionality with:
- Proper path validation and resolution
- Multiple path format support (relative, absolute, with/without leading slash)
- Comprehensive error handling
- Data validation
- Detailed logging

Author: Data Science Agent System
Created: 2025-10-19
"""

import pandas as pd
from loguru import logger
from typing import Optional, Dict, Any, Union
from pathlib import Path
import os




class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass


class DataLoader:
    """
    Comprehensive data loader with robust path resolution and error handling.
    
    Handles multiple path formats:
    - Relative: 'data/sales_data.csv'
    - Absolute: '/data/sales_data.csv'
    - With project root: '/home/claude/data/sales_data.csv'
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            project_root: Root directory for resolving relative paths.
                         Defaults to current working directory.
        """
        self.project_root = project_root or Path.cwd()
        logger.info(f"DataLoader initialized with project_root: {self.project_root}")
    
    def resolve_path(self, file_path: str) -> Path:
        """
        Resolve file path to absolute path, trying multiple strategies.
        
        Args:
            file_path: Path to resolve (can be relative or absolute)
            
        Returns:
            Resolved absolute Path
            
        Raises:
            DataLoadError: If path cannot be resolved
        """
        if not file_path:
            raise DataLoadError("File path is empty or None")
        
        # Remove leading/trailing whitespace
        file_path = file_path.strip()
        
        # Strategy 1: Try as absolute path
        path = Path(file_path)
        if path.is_absolute() and path.exists():
            logger.info(f"✓ Path resolved (absolute): {path}")
            return path
        
        # Strategy 2: Try relative to project root
        path = self.project_root / file_path.lstrip('/')
        if path.exists():
            logger.info(f"✓ Path resolved (relative to project_root): {path}")
            return path
        
        # Strategy 3: Try relative to cwd
        path = Path.cwd() / file_path.lstrip('/')
        if path.exists():
            logger.info(f"✓ Path resolved (relative to cwd): {path}")
            return path
        
        # Strategy 4: Try common data directories
        for data_dir in ['data', 'datasets', '../data']:
            path = self.project_root / data_dir / Path(file_path).name
            if path.exists():
                logger.info(f"✓ Path resolved (in {data_dir}): {path}")
                return path
        
        # If nothing works, raise error with diagnostic info
        error_msg = f"""
Cannot find file: {file_path}

Tried the following locations:
1. Absolute path: {Path(file_path)}
2. Relative to project root: {self.project_root / file_path.lstrip('/')}
3. Relative to cwd: {Path.cwd() / file_path.lstrip('/')}
4. Common data directories in project root

Current working directory: {Path.cwd()}
Project root: {self.project_root}
File exists check: {Path(file_path).exists()}
"""
        raise DataLoadError(error_msg)
    
    def load_from_data_source(
        self,
        data_source: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Load data from a data_source dictionary.
        
        This is the main entry point when data_source comes from
        the workflow orchestrator.
        
        Args:
            data_source: Dictionary with keys:
                - type: 'csv', 'excel', 'parquet', etc.
                - path: file path
                - parameters: optional dict with loading parameters
                
        Returns:
            Loaded DataFrame
            
        Raises:
            DataLoadError: If data cannot be loaded
            
        Example:
            >>> data_source = {
            ...     "type": "csv",
            ...     "path": "data/sales_data.csv",
            ...     "parameters": {"delimiter": ",", "header": True}
            ... }
            >>> df = loader.load_from_data_source(data_source)
        """
        if not data_source:
            raise DataLoadError("data_source is None or empty")
        
        # Extract and validate components
        data_type = data_source.get('type', '').lower()
        file_path = data_source.get('path') or data_source.get('location')
        parameters = data_source.get('parameters', {})
        
        logger.info(f"Loading data from data_source:")
        logger.info(f"  - type: {data_type}")
        logger.info(f"  - path: {file_path}")
        logger.info(f"  - parameters: {parameters}")
        
        if not file_path:
            raise DataLoadError(
                f"No path found in data_source. "
                f"Available keys: {list(data_source.keys())}"
            )
        
        # Route to appropriate loader
        if data_type == 'csv':
            return self.load_csv(
                file_path=file_path,
                delimiter=parameters.get('delimiter', ','),
                header=parameters.get('header', True)
            )
        elif data_type == 'excel':
            return self.load_excel(file_path, **parameters)
        elif data_type == 'parquet':
            return self.load_parquet(file_path)
        else:
            raise DataLoadError(
                f"Unsupported data type: {data_type}. "
                f"Supported types: csv, excel, parquet"
            )
    
    def load_csv(
        self,
        file_path: Union[str, Path],
        delimiter: str = ',',
        header: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a CSV file with comprehensive error handling.
        
        Args:
            file_path: Path to CSV file (can be relative or absolute)
            delimiter: Column delimiter
            header: Whether file has header row
            **kwargs: Additional parameters for pd.read_csv
            
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            DataLoadError: If data cannot be loaded or is invalid
        """
        logger.info(f"=" * 70)
        logger.info(f"LOADING CSV DATA")
        logger.info(f"=" * 70)
        logger.info(f"File path (input): {file_path}")
        logger.info(f"Delimiter: '{delimiter}'")
        logger.info(f"Header: {header}")
        
        # Handle None or empty path
        if file_path is None:
            raise DataLoadError(
                "File path is None. "
                "This usually means the data_source was not properly passed "
                "from the orchestrator to the data loading function."
            )
        
        try:
            # Resolve path
            resolved_path = self.resolve_path(str(file_path))
            logger.info(f"File path (resolved): {resolved_path}")
            logger.info(f"File exists: {resolved_path.exists()}")
            logger.info(f"File size: {resolved_path.stat().st_size if resolved_path.exists() else 'N/A'} bytes")
            
            # Load data
            logger.info("Reading CSV file...")
            df = pd.read_csv(
                resolved_path,
                delimiter=delimiter,
                header=0 if header else None,
                **kwargs
            )
            
            # Validate loaded data
            self._validate_dataframe(df, str(resolved_path))
            
            logger.info(f"✓ Data loaded successfully!")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            logger.info(f"=" * 70)
            
            return df
            
        except FileNotFoundError as e:
            raise DataLoadError(f"File not found: {file_path}") from e
        except pd.errors.ParserError as e:
            raise DataLoadError(f"CSV parsing error: {e}") from e
        except Exception as e:
            raise DataLoadError(f"Unexpected error loading CSV: {e}") from e
    
    def load_excel(
        self,
        file_path: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from an Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index to load
            **kwargs: Additional parameters for pd.read_excel
            
        Returns:
            Loaded DataFrame
        """
        try:
            resolved_path = self.resolve_path(str(file_path))
            logger.info(f"Loading Excel file: {resolved_path}")
            
            df = pd.read_excel(
                resolved_path,
                sheet_name=sheet_name,
                **kwargs
            )
            
            self._validate_dataframe(df, str(resolved_path))
            logger.info(f"✓ Excel data loaded: {df.shape}")
            
            return df
            
        except Exception as e:
            raise DataLoadError(f"Error loading Excel file: {e}") from e
    
    def load_parquet(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a Parquet file.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional parameters for pd.read_parquet
            
        Returns:
            Loaded DataFrame
        """
        try:
            resolved_path = self.resolve_path(str(file_path))
            logger.info(f"Loading Parquet file: {resolved_path}")
            
            df = pd.read_parquet(resolved_path, **kwargs)
            
            self._validate_dataframe(df, str(resolved_path))
            logger.info(f"✓ Parquet data loaded: {df.shape}")
            
            return df
            
        except Exception as e:
            raise DataLoadError(f"Error loading Parquet file: {e}") from e
    
    def _validate_dataframe(self, df: pd.DataFrame, source: str) -> None:
        """
        Validate loaded DataFrame.
        
        Args:
            df: DataFrame to validate
            source: Data source for error messages
            
        Raises:
            DataLoadError: If validation fails
        """
        if df is None:
            raise DataLoadError(f"Loaded data is None from {source}")
        
        if df.empty:
            logger.warning(f"⚠️  Loaded data is empty from {source}")
        
        if len(df.columns) == 0:
            raise DataLoadError(f"No columns found in {source}")
        
        # Check for duplicate columns
        if len(df.columns) != len(set(df.columns)):
            duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
            logger.warning(f"⚠️  Duplicate columns found: {set(duplicates)}")


def load_csv_data(
    file_path: Optional[str],
    delimiter: str = ',',
    header: bool = True
) -> Optional[pd.DataFrame]:
    """
    Legacy function for backward compatibility.
    
    Wraps the DataLoader class for simple CSV loading.
    
    Args:
        file_path: Path to CSV file
        delimiter: Column delimiter
        header: Whether file has header row
        
    Returns:
        DataFrame if successful, None otherwise
    """
    if file_path is None:
        logger.error("File path is None. Please provide a valid CSV file path.")
        return None
    
    try:
        loader = DataLoader()
        return loader.load_csv(file_path, delimiter, header)
    except DataLoadError as e:
        logger.error(f"Data loading failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Configure logging for demo
    
    print("\n" + "="*70)
    print("DATA LOADER MODULE - DEMO")
    print("="*70)
    
    # Example 1: Load from data_source dictionary (orchestrator format)
    print("\n1. Loading from data_source dictionary:")
    data_source = {
        "type": "csv",
        "path": "data/sales_data.csv",
        "parameters": {
            "delimiter": ",",
            "header": True
        }
    }
    
    loader = DataLoader()
    try:
        df = loader.load_from_data_source(data_source)
        print(f"✓ Success! Loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
    except DataLoadError as e:
        print(f"✗ Error: {e}")
    
    # Example 2: Direct CSV loading
    print("\n2. Direct CSV loading:")
    try:
        df = loader.load_csv("data/sales_data.csv")
        print(f"✓ Success! Loaded {len(df)} rows")
    except DataLoadError as e:
        print(f"✗ Error: {e}")
    
    # Example 3: Legacy function
    print("\n3. Legacy load_csv_data function:")
    df = load_csv_data("data/sales_data.csv")
    if df is not None:
        print(f"✓ Success! Loaded {len(df)} rows")
    else:
        print("✗ Error: See logs above")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)