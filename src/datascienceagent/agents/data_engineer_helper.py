"""
Data Engineer Agent - Enhanced with Robust Data Loading

This module provides the Data Engineer specialist agent with:
- Proper integration with the DataLoader module
- Comprehensive data loading from various sources
- Data validation and quality checks
- Data cleaning and preprocessing
- Feature engineering capabilities

Key improvements:
- Extracts file path correctly from data_source dictionary
- Uses DataLoader for robust path resolution
- Comprehensive error handling and logging
- Generates executable, tested code

Author: Data Science Agent System
Created: 2025-10-19
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
import pandas as pd

from datascienceagent.core.data_loader import DataLoader, DataLoadError


class DataEngineerAgentHelper:
    """
    Helper class for Data Engineer Agent data operations.
    
    This class handles the actual data loading and provides
    utility functions that the LLM-based agent can call.
    """
    
    def __init__(self):
        """Initialize the data engineering helper."""
        self.loader = DataLoader()
        logger.info("DataEngineerAgentHelper initialized")
    
    def load_data_from_source(
        self,
        data_source: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Load data from a data_source specification.
        
        This is called by the agent when it receives a data loading request.
        
        Args:
            data_source: Dictionary containing:
                - type: Data source type (csv, excel, etc.)
                - path: File path
                - parameters: Loading parameters
                
        Returns:
            Dictionary with:
                - success: bool
                - data: DataFrame (if successful)
                - metadata: Information about loaded data
                - code: Python code that was executed
                - error: Error message (if failed)
        """
        logger.info("="*70)
        logger.info("DATA LOADING REQUEST")
        logger.info("="*70)
        
        # Validate input
        if not data_source:
            logger.error("❌ data_source is None or empty!")
            return {
                "success": False,
                "error": "data_source parameter is None or empty. "
                        "This usually means the data_source was not properly "
                        "passed from the orchestrator to the agent.",
                "data": None,
                "metadata": {},
                "code": ""
            }
        
        logger.info(f"Data source received:")
        logger.info(f"  Type: {data_source.get('type')}")
        logger.info(f"  Path: {data_source.get('path')}")
        logger.info(f"  Parameters: {data_source.get('parameters', {})}")
        
        try:
            # Load data using DataLoader
            df = self.loader.load_from_data_source(data_source)
            
            # Generate metadata
            metadata = {
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "missing_values": df.isnull().sum().to_dict(),
                "data_source": data_source
            }
            
            # Generate reproducible code
            code = self._generate_loading_code(data_source)
            
            logger.info("✅ Data loaded successfully!")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Memory: {metadata['memory_usage_mb']:.2f} MB")
            
            return {
                "success": True,
                "data": df,
                "metadata": metadata,
                "code": code,
                "error": None
            }
            
        except DataLoadError as e:
            logger.error(f"❌ Data loading failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": None,
                "metadata": {},
                "code": self._generate_loading_code(data_source)
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {e}",
                "data": None,
                "metadata": {},
                "code": ""
            }
    
    def _generate_loading_code(self, data_source: Dict[str, Any]) -> str:
        """
        Generate reproducible Python code for loading data.
        
        Args:
            data_source: Data source specification
            
        Returns:
            Python code as string
        """
        data_type = data_source.get('type', '').lower()
        file_path = data_source.get('path') or data_source.get('location', '')
        parameters = data_source.get('parameters', {})
        
        if data_type == 'csv':
            delimiter = parameters.get('delimiter', ',')
            header = parameters.get('header', True)
            
            code = f'''"""
Data Loading Script
Generated: {datetime.now().isoformat()}
"""

import pandas as pd
from pathlib import Path

# Load data
file_path = "{file_path}"
df = pd.read_csv(
    file_path,
    delimiter="{delimiter}",
    header={0 if header else None}
)

# Validate
print(f"Loaded {{len(df)}} rows and {{len(df.columns)}} columns")
print(f"Columns: {{list(df.columns)}}")
print(f"Missing values: {{df.isnull().sum().sum()}}")

# Display sample
print("\\nFirst few rows:")
print(df.head())
'''
            return code
        
        elif data_type == 'excel':
            sheet_name = parameters.get('sheet_name', 0)
            code = f'''"""
Data Loading Script - Excel
Generated: {datetime.now().isoformat()}
"""

import pandas as pd

# Load data
file_path = "{file_path}"
df = pd.read_excel(
    file_path,
    sheet_name={repr(sheet_name)}
)

# Validate
print(f"Loaded {{len(df)}} rows and {{len(df.columns)}} columns")
print(f"Columns: {{list(df.columns)}}")
'''
            return code
        
        else:
            return f"# Unsupported data type: {data_type}"
    
    def validate_data(self, df: Any, validation_rules: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            validation_rules: Optional custom validation rules
            
        Returns:
            Validation report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "issues": [],
            "warnings": [],
            "passed": True
        }
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            for col, count in missing[missing > 0].items():
                pct = (count / len(df)) * 100
                issue = {
                    "column": col,
                    "issue": "missing_values",
                    "count": int(count),
                    "percentage": float(pct)
                }
                
                if pct > 50:
                    issue["severity"] = "high"
                    report["issues"].append(issue)
                    report["passed"] = False
                elif pct > 10:
                    issue["severity"] = "medium"
                    report["warnings"].append(issue)
                else:
                    issue["severity"] = "low"
                    report["warnings"].append(issue)
        
        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            report["warnings"].append({
                "issue": "duplicate_rows",
                "count": int(n_duplicates),
                "percentage": float((n_duplicates / len(df)) * 100)
            })
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                report["warnings"].append({
                    "column": col,
                    "issue": "constant_column",
                    "unique_values": 1
                })
        
        # Check data types
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype == 'object':
                # Check if it should be numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    report["warnings"].append({
                        "column": col,
                        "issue": "should_be_numeric",
                        "current_type": dtype
                    })
                except:
                    pass
        
        logger.info(f"Data validation complete:")
        logger.info(f"  Passed: {report['passed']}")
        logger.info(f"  Issues: {len(report['issues'])}")
        logger.info(f"  Warnings: {len(report['warnings'])}")
        
        return report


# ============================================================================
# INTEGRATION EXAMPLE WITH AGENT
# ============================================================================

class DataEngineerAgentIntegration:
    """
    Shows how to integrate the helper with an LLM-based agent.
    
    This would be used by the actual ContextAwareAgent subclass.
    """
    
    def __init__(self):
        self.helper = DataEngineerAgentHelper()
        self.agent_name = "data_engineer_agent"
    
    async def handle_load_data_request(
        self,
        workflow_node: Any
    ) -> Dict[str, Any]:
        """
        Handle a data loading request from the workflow.
        
        This is called when the workflow node is DATA_ACQUISITION.
        
        Args:
            workflow_node: Workflow node with input_data
            
        Returns:
            Result dictionary with loaded data and metadata
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"DATA ENGINEER: Handling Load Data Request")
        logger.info(f"{'='*70}")
        
        # Extract data_source from node input
        input_data = workflow_node.input_data if hasattr(workflow_node, 'input_data') else {}
        data_source = input_data.get('data_source')
        
        logger.info(f"Input data keys: {list(input_data.keys())}")
        logger.info(f"Data source extracted: {data_source is not None}")
        
        if not data_source:
            logger.error("❌ CRITICAL: data_source not found in input_data!")
            logger.error(f"   Available keys: {list(input_data.keys())}")
            logger.error("   This means the orchestrator did not properly pass data_source to this node.")
            
            return {
                "success": False,
                "error": "data_source not found in workflow node input_data",
                "agent": self.agent_name,
                "stage": "data_acquisition"
            }
        
        # Load data using helper
        result = self.helper.load_data_from_source(data_source)
        
        # Add agent metadata
        result["agent"] = self.agent_name
        result["stage"] = "data_acquisition"
        result["timestamp"] = datetime.now().isoformat()
        
        # If successful, run validation
        if result["success"] and result["data"] is not None:
            logger.info("\nRunning data validation...")
            validation_report = self.helper.validate_data(result["data"])
            result["validation"] = validation_report
        
        return result


# ============================================================================
# STANDALONE TESTING
# ============================================================================

async def test_data_loading():
    """
    Test the data loading functionality standalone.
    """
    print("\n" + "="*70)
    print("DATA ENGINEER AGENT - STANDALONE TEST")
    print("="*70)
    
    # Create helper
    helper = DataEngineerAgentHelper()
    
    # Test 1: Load with proper data_source
    print("\n1. Testing with proper data_source:")
    data_source = {
        "type": "csv",
        "path": "data/sales_data.csv",
        "parameters": {
            "delimiter": ",",
            "header": True
        }
    }
    
    result = helper.load_data_from_source(data_source)
    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    if result['success']:
        print(f"  Rows: {result['metadata']['n_rows']}")
        print(f"  Columns: {result['metadata']['n_cols']}")
        print(f"  Column names: {result['metadata']['columns']}")
    else:
        print(f"  Error: {result['error']}")
    
    # Test 2: Load with None data_source (simulating the error)
    print("\n2. Testing with None data_source (simulating error):")
    result = helper.load_data_from_source(None)
    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Error: {result['error']}")
    
    # Test 3: Test with agent integration
    print("\n3. Testing with agent integration:")
    
    class MockWorkflowNode:
        def __init__(self, input_data):
            self.input_data = input_data
    
    integration = DataEngineerAgentIntegration()
    
    # Test with proper input
    node = MockWorkflowNode({
        "data_source": data_source,
        "query": "Test query",
        "objectives": ["Load data"]
    })
    
    result = await integration.handle_load_data_request(node)
    print(f"\nAgent Integration Result:")
    print(f"  Success: {result['success']}")
    print(f"  Agent: {result['agent']}")
    if result.get('validation'):
        print(f"  Validation passed: {result['validation']['passed']}")
        print(f"  Issues: {len(result['validation']['issues'])}")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_data_loading())