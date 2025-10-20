"""
Data Passing Fix for Agent Workflow System

This module provides comprehensive fixes and enhancements to ensure
data (especially data_source) is properly passed between agents in the workflow.

KEY ISSUES ADDRESSED:
1. Initial data (data_source) not reaching DATA_ACQUISITION node
2. Loaded data not accessible to subsequent agents
3. Missing configuration of needs_initial_data flags
4. Output data not properly stored in node.output_data

SOLUTION COMPONENTS:
1. Enhanced WorkflowNode with data tracking
2. Improved prepare_node_input logic
3. Data passing validation utilities
4. Agent output standardization
5. Complete integration example

Author: Data Science Agent System
Created: 2025-10-20
"""

from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
from dataclasses import dataclass, field

from loguru import logger


# ============================================================================
# PART 1: ENHANCED DATA STRUCTURES
# ============================================================================


@dataclass
class DataPassingMetadata:
    """
    Metadata about how data is passed to a node.
    
    This helps debug data passing issues by tracking:
    - What data the node received
    - Where it came from (dependencies vs initial_data)
    - What data the node produced
    - What data downstream nodes can access
    """
    node_id: str
    stage_name: str
    
    # Input tracking
    received_from_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    received_from_initial_data: List[str] = field(default_factory=list)
    total_input_keys: List[str] = field(default_factory=list)
    
    # Output tracking
    produced_output_keys: List[str] = field(default_factory=list)
    output_data_size_bytes: int = 0
    
    # Validation
    missing_required_keys: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "node_id": self.node_id,
            "stage_name": self.stage_name,
            "received_from_dependencies": self.received_from_dependencies,
            "received_from_initial_data": self.received_from_initial_data,
            "total_input_keys": self.total_input_keys,
            "produced_output_keys": self.produced_output_keys,
            "output_data_size_bytes": self.output_data_size_bytes,
            "missing_required_keys": self.missing_required_keys,
            "warnings": self.warnings
        }


# ============================================================================
# PART 2: ENHANCED WORKFLOW EXECUTOR WITH DATA VALIDATION
# ============================================================================


class EnhancedWorkflowExecutor:
    """
    Enhanced executor that validates and tracks data passing between nodes.
    
    FIXES:
    - Validates that nodes receive required data
    - Tracks data flow through the workflow
    - Provides detailed diagnostics when data is missing
    - Ensures output_data is properly structured
    """
    
    def __init__(self, workflow):
        """
        Initialize enhanced executor.
        
        Args:
            workflow: WorkflowGraph instance
        """
        self.workflow = workflow
        self.data_passing_log: List[DataPassingMetadata] = []
        logger.info(f"EnhancedWorkflowExecutor initialized for workflow: {workflow.name}")
    
    def prepare_node_input(
        self,
        node_id: str,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare input data for a node with validation and tracking.
        
        This is the CRITICAL method that combines:
        1. Outputs from dependency nodes
        2. Initial request data (query, data_source, objectives, etc.)
        
        Args:
            node_id: ID of node to prepare input for
            validate: Whether to validate required data is present
            
        Returns:
            Dictionary with all input data for the node
        """
        node = self.workflow.nodes[node_id]
        input_data = {}
        
        # Track what data we're collecting
        metadata = DataPassingMetadata(
            node_id=node_id,
            stage_name=node.stage.value
        )
        
        # ====================================================================
        # STEP 1: Collect outputs from dependency nodes
        # ====================================================================
        logger.info(f"📥 Preparing input for {node.stage.value}")
        logger.info(f"   Dependencies: {len(node.depends_on)} nodes")
        
        for dep_id in node.depends_on:
            dep_node = self.workflow.nodes[dep_id]
            
            if dep_node.output_data:
                # Store dependency output with stage name as key
                stage_key = dep_node.stage.value
                input_data[stage_key] = dep_node.output_data
                
                # Track what we received
                output_keys = list(dep_node.output_data.keys()) if isinstance(dep_node.output_data, dict) else ["data"]
                metadata.received_from_dependencies[stage_key] = output_keys
                
                logger.info(f"   ✓ From {stage_key}: {output_keys}")
            else:
                logger.warning(f"   ⚠️ Dependency {dep_node.stage.value} has no output_data!")
                metadata.warnings.append(f"Dependency {dep_node.stage.value} produced no output")
        
        # ====================================================================
        # STEP 2: Add initial request data (CRITICAL FIX!)
        # ====================================================================
        if node.needs_initial_data:
            logger.info(f"   Node needs initial data: {node.initial_data_keys or 'ALL'}")
            
            if node.initial_data_keys:
                # Only include specific keys
                for key in node.initial_data_keys:
                    if key in self.workflow.initial_data:
                        input_data[key] = self.workflow.initial_data[key]
                        metadata.received_from_initial_data.append(key)
                        logger.info(f"   ✓ Added initial data key: {key}")
                    else:
                        logger.warning(f"   ⚠️ Requested initial data key '{key}' not found!")
                        metadata.missing_required_keys.append(key)
            else:
                # Include all initial data
                input_data.update(self.workflow.initial_data)
                metadata.received_from_initial_data = list(self.workflow.initial_data.keys())
                logger.info(f"   ✓ Added all initial data: {metadata.received_from_initial_data}")
        
        # ====================================================================
        # STEP 3: Validate and track
        # ====================================================================
        metadata.total_input_keys = list(input_data.keys())
        
        logger.info(f"   📊 Total input keys: {metadata.total_input_keys}")
        
        if validate:
            self._validate_node_input(node, input_data, metadata)
        
        # Store metadata for debugging
        self.data_passing_log.append(metadata)
        
        return input_data
    
    def _validate_node_input(
        self,
        node,
        input_data: Dict[str, Any],
        metadata: DataPassingMetadata
    ):
        """
        Validate that node has required input data.
        
        Args:
            node: WorkflowNode instance
            input_data: Prepared input data
            metadata: Tracking metadata
        """
        # Check for critical requirements based on stage
        required_keys = self._get_required_keys_for_stage(node.stage.value)
        
        for key in required_keys:
            if key not in input_data:
                warning = f"Node {node.stage.value} missing recommended key: {key}"
                logger.warning(f"   ⚠️ {warning}")
                metadata.warnings.append(warning)
    
    def _get_required_keys_for_stage(self, stage_name: str) -> List[str]:
        """
        Get list of keys that a stage typically needs.
        
        This is a heuristic - stages can work without these but
        warnings help identify configuration issues.
        """
        requirements = {
            "data_acquisition": ["data_source"],
            "data_cleaning": ["data_acquisition"],
            "eda": ["data_acquisition"],
            "feature_engineering": ["data_acquisition", "data_cleaning"],
            "modeling": ["data_acquisition", "feature_engineering"],
            "model_evaluation": ["modeling"],
            "interpretation": ["modeling", "model_evaluation"],
            "report_generation": ["interpretation"]
        }
        return requirements.get(stage_name.lower(), [])
    
    def validate_node_output(
        self,
        node,
        output_data: Dict[str, Any]
    ) -> bool:
        """
        Validate that node produced proper output.
        
        CRITICAL: Ensures output_data is in correct format for downstream nodes.
        
        Args:
            node: WorkflowNode instance
            output_data: Output data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not output_data:
            logger.error(f"❌ Node {node.stage.value} produced no output!")
            return False
        
        if not isinstance(output_data, dict):
            logger.error(f"❌ Node {node.stage.value} output is not a dictionary!")
            return False
        
        # Check for success indicator
        if "success" in output_data and not output_data["success"]:
            logger.warning(f"⚠️ Node {node.stage.value} reported failure in output")
            return False
        
        # Track what was produced
        for entry in self.data_passing_log:
            if entry.node_id == node.id:
                entry.produced_output_keys = list(output_data.keys())
                break
        
        logger.info(f"   ✅ Node output validated: {list(output_data.keys())}")
        return True
    
    def get_data_flow_report(self) -> str:
        """
        Generate a report showing how data flowed through the workflow.
        
        Returns:
            Formatted report string
        """
        report = [
            "=" * 70,
            "DATA FLOW REPORT",
            "=" * 70,
            ""
        ]
        
        for entry in self.data_passing_log:
            report.append(f"Node: {entry.stage_name}")
            report.append(f"  Node ID: {entry.node_id}")
            
            if entry.received_from_initial_data:
                report.append(f"  ✓ From initial data: {entry.received_from_initial_data}")
            
            if entry.received_from_dependencies:
                report.append(f"  ✓ From dependencies:")
                for stage, keys in entry.received_from_dependencies.items():
                    report.append(f"    - {stage}: {keys}")
            
            if entry.produced_output_keys:
                report.append(f"  ✓ Produced: {entry.produced_output_keys}")
            
            if entry.warnings:
                report.append(f"  ⚠️ Warnings:")
                for warning in entry.warnings:
                    report.append(f"    - {warning}")
            
            if entry.missing_required_keys:
                report.append(f"  ❌ Missing: {entry.missing_required_keys}")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_data_flow_report(self, output_path: Path):
        """Save data flow report to file"""
        report = self.get_data_flow_report()
        output_path.write_text(report)
        logger.info(f"📄 Data flow report saved to: {output_path}")


# ============================================================================
# PART 3: AGENT OUTPUT STANDARDIZATION
# ============================================================================


class StandardizedAgentOutput:
    """
    Standardized format for agent outputs to ensure consistency.
    
    FIXES: Ensures all agents return data in format that can be
    properly accessed by downstream agents.
    """
    
    @staticmethod
    def create_data_acquisition_output(
        data,
        metadata: Dict[str, Any],
        code: str = "",
        validation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized output for DATA_ACQUISITION stage.
        
        Args:
            data: Loaded DataFrame or data object
            metadata: Data metadata (shape, columns, types, etc.)
            code: Code used to load data
            validation: Optional validation report
            
        Returns:
            Standardized output dictionary
        """
        return {
            "success": True,
            "stage": "data_acquisition",
            "timestamp": datetime.now().isoformat(),
            
            # Critical: Data must be accessible to downstream nodes
            "data": data,
            "metadata": metadata,
            
            # Optional but helpful
            "code": code,
            "validation": validation or {},
            
            # For tracking
            "data_available": True,
            "data_type": type(data).__name__,
            "data_shape": getattr(data, 'shape', None)
        }
    
    @staticmethod
    def create_data_cleaning_output(
        cleaned_data,
        cleaning_report: Dict[str, Any],
        code: str = ""
    ) -> Dict[str, Any]:
        """Create standardized output for DATA_CLEANING stage"""
        return {
            "success": True,
            "stage": "data_cleaning",
            "timestamp": datetime.now().isoformat(),
            
            # Critical: Cleaned data for downstream
            "data": cleaned_data,
            "cleaning_report": cleaning_report,
            "code": code,
            
            # Tracking
            "data_available": True,
            "data_type": type(cleaned_data).__name__,
            "data_shape": getattr(cleaned_data, 'shape', None)
        }
    
    @staticmethod
    def create_modeling_output(
        model,
        predictions,
        metrics: Dict[str, float],
        code: str = ""
    ) -> Dict[str, Any]:
        """Create standardized output for MODELING stage"""
        return {
            "success": True,
            "stage": "modeling",
            "timestamp": datetime.now().isoformat(),
            
            # Model artifacts
            "model": model,
            "predictions": predictions,
            "metrics": metrics,
            "code": code,
            
            # Tracking
            "model_available": True,
            "model_type": type(model).__name__
        }
    
    @staticmethod
    def create_error_output(
        stage: str,
        error_message: str,
        attempted_code: str = ""
    ) -> Dict[str, Any]:
        """Create standardized error output"""
        return {
            "success": False,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "code": attempted_code,
            "data_available": False
        }


# ============================================================================
# PART 4: DATA ACCESS UTILITIES
# ============================================================================


class DataAccessHelper:
    """
    Helper methods to safely access data from node inputs.
    
    FIXES: Provides safe, documented ways for agents to access
    data from previous stages.
    """
    
    @staticmethod
    def get_loaded_data(node_input: Dict[str, Any]) -> Optional[Any]:
        """
        Safely extract loaded data from DATA_ACQUISITION stage.
        
        Args:
            node_input: Input dictionary passed to agent
            
        Returns:
            DataFrame or None if not found
        """
        # Try different possible locations
        locations = [
            ("data_acquisition", "data"),
            ("data", None),
            ("loaded_data", None)
        ]
        
        for stage_key, data_key in locations:
            if stage_key in node_input:
                stage_output = node_input[stage_key]
                if isinstance(stage_output, dict):
                    if data_key and data_key in stage_output:
                        return stage_output[data_key]
                    elif "data" in stage_output:
                        return stage_output["data"]
                else:
                    return stage_output
        
        logger.error("❌ Could not find loaded data in node input!")
        logger.error(f"   Available keys: {list(node_input.keys())}")
        return None
    
    @staticmethod
    def get_data_source(node_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Safely extract data_source specification.
        
        Args:
            node_input: Input dictionary passed to agent
            
        Returns:
            data_source dict or None
        """
        if "data_source" in node_input:
            return node_input["data_source"]
        
        logger.error("❌ data_source not found in node input!")
        logger.error(f"   Available keys: {list(node_input.keys())}")
        logger.error("   This usually means:")
        logger.error("   1. Node is missing needs_initial_data=True")
        logger.error("   2. Node is missing initial_data_keys=['data_source']")
        logger.error("   3. Orchestrator didn't pass data_source to workflow")
        return None
    
    @staticmethod
    def get_query(node_input: Dict[str, Any]) -> Optional[str]:
        """Safely extract user query"""
        return node_input.get("query")
    
    @staticmethod
    def get_objectives(node_input: Dict[str, Any]) -> List[str]:
        """Safely extract analysis objectives"""
        return node_input.get("objectives", [])


# ============================================================================
# PART 5: VALIDATION AND DIAGNOSTICS
# ============================================================================


class WorkflowDataValidator:
    """
    Validates workflow configuration for proper data passing.
    
    Use this to check if a workflow is correctly configured
    BEFORE execution.
    """
    
    @staticmethod
    def validate_workflow(workflow) -> List[str]:
        """
        Validate workflow configuration.
        
        Returns:
            List of issues found (empty if valid)
        """
        issues = []
        
        # Check initial_data is set
        if not workflow.initial_data:
            issues.append("WARNING: workflow.initial_data is empty")
        
        # Check critical nodes have proper configuration
        for node in workflow.nodes.values():
            if node.stage.value == "data_acquisition":
                if not node.needs_initial_data:
                    issues.append(
                        f"CRITICAL: {node.stage.value} node missing needs_initial_data=True"
                    )
                if "data_source" not in (node.initial_data_keys or []):
                    issues.append(
                        f"CRITICAL: {node.stage.value} node missing 'data_source' in initial_data_keys"
                    )
        
        return issues
    
    @staticmethod
    def diagnose_data_passing_failure(
        node,
        node_input: Dict[str, Any],
        expected_data: List[str]
    ) -> str:
        """
        Diagnose why a node isn't receiving expected data.
        
        Args:
            node: WorkflowNode that has data issues
            node_input: The input dict the node received
            expected_data: List of keys the node expected
            
        Returns:
            Diagnostic message
        """
        diagnosis = [
            f"\n{'='*70}",
            f"DATA PASSING DIAGNOSTIC for {node.stage.value}",
            f"{'='*70}",
            f"Node ID: {node.id}",
            ""
        ]
        
        # Check what node received
        diagnosis.append(f"Received keys: {list(node_input.keys())}")
        diagnosis.append(f"Expected keys: {expected_data}")
        missing = set(expected_data) - set(node_input.keys())
        diagnosis.append(f"Missing keys: {missing}")
        diagnosis.append("")
        
        # Check node configuration
        diagnosis.append("Node configuration:")
        diagnosis.append(f"  needs_initial_data: {node.needs_initial_data}")
        diagnosis.append(f"  initial_data_keys: {node.initial_data_keys}")
        diagnosis.append(f"  depends_on: {node.depends_on}")
        diagnosis.append("")
        
        # Suggest fixes
        diagnosis.append("Suggested fixes:")
        
        for key in missing:
            if key == "data_source":
                diagnosis.append("  ❌ Missing 'data_source':")
                diagnosis.append("     1. Ensure node has needs_initial_data=True")
                diagnosis.append("     2. Ensure node has initial_data_keys=['data_source', ...]")
                diagnosis.append("     3. Ensure orchestrator passes data_source in initial_data")
            
            elif key in ["data", "loaded_data"]:
                diagnosis.append(f"  ❌ Missing '{key}':")
                diagnosis.append("     1. Check DATA_ACQUISITION node completed successfully")
                diagnosis.append("     2. Check DATA_ACQUISITION node.output_data contains 'data'")
                diagnosis.append("     3. Ensure this node depends on DATA_ACQUISITION")
        
        return "\n".join(diagnosis)


# ============================================================================
# PART 6: INTEGRATION EXAMPLE
# ============================================================================


def demonstrate_fix():
    """
    Demonstrate how the fixes work together.
    
    This shows the complete pattern for ensuring data passes correctly.
    """
    print("\n" + "="*70)
    print("DATA PASSING FIX DEMONSTRATION")
    print("="*70)
    
    print("\n1. CRITICAL FIX: Ensure initial_data passed to workflow")
    print("-" * 70)
    print("""
# In your orchestrator or main script:

initial_data = {
    "query": "Analyze sales trends",
    "data_source": {
        "type": "csv",
        "path": "data/sales_data.csv",
        "parameters": {"delimiter": ","}
    },
    "objectives": ["Find trends", "Build model"]
}

# CRITICAL: Pass initial_data when creating workflow
workflow = create_standard_regression_workflow(initial_data=initial_data)
    """)
    
    print("\n2. CRITICAL FIX: Configure DATA_ACQUISITION node")
    print("-" * 70)
    print("""
# In your workflow builder:

data_acquisition_node = self.add_node(
    stage=WorkflowStage.DATA_ACQUISITION,
    description="Load data from source",
    agent_name="data_engineer_agent",
    needs_initial_data=True,  # CRITICAL!
    initial_data_keys=["data_source", "query"]  # CRITICAL!
)
    """)
    
    print("\n3. CRITICAL FIX: Use EnhancedWorkflowExecutor")
    print("-" * 70)
    print("""
# In your orchestrator:

executor = EnhancedWorkflowExecutor(workflow)

# This will properly combine initial_data and dependency outputs
input_data = executor.prepare_node_input(node_id)

# Now input_data contains BOTH:
# - data_source (from initial_data)
# - outputs from dependency nodes
    """)
    
    print("\n4. CRITICAL FIX: Agents must use standardized output")
    print("-" * 70)
    print("""
# In your DATA_ACQUISITION agent:

from fix_data_passing import StandardizedAgentOutput, DataAccessHelper

# Load the data
data_source = DataAccessHelper.get_data_source(node.input_data)
df = load_csv(data_source["path"])

# Return in standard format
output = StandardizedAgentOutput.create_data_acquisition_output(
    data=df,
    metadata={"shape": df.shape, "columns": list(df.columns)},
    code=code_used
)

# CRITICAL: Set as node output
node.output_data = output
return output
    """)
    
    print("\n5. CRITICAL FIX: Downstream agents access data correctly")
    print("-" * 70)
    print("""
# In your downstream agents (EDA, MODELING, etc.):

from fix_data_passing import DataAccessHelper

# Safely get the data
df = DataAccessHelper.get_loaded_data(node.input_data)

if df is None:
    # Data not found - error handling
    logger.error("Cannot access data from DATA_ACQUISITION")
    return StandardizedAgentOutput.create_error_output(
        stage="eda",
        error_message="Data not available"
    )

# Now work with the data
...
    """)
    
    print("\n6. VALIDATION: Check workflow configuration")
    print("-" * 70)
    print("""
# Before running workflow:

from fix_data_passing import WorkflowDataValidator

issues = WorkflowDataValidator.validate_workflow(workflow)
if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
    """)
    
    print("\n7. DIAGNOSTICS: Debug data passing issues")
    print("-" * 70)
    print("""
# After execution, get detailed report:

report = executor.get_data_flow_report()
print(report)

# Or save to file:
executor.save_data_flow_report(Path("data_flow_report.txt"))
    """)
    
    print("\n" + "="*70)
    print("SUMMARY OF FIXES")
    print("="*70)
    print("""
To fix data passing between agents:

1. ✅ Pass initial_data to workflow creation
2. ✅ Set needs_initial_data=True on nodes that need it
3. ✅ Specify initial_data_keys for those nodes
4. ✅ Use EnhancedWorkflowExecutor.prepare_node_input()
5. ✅ Agents return standardized output format
6. ✅ Agents use DataAccessHelper to get data
7. ✅ Validate workflow config before execution
8. ✅ Use data flow reports for debugging

Key insight: Data must flow through BOTH channels:
- Initial data (query, data_source) → nodes that need it
- Output data (results) → downstream dependencies
    """)


if __name__ == "__main__":
    demonstrate_fix()