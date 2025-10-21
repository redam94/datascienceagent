"""
Complete Working Example: Data Passing Between Agents - FIXED

This example demonstrates a COMPLETE, WORKING implementation of the
data passing fixes. Copy this pattern to fix your workflow.

File: working_example_data_passing.py
Author: Data Science Agent System
Created: 2025-10-20

WHAT THIS FIXES:
- DATA_ACQUISITION agent can now find data_source
- Subsequent agents can access the loaded data
- Full traceability of data flow through workflow
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Import the fix components
from datascienceagent.core.fix_data_passing import (
    EnhancedWorkflowExecutor,
    StandardizedAgentOutput,
    DataAccessHelper,
    WorkflowDataValidator
)

from datascienceagent.agents.output_aware_specialist_agents import (
    OutputAwareAgentFactory
)

# ============================================================================
# MOCK WORKFLOW COMPONENTS (simplified for demonstration)
# ============================================================================


class WorkflowStage:
    """Simplified workflow stages"""
    DATA_ACQUISITION = "data_acquisition"
    EDA = "eda"
    MODELING = "modeling"


class WorkflowNode:
    """Simplified workflow node"""
    def __init__(self, id, stage, needs_initial_data=False, initial_data_keys=None, depends_on=None):
        self.id = id
        self.stage = type('Stage', (), {'value': stage})()
        self.needs_initial_data = needs_initial_data
        self.initial_data_keys = initial_data_keys or []
        self.depends_on = depends_on or []
        self.input_data = {}
        self.output_data = None
        self.status = "pending"


class WorkflowGraph:
    """Simplified workflow graph"""
    def __init__(self, name, initial_data=None):
        self.name = name
        self.initial_data = initial_data or {}
        self.nodes = {}


# ============================================================================
# MOCK AGENTS (showing correct pattern)
# ============================================================================


class DataEngineerAgent:
    """
    Data Engineer Agent - FIXED VERSION
    
    Shows correct pattern for:
    - Accessing data_source from input_data
    - Loading data
    - Returning standardized output
    """
    
    async def execute(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute data acquisition with proper data handling"""
        
        logger.info("\n" + "="*70)
        logger.info("DATA ENGINEER AGENT - Executing")
        logger.info("="*70)
        
        # STEP 1: Use helper to safely get data_source
        data_source = DataAccessHelper.get_data_source(node.input_data)
        
        if not data_source:
            logger.error("❌ FAILED: No data_source in input!")
            return StandardizedAgentOutput.create_error_output(
                stage="data_acquisition",
                error_message="data_source not provided in node input"
            )
        
        logger.info(f"✓ Found data_source: {data_source}")
        
        # STEP 2: Load the data (mocked here)
        logger.info(f"Loading data from: {data_source.get('path', 'unknown')}")
        
        # For demo, create synthetic data
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })
        
        logger.info(f"✓ Loaded data shape: {df.shape}")
        
        # STEP 3: Return standardized output
        output = StandardizedAgentOutput.create_data_acquisition_output(
            data=df,
            metadata={
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            },
            code="# Code that loaded the data..."
        )
        
        logger.info("✓ Data acquisition complete!")
        return output


class EDAAgent:
    """
    EDA Agent - FIXED VERSION
    
    Shows correct pattern for:
    - Accessing loaded data from previous stage
    - Processing the data
    - Returning standardized output
    """
    
    async def execute(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute EDA with proper data access"""
        
        logger.info("\n" + "="*70)
        logger.info("EDA AGENT - Executing")
        logger.info("="*70)
        
        # STEP 1: Use helper to safely get loaded data
        df = DataAccessHelper.get_loaded_data(node.input_data)
        
        if df is None:
            logger.error("❌ FAILED: No data available!")
            logger.error(f"   Input keys: {list(node.input_data.keys())}")
            return StandardizedAgentOutput.create_error_output(
                stage="eda",
                error_message="Data not available from DATA_ACQUISITION stage"
            )
        
        logger.info(f"✓ Got data with shape: {df.shape}")
        
        # STEP 2: Perform EDA
        eda_results = {
            "summary_stats": df.describe().to_dict(),
            "correlations": df.corr().to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        logger.info(f"✓ EDA complete: {len(eda_results)} analyses performed")
        
        # STEP 3: Return results in standard format
        return {
            "success": True,
            "stage": "eda",
            "timestamp": datetime.now().isoformat(),
            "eda_results": eda_results,
            "data": df,  # Pass data forward
            "data_available": True
        }


class ModelingAgent:
    """
    Modeling Agent - FIXED VERSION
    
    Shows correct pattern for using data from multiple previous stages.
    """
    
    async def execute(self, node: WorkflowNode) -> Dict[str, Any]:
        """Execute modeling with proper data access"""
        
        logger.info("\n" + "="*70)
        logger.info("MODELING AGENT - Executing")
        logger.info("="*70)
        
        # STEP 1: Get data (prefer from latest processing stage)
        df = DataAccessHelper.get_loaded_data(node.input_data)
        
        if df is None:
            logger.error("❌ FAILED: No data available!")
            return StandardizedAgentOutput.create_error_output(
                stage="modeling",
                error_message="Data not available"
            )
        
        logger.info(f"✓ Got data with shape: {df.shape}")
        
        # STEP 2: Get EDA results for context
        eda_results = None
        if "eda" in node.input_data:
            eda_results = node.input_data["eda"].get("eda_results")
            logger.info(f"✓ Got EDA results: {list(eda_results.keys()) if eda_results else 'None'}")
        
        # STEP 3: Build model (mocked)
        from sklearn.linear_model import LinearRegression
        
        X = df[['feature1', 'feature2']]
        y = df['target']
        
        model = LinearRegression()
        model.fit(X, y)
        
        score = model.score(X, y)
        logger.info(f"✓ Model trained. R² score: {score:.4f}")
        
        # STEP 4: Return standardized output
        return StandardizedAgentOutput.create_modeling_output(
            model=model,
            predictions=model.predict(X),
            metrics={"r2_score": score},
            code="# Model training code..."
        )


# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================


async def run_fixed_workflow():
    """
    Run a complete workflow with fixed data passing.
    
    This demonstrates the CORRECT way to ensure data flows
    between agents.
    """
    
    print("\n" + "="*80)
    print("COMPLETE WORKFLOW EXAMPLE - FIXED DATA PASSING")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Create initial_data (CRITICAL!)
    # ========================================================================
    print("\n1. Creating initial data...")
    
    initial_data = {
        "query": "Analyze feature relationships and build predictive model",
        "data_source": {
            "type": "csv",
            "path": "data/sales_data.csv",
            "parameters": {"delimiter": ","}
        },
        "objectives": [
            "Understand data structure",
            "Build regression model",
            "Evaluate performance"
        ]
    }
    
    print(f"   ✓ Initial data created with keys: {list(initial_data.keys())}")
    
    # ========================================================================
    # STEP 2: Create workflow WITH initial_data (CRITICAL!)
    # ========================================================================
    print("\n2. Creating workflow...")
    
    workflow = WorkflowGraph(
        name="analysis_workflow",
        initial_data=initial_data  # CRITICAL: Pass initial_data
    )
    
    # Create nodes with proper configuration
    node1 = WorkflowNode(
        id="node1",
        stage=WorkflowStage.DATA_ACQUISITION,
        needs_initial_data=True,  # CRITICAL!
        initial_data_keys=["data_source", "query"]  # CRITICAL!
    )
    
    node2 = WorkflowNode(
        id="node2",
        stage=WorkflowStage.EDA,
        depends_on=["node1"]
    )
    
    node3 = WorkflowNode(
        id="node3",
        stage=WorkflowStage.MODELING,
        depends_on=["node1", "node2"]
    )
    
    workflow.nodes = {
        "node1": node1,
        "node2": node2,
        "node3": node3
    }
    
    print(f"   ✓ Workflow created with {len(workflow.nodes)} nodes")
    
    # ========================================================================
    # STEP 3: Validate workflow configuration
    # ========================================================================
    print("\n3. Validating workflow configuration...")
    
    issues = WorkflowDataValidator.validate_workflow(workflow)
    if issues:
        print("   ⚠️ Configuration issues found:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print("   ✓ Workflow configuration valid")
    
    # ========================================================================
    # STEP 4: Create enhanced executor
    # ========================================================================
    print("\n4. Creating enhanced executor...")
    
    executor = EnhancedWorkflowExecutor(workflow)
    print("   ✓ Executor created")
    
    # ========================================================================
    # STEP 5: Create agents
    # ========================================================================
    print("\n5. Creating agents...")
    
    agents = {
        WorkflowStage.DATA_ACQUISITION: DataEngineerAgent(),
        WorkflowStage.EDA: EDAAgent(),
        WorkflowStage.MODELING: ModelingAgent()
    }
    
    print(f"   ✓ Created {len(agents)} agents")
    
    # ========================================================================
    # STEP 6: Execute workflow
    # ========================================================================
    print("\n6. Executing workflow nodes...")
    print("="*80)
    
    execution_order = ["node1", "node2", "node3"]
    
    for node_id in execution_order:
        node = workflow.nodes[node_id]
        
        # CRITICAL: Use enhanced executor to prepare input
        node.input_data = executor.prepare_node_input(node_id)
        
        # Execute with appropriate agent
        agent = agents[node.stage.value]
        output = await agent.execute(node)
        
        # CRITICAL: Store output for downstream nodes
        node.output_data = output
        
        # Validate output
        if executor.validate_node_output(node, output):
            node.status = "completed"
        else:
            node.status = "failed"
            break
    
    # ========================================================================
    # STEP 7: Generate data flow report
    # ========================================================================
    print("\n" + "="*80)
    print("7. Data Flow Report")
    print("="*80)
    
    report = executor.get_data_flow_report()
    print(report)
    
    # Save report
    report_path = Path("results/data_flow_report.txt")
    executor.save_data_flow_report(report_path)
    
    # ========================================================================
    # STEP 8: Summary
    # ========================================================================
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    for node_id, node in workflow.nodes.items():
        status_icon = "✅" if node.status == "completed" else "❌"
        print(f"{status_icon} {node.stage.value}: {node.status}")
        
        if node.output_data and node.output_data.get("success"):
            output_keys = list(node.output_data.keys())
            print(f"   Output keys: {output_keys}")
    
    print("\n✅ Workflow execution complete!")
    print(f"📄 Full data flow report saved to: {report_path}")


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║   WORKING EXAMPLE: FIXED DATA PASSING BETWEEN AGENTS                  ║
║                                                                        ║
║   This example demonstrates the complete solution for ensuring         ║
║   data (especially data_source) passes correctly between agents.      ║
║                                                                        ║
║   Key Components:                                                      ║
║   1. Initial data passed to workflow                                   ║
║   2. Nodes configured with needs_initial_data flags                    ║
║   3. EnhancedWorkflowExecutor for data preparation                     ║
║   4. Standardized output formats                                       ║
║   5. Data access helpers                                               ║
║   6. Validation and diagnostics                                        ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(run_fixed_workflow())