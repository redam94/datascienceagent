"""
Workflow Graph System for Data Science Agents - REFACTORED

Key improvements:
1. WorkflowGraph now stores initial_data from the analysis request
2. Nodes can access both dependency outputs AND initial request data
3. WorkflowBuilder accepts and propagates initial data to relevant nodes
"""

import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque

from pydantic import BaseModel, Field


# ============================================================================
# CORE MODELS
# ============================================================================

class WorkflowStage(str, Enum):
    """Standard stages in a data science workflow"""
    # Planning phase
    STATISTICAL_PLANNING = "statistical_planning"
    RESEARCH = "research"
    
    # Data phase
    DATA_ACQUISITION = "data_acquisition"
    DATA_VALIDATION = "data_validation"
    DATA_CLEANING = "data_cleaning"
    
    # Analysis phase
    EDA = "eda"
    FEATURE_ENGINEERING = "feature_engineering"
    
    # Post-EDA data phase
    POST_EDA_PROCESSING = "post_eda_processing"
    
    # Modeling phase
    MODELING = "modeling"
    MODEL_SPECIFICATION = "model_specification"
    MODEL_FITTING = "model_fitting"
    MODEL_DIAGNOSTICS = "model_diagnostics"
    
    # Interpretation phase
    INTERPRETATION = "interpretation"
    REPORT_GENERATION = "report_generation"


class StageStatus(str, Enum):
    """Status of a workflow stage"""
    PENDING = "pending"
    READY = "ready"  # Dependencies met, ready to execute
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowNode(BaseModel):
    """A node in the workflow graph representing a single stage"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stage: WorkflowStage
    description: str
    status: StageStatus = StageStatus.PENDING
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)  # IDs of prerequisite nodes
    
    # Agent assignment
    agent_name: str
    
    # Execution metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    # Context specification
    required_context_types: List[str] = Field(default_factory=list)
    context_scope: str = "current_session"  # or "include_history"
    
    # Input/Output
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    
    # Flags for data access
    needs_initial_data: bool = False  # NEW: Flag to indicate node needs initial request data
    initial_data_keys: List[str] = Field(default_factory=list)  # NEW: Specific keys from initial data
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowGraph(BaseModel):
    """
    A directed acyclic graph representing a data science workflow.
    
    NEW: Now includes initial_data from the analysis request that can be accessed by any node.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    
    # Graph structure
    nodes: Dict[str, WorkflowNode] = Field(default_factory=dict)
    
    # NEW: Initial request data (query, data_source, objectives, etc.)
    initial_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    workflow_type: str = "standard"
    
    # Execution tracking
    current_stage: Optional[str] = None
    completed_stages: List[str] = Field(default_factory=list)
    failed_stages: List[str] = Field(default_factory=list)


# ============================================================================
# WORKFLOW BUILDER - REFACTORED
# ============================================================================

class WorkflowBuilder:
    """
    Fluent API for building standardized workflows.
    
    REFACTORED: Now accepts and propagates initial data to nodes that need it.
    """
    
    def __init__(self, name: str, description: str = "", initial_data: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.nodes: Dict[str, WorkflowNode] = {}
        self._last_phase_nodes: Dict[str, List[str]] = defaultdict(list)
        self.initial_data = initial_data or {}  # NEW: Store initial data
    
    def add_node(
        self,
        stage: WorkflowStage,
        description: str,
        agent_name: str,
        depends_on: Optional[List[str]] = None,
        required_context_types: Optional[List[str]] = None,
        context_scope: str = "current_session",
        metadata: Optional[Dict[str, Any]] = None,
        needs_initial_data: bool = False,  # NEW
        initial_data_keys: Optional[List[str]] = None  # NEW
    ) -> str:
        """Add a single node to the workflow graph"""
        node = WorkflowNode(
            stage=stage,
            description=description,
            agent_name=agent_name,
            depends_on=depends_on or [],
            required_context_types=required_context_types or [],
            context_scope=context_scope,
            metadata=metadata or {},
            needs_initial_data=needs_initial_data,
            initial_data_keys=initial_data_keys or []
        )
        self.nodes[node.id] = node
        return node.id
    
    def add_planning_phase(self) -> "WorkflowBuilder":
        """
        Add statistical planning phase.
        This should happen BEFORE modeling.
        """
        # Research appropriate methods - needs query from initial data
        research_id = self.add_node(
            stage=WorkflowStage.RESEARCH,
            description="Research appropriate statistical methods and best practices",
            agent_name="statistician",
            required_context_types=["user_query", "past_analyses"],
            context_scope="include_history",
            needs_initial_data=True,  # NEW
            initial_data_keys=["query", "objectives"]  # NEW
        )
        
        # Plan statistical approach
        planning_id = self.add_node(
            stage=WorkflowStage.STATISTICAL_PLANNING,
            description="Plan statistical analysis approach and validation strategy",
            agent_name="statistician",
            depends_on=[research_id],
            required_context_types=["research_findings", "data_summary"],
            context_scope="current_session",
            needs_initial_data=True,  # NEW
            initial_data_keys=["query", "objectives", "data_source"]  # NEW
        )
        
        self._last_phase_nodes["planning"] = [research_id, planning_id]
        return self
    
    def add_data_phase(self, depends_on_planning: bool = True) -> "WorkflowBuilder":
        """
        Add data acquisition and cleaning phase.
        CRITICAL: DATA_ACQUISITION needs data_source from initial data!
        """
        deps = self._last_phase_nodes["planning"] if depends_on_planning else []
        
        # Acquire data - CRITICAL: This node MUST have access to data_source
        acquisition_id = self.add_node(
            stage=WorkflowStage.DATA_ACQUISITION,
            description="Load and acquire data from specified sources",
            agent_name="data_engineer",
            depends_on=deps,
            required_context_types=["data_source_spec", "loading_requirements"],
            context_scope="current_session",
            needs_initial_data=True,  # NEW: CRITICAL!
            initial_data_keys=["data_source", "query"]  # NEW: CRITICAL!
        )
        
        # Validate data
        validation_id = self.add_node(
            stage=WorkflowStage.DATA_VALIDATION,
            description="Validate data quality and schema",
            agent_name="data_engineer",
            depends_on=[acquisition_id],
            required_context_types=["data_quality_checks", "expected_schema"],
            context_scope="current_session"
        )
        
        # Clean data
        cleaning_id = self.add_node(
            stage=WorkflowStage.DATA_CLEANING,
            description="Clean and preprocess data",
            agent_name="data_engineer",
            depends_on=[validation_id],
            required_context_types=["validation_report", "cleaning_strategies"],
            context_scope="include_history"
        )
        
        self._last_phase_nodes["data"] = [acquisition_id, validation_id, cleaning_id]
        return self
    
    def add_eda_phase(self) -> "WorkflowBuilder":
        """
        Add exploratory data analysis phase.
        This should happen AFTER data cleaning and BEFORE final modeling.
        """
        deps = self._last_phase_nodes["data"]
        
        eda_id = self.add_node(
            stage=WorkflowStage.EDA,
            description="Perform exploratory data analysis and visualization",
            agent_name="eda",
            depends_on=deps,
            required_context_types=["clean_data", "analysis_objectives", "visualization_preferences"],
            context_scope="current_session",
            needs_initial_data=True,  # NEW
            initial_data_keys=["objectives"]  # NEW
        )
        
        # Feature engineering based on EDA insights
        feature_eng_id = self.add_node(
            stage=WorkflowStage.FEATURE_ENGINEERING,
            description="Engineer features based on EDA insights",
            agent_name="data_engineer",
            depends_on=[eda_id],
            required_context_types=["eda_insights", "feature_suggestions", "domain_knowledge"],
            context_scope="current_session"
        )
        
        # Post-EDA processing to exploit EDA findings
        post_eda_id = self.add_node(
            stage=WorkflowStage.POST_EDA_PROCESSING,
            description="Additional data processing based on EDA findings",
            agent_name="data_engineer",
            depends_on=[feature_eng_id],
            required_context_types=["eda_findings", "transformation_needs"],
            context_scope="current_session"
        )
        
        self._last_phase_nodes["eda"] = [eda_id, feature_eng_id, post_eda_id]
        return self
    
    def add_modeling_phase(self) -> "WorkflowBuilder":
        """
        Add modeling phase.
        This should happen AFTER statistical planning and data preparation.
        """
        # Depends on both planning and EDA phases
        deps = (self._last_phase_nodes.get("planning", []) + 
                self._last_phase_nodes.get("eda", []))
        
        # Specify model
        spec_id = self.add_node(
            stage=WorkflowStage.MODEL_SPECIFICATION,
            description="Specify model structure and parameters",
            agent_name="modeling",
            depends_on=deps,
            required_context_types=["plan", "processed_data", "eda_insights"],
            context_scope="current_session"
        )
        
        # Fit model
        fit_id = self.add_node(
            stage=WorkflowStage.MODEL_FITTING,
            description="Fit statistical model and estimate parameters",
            agent_name="modeling",
            depends_on=[spec_id],
            required_context_types=["model_spec", "training_data", "code_chunk"],
            context_scope="include_history"
        )
        
        # Diagnostics
        diagnostics_id = self.add_node(
            stage=WorkflowStage.MODEL_DIAGNOSTICS,
            description="Run model diagnostics and validation checks",
            agent_name="modeling",
            depends_on=[fit_id],
            required_context_types=["fitted_model", "diagnostic_procedures"],
            context_scope="include_history"
        )
        
        self._last_phase_nodes["modeling"] = [spec_id, fit_id, diagnostics_id]
        return self
    
    def add_interpretation_phase(self) -> "WorkflowBuilder":
        """
        Add interpretation and reporting phase.
        This should happen AFTER modeling and diagnostics.
        """
        deps = self._last_phase_nodes["modeling"]
        
        # Interpret results
        interp_id = self.add_node(
            stage=WorkflowStage.INTERPRETATION,
            description="Interpret model results and assess practical significance",
            agent_name="interpreter",
            depends_on=deps,
            required_context_types=["model_result", "diagnostics", "business_context"],
            context_scope="current_session",
            needs_initial_data=True,  # NEW
            initial_data_keys=["query", "objectives"]  # NEW
        )
        
        # Generate report
        report_id = self.add_node(
            stage=WorkflowStage.REPORT_GENERATION,
            description="Generate comprehensive analysis report",
            agent_name="interpreter",
            depends_on=[interp_id],
            required_context_types=["interpretation", "all_outputs", "reporting_requirements"],
            context_scope="current_session",
            needs_initial_data=True,  # NEW
            initial_data_keys=["query", "objectives"]  # NEW
        )
        
        self._last_phase_nodes["interpretation"] = [interp_id, report_id]
        return self
    
    def build(self) -> WorkflowGraph:
        """Build the final workflow graph with initial data"""
        # Validate no cycles
        if self._has_cycle():
            raise ValueError("Workflow graph contains cycles")
        
        return WorkflowGraph(
            name=self.name,
            description=self.description,
            nodes=self.nodes,
            initial_data=self.initial_data  # NEW: Pass initial data to graph
        )
    
    def _has_cycle(self) -> bool:
        """Check if graph has cycles using DFS"""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for dep_id in self.nodes[node_id].depends_on:
                if dep_id not in visited:
                    if dfs(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        
        return False


# ============================================================================
# WORKFLOW EXECUTOR - REFACTORED
# ============================================================================

class WorkflowExecutor:
    """
    Executes workflows by processing nodes in topological order.
    
    REFACTORED: Now handles initial data propagation to nodes.
    """
    
    def __init__(self, workflow: WorkflowGraph):
        self.workflow = workflow
    
    def get_ready_nodes(self) -> List[WorkflowNode]:
        """Get all nodes that are ready to execute (dependencies met)"""
        ready = []
        
        for node in self.workflow.nodes.values():
            if node.status != StageStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_met = all(
                self.workflow.nodes[dep_id].status == StageStatus.COMPLETED
                for dep_id in node.depends_on
            )
            
            if deps_met:
                node.status = StageStatus.READY
                ready.append(node)
        
        return ready
    
    def prepare_node_input(self, node_id: str) -> Dict[str, Any]:
        """
        NEW/REFACTORED: Prepare input data for a node.
        
        Combines:
        1. Outputs from dependency nodes
        2. Initial request data (if node needs it)
        
        This is the KEY method that fixes the data passing issue!
        """
        node = self.workflow.nodes[node_id]
        input_data = {}
        
        # 1. Get outputs from dependency nodes
        for dep_id in node.depends_on:
            dep_node = self.workflow.nodes[dep_id]
            if dep_node.output_data:
                # Store with stage name as key
                input_data[dep_node.stage.value] = dep_node.output_data
        
        # 2. NEW: Add initial data if node needs it
        if node.needs_initial_data:
            if node.initial_data_keys:
                # Only include specific keys
                for key in node.initial_data_keys:
                    if key in self.workflow.initial_data:
                        input_data[key] = self.workflow.initial_data[key]
            else:
                # Include all initial data
                input_data.update(self.workflow.initial_data)
        
        return input_data
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get nodes in topological order, grouped by execution level.
        Nodes in the same level can be executed in parallel.
        """
        # Compute in-degree for each node
        in_degree = {node_id: len(node.depends_on) 
                     for node_id, node in self.workflow.nodes.items()}
        
        # Find nodes with no dependencies (level 0)
        queue = deque([node_id for node_id, degree in in_degree.items() 
                       if degree == 0])
        
        levels = []
        while queue:
            # All nodes in current level
            current_level = list(queue)
            levels.append(current_level)
            
            # Process current level
            next_queue = []
            for node_id in current_level:
                # Find nodes that depend on this one
                for other_id, other_node in self.workflow.nodes.items():
                    if node_id in other_node.depends_on:
                        in_degree[other_id] -= 1
                        if in_degree[other_id] == 0:
                            next_queue.append(other_id)
            
            queue = deque(next_queue)
        
        return levels
    
    def mark_completed(self, node_id: str, output_data: Dict[str, Any]):
        """Mark a node as completed"""
        node = self.workflow.nodes[node_id]
        node.status = StageStatus.COMPLETED
        node.completed_at = datetime.now()
        node.output_data = output_data
        self.workflow.completed_stages.append(node_id)
    
    def mark_failed(self, node_id: str, error: str):
        """Mark a node as failed"""
        node = self.workflow.nodes[node_id]
        node.status = StageStatus.FAILED
        node.error = error
        self.workflow.failed_stages.append(node_id)
    
    def can_skip_node(self, node_id: str) -> bool:
        """Check if a node can be skipped due to failed dependencies"""
        node = self.workflow.nodes[node_id]
        
        # Check if any dependencies failed
        for dep_id in node.depends_on:
            if self.workflow.nodes[dep_id].status == StageStatus.FAILED:
                return True
        
        return False
    
    def get_progress(self) -> Dict[str, Any]:
        """Get workflow execution progress"""
        total = len(self.workflow.nodes)
        completed = len([n for n in self.workflow.nodes.values() 
                        if n.status == StageStatus.COMPLETED])
        failed = len([n for n in self.workflow.nodes.values() 
                     if n.status == StageStatus.FAILED])
        in_progress = len([n for n in self.workflow.nodes.values() 
                          if n.status == StageStatus.IN_PROGRESS])
        
        return {
            "total_nodes": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": total - completed - failed - in_progress,
            "progress_pct": (completed / total * 100) if total > 0 else 0
        }


# ============================================================================
# STANDARD WORKFLOWS - REFACTORED
# ============================================================================

def create_standard_regression_workflow(initial_data: Optional[Dict[str, Any]] = None) -> WorkflowGraph:
    """
    Create standard workflow for regression analysis.
    
    NEW: Accepts initial_data to pass through to nodes.
    """
    return (WorkflowBuilder(
        name="standard_regression",
        description="Complete regression analysis workflow",
        initial_data=initial_data  # NEW!
    )
    .add_planning_phase()
    .add_data_phase()
    .add_eda_phase()
    .add_modeling_phase()
    .add_interpretation_phase()
    .build())


def create_exploratory_workflow(initial_data: Optional[Dict[str, Any]] = None) -> WorkflowGraph:
    """Create workflow focused on exploration (no modeling)"""
    return (WorkflowBuilder(
        name="exploratory_analysis",
        description="Exploratory analysis without modeling",
        initial_data=initial_data  # NEW!
    )
    .add_data_phase(depends_on_planning=False)
    .add_eda_phase()
    .add_interpretation_phase()
    .build())


def create_modeling_focused_workflow(initial_data: Optional[Dict[str, Any]] = None) -> WorkflowGraph:
    """Create workflow assuming data is already prepared"""
    return (WorkflowBuilder(
        name="modeling_focused",
        description="Modeling workflow with pre-prepared data",
        initial_data=initial_data  # NEW!
    )
    .add_planning_phase()
    .add_modeling_phase()
    .add_interpretation_phase()
    .build())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create a standard regression workflow WITH initial data
    initial_data = {
        "query": "Analyze sales data",
        "data_source": {
            "type": "csv",
            "path": "/data/sales.csv",
            "parameters": {"delimiter": ","}
        },
        "objectives": [
            "Identify key drivers",
            "Build predictive model"
        ]
    }
    
    workflow = create_standard_regression_workflow(initial_data=initial_data)
    
    print(f"Created workflow: {workflow.name}")
    print(f"Total nodes: {len(workflow.nodes)}")
    print(f"Initial data keys: {list(workflow.initial_data.keys())}")
    
    # Find the data acquisition node and check it has access to data_source
    for node in workflow.nodes.values():
        if node.stage == WorkflowStage.DATA_ACQUISITION:
            print(f"\nData Acquisition Node:")
            print(f"  needs_initial_data: {node.needs_initial_data}")
            print(f"  initial_data_keys: {node.initial_data_keys}")
    
    # Get execution order
    executor = WorkflowExecutor(workflow)
    levels = executor.get_execution_order()
    
    print("\nExecution order (by level):")
    for i, level in enumerate(levels):
        print(f"  Level {i}: {len(level)} nodes")
        for node_id in level:
            node = workflow.nodes[node_id]
            # Prepare input to see what data node will receive
            input_data = executor.prepare_node_input(node_id)
            print(f"    - {node.stage.value}")
            if input_data:
                print(f"      Input keys: {list(input_data.keys())}")