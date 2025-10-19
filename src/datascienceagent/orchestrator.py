"""
Orchestrator for Data Science Workflow Execution

Coordinates specialist agents using standardized workflow graphs.
Ensures proper dependency management and context flow between agents.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from pydantic import BaseModel, Field
from loguru import logger

from datascienceagent.core.workflow_graph import (
    WorkflowGraph, WorkflowExecutor, WorkflowStage,
    StageStatus, create_standard_regression_workflow
)
from datascienceagent.agents.specialist_agents import AgentFactory
from datascienceagent.utils.model_utils import process_model

# ============================================================================
# ORCHESTRATOR
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request for analysis"""
    query: str
    workflow_type: str = "standard_regression"  # or "exploratory", "modeling_focused"
    data_source: Optional[Dict[str, Any]] = None
    objectives: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Complete analysis result"""
    workflow_id: str
    query: str
    status: str  # "completed", "failed", "partial"
    
    # Results by stage
    stage_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Final outputs
    interpretation: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    
    # Execution metadata
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_seconds: float = 0
    
    # Errors
    errors: List[str] = Field(default_factory=list)


class WorkflowOrchestrator:
    """
    Orchestrates complete data science workflows.
    
    Responsibilities:
    - Create appropriate workflow graphs
    - Initialize specialist agents
    - Execute workflows in proper order
    - Manage context flow between agents
    - Handle errors and recovery
    - Aggregate results
    """
    
    def __init__(
        self,
        context_manager: Optional[Any] = None,
        model: str = "openai:gpt-4"
    ):
        self.context_manager = context_manager
        self.model = process_model(model)
        self.agents: Dict[str, Any] = {}
        
        # Initialize specialist agents
        self._initialize_agents()
        
        # Track active workflows
        self.active_workflows: Dict[str, WorkflowGraph] = {}
        self.workflow_executors: Dict[str, WorkflowExecutor] = {}
    
    def _initialize_agents(self):
        """Initialize all specialist agents"""
        agent_types = [
            "statistician",
            "data_engineer",
            "eda",
            "modeling",
            "interpreter"
        ]
        
        for agent_type in agent_types:
            self.agents[f"{agent_type}_agent"] = AgentFactory.create_agent(
                agent_type,
                model=self.model,
                context_manager=self.context_manager
            )
    
    async def execute_analysis(
        self,
        request: AnalysisRequest
    ) -> AnalysisResult:
        """
        Execute complete analysis workflow.
        
        Args:
            request: Analysis request with query and parameters
            
        Returns:
            AnalysisResult with all outputs
        """
        start_time = datetime.now()
        
        # Store user query in context
        if self.context_manager:
            self.context_manager.store_user_query(request.query)
        
        # Create workflow graph
        workflow = self._create_workflow(request)
        self.active_workflows[workflow.id] = workflow
        
        # Create executor
        executor = WorkflowExecutor(workflow)
        self.workflow_executors[workflow.id] = executor
        
        # Execute workflow
        result = AnalysisResult(
            workflow_id=workflow.id,
            query=request.query,
            status="in_progress",
            started_at=start_time
        )
        
        try:
            # Execute in topological order
            execution_levels = executor.get_execution_order()
            
            for level_idx, node_ids in enumerate(execution_levels):
                logger.info(f"EXECUTING LEVEL {level_idx + 1}/{len(execution_levels)}")
            
                
                # Execute nodes in this level (can be parallel)
                level_results = await self._execute_level(
                    workflow,
                    executor,
                    node_ids
                )
                
                # Store results
                for node_id, node_result in level_results.items():
                    node = workflow.nodes[node_id]
                    result.stage_results[node.stage.value] = node_result
            
            # Generate final interpretation and report
            if all(node.status == StageStatus.COMPLETED 
                   for node in workflow.nodes.values()):
                result.status = "completed"
                
                # Get final outputs from interpreter
                interp_stage = self._find_stage(workflow, WorkflowStage.INTERPRETATION)
                if interp_stage:
                    result.interpretation = interp_stage.output_data
                
                report_stage = self._find_stage(workflow, WorkflowStage.REPORT_GENERATION)
                if report_stage:
                    result.report = report_stage.output_data
            else:
                result.status = "partial"
                result.errors = [
                    f"{node.stage.value}: {node.error}"
                    for node in workflow.nodes.values()
                    if node.status == StageStatus.FAILED
                ]
        
        except Exception as e:
            result.status = "failed"
            result.errors.append(f"Orchestration error: {str(e)}")
        
        finally:
            result.completed_at = datetime.now()
            result.execution_time_seconds = (
                result.completed_at - start_time
            ).total_seconds()
        
        return result
    
    def _create_workflow(self, request: AnalysisRequest) -> WorkflowGraph:
        """Create appropriate workflow graph for request"""
        if request.workflow_type == "standard_regression":
            return create_standard_regression_workflow()
        elif request.workflow_type == "exploratory":
            from datascienceagent.core.workflow_graph import create_exploratory_workflow
            return create_exploratory_workflow()
        elif request.workflow_type == "modeling_focused":
            from datascienceagent.core.workflow_graph import create_modeling_focused_workflow
            return create_modeling_focused_workflow()
        else:
            raise ValueError(f"Unknown workflow type: {request.workflow_type}")
    
    async def _execute_level(
        self,
        workflow: WorkflowGraph,
        executor: WorkflowExecutor,
        node_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute all nodes in a level.
        
        Can execute in parallel since dependencies are met.
        """
        results = {}
        
        # For now, execute sequentially
        # In production, could use asyncio.gather for parallelism
        for node_id in node_ids:
            node = workflow.nodes[node_id]
            
            # Check if node can be skipped due to failed dependencies
            if executor.can_skip_node(node_id):
                node.status = StageStatus.SKIPPED
                logger.warning(f"⏭️  Skipping {node.stage.value} (dependency failed)")
                continue
            
            # Execute node
            logger.info(f"▶️  Executing: {node.stage.value}")
            logger.info(f"   Description: {node.description}")
            
            try:
                node.status = StageStatus.IN_PROGRESS
                node.started_at = datetime.now()
                
                # Get agent for this node
                agent = self.agents.get(node.agent_name)
                if not agent:
                    raise ValueError(f"Agent not found: {node.agent_name}")
                
                # Prepare input from dependencies
                input_data = self._prepare_input(workflow, node)
                node.input_data = input_data
                
                # Execute with agent
                output = await self._execute_node(agent, node)
                
                # Mark completed
                executor.mark_completed(node_id, output)
                results[node_id] = output
                
                logger.info(f"   ✅ Completed in {(datetime.now() - node.started_at).total_seconds():.1f}s")
            
            except Exception as e:
                error_msg = f"Error executing {node.stage.value}: {str(e)}"
                logger.error(f"   ❌ Failed: {error_msg}")
                
                executor.mark_failed(node_id, error_msg)
                results[node_id] = {"error": error_msg}
        
        return results

    def _prepare_input(
        self,
        workflow: WorkflowGraph,
        node: Any
    ) -> Dict[str, Any]:
        """Prepare input data from dependencies"""
        input_data = {}
        
        # Get outputs from dependencies
        for dep_id in node.depends_on:
            dep_node = workflow.nodes[dep_id]
            if dep_node.output_data:
                input_data[dep_node.stage.value] = dep_node.output_data
        
        return input_data
    
    async def _execute_node(
        self,
        agent: Any,
        node: Any
    ) -> Dict[str, Any]:
        """Execute a single workflow node with appropriate agent"""
        # Route to appropriate agent method based on stage
        stage = node.stage
        
        if stage == WorkflowStage.RESEARCH:
            return await agent.execute_with_context(
                "Research statistical methods for this problem",
                node
            )
        
        elif stage == WorkflowStage.STATISTICAL_PLANNING:
            return await agent.plan_analysis(
                node.input_data.get("query", ""),
                node.input_data.get("data_summary"),
                node
            )
        
        elif stage == WorkflowStage.DATA_ACQUISITION:
            return await agent.load_data(
                node.input_data.get("data_source", {}),
                node
            )
        
        elif stage == WorkflowStage.DATA_VALIDATION:
            return await agent.execute_with_context(
                "Validate data quality and schema",
                node
            )
        
        elif stage == WorkflowStage.DATA_CLEANING:
            return await agent.clean_data(
                node.input_data.get("data_info", {}),
                node.input_data.get("validation_report", {}),
                node
            )
        
        elif stage == WorkflowStage.EDA:
            return await agent.explore_data(
                node.input_data.get("data_info", {}),
                node.input_data.get("objectives", []),
                node
            )
        
        elif stage == WorkflowStage.FEATURE_ENGINEERING:
            return await agent.engineer_features(
                node.input_data.get("eda_insights", {}),
                node
            )
        
        elif stage == WorkflowStage.POST_EDA_PROCESSING:
            return await agent.execute_with_context(
                "Process data based on EDA findings",
                node
            )
        
        elif stage == WorkflowStage.MODEL_SPECIFICATION:
            return await agent.specify_model(
                node.input_data.get("statistical_plan", {}),
                node.input_data.get("data_info", {}),
                node.input_data.get("eda_insights", {}),
                node
            )
        
        elif stage == WorkflowStage.MODEL_FITTING:
            return await agent.fit_model(
                node.input_data.get("model_spec", {}),
                node
            )
        
        elif stage == WorkflowStage.MODEL_DIAGNOSTICS:
            return await agent.run_diagnostics(
                node.input_data.get("fitted_model", {}),
                node
            )
        
        elif stage == WorkflowStage.INTERPRETATION:
            return await agent.interpret_results(
                node.input_data.get("model_results", {}),
                node.input_data.get("diagnostics", {}),
                node.input_data.get("objectives", []),
                node
            )
        
        elif stage == WorkflowStage.REPORT_GENERATION:
            return await agent.generate_report(
                node.input_data.get("interpretation", {}),
                node.input_data,
                node
            )
        
        else:
            return await agent.execute_with_context(
                node.description,
                node
            )
    
    def _find_stage(
        self,
        workflow: WorkflowGraph,
        stage: WorkflowStage
    ) -> Optional[Any]:
        """Find node by stage"""
        for node in workflow.nodes.values():
            if node.stage == stage:
                return node
        return None
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return {"error": "Workflow not found"}
        
        executor = self.workflow_executors.get(workflow_id)
        if not executor:
            return {"error": "Executor not found"}
        
        progress = executor.get_progress()
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "progress": progress,
            "current_stages": [
                {
                    "stage": node.stage.value,
                    "status": node.status.value,
                    "description": node.description
                }
                for node in workflow.nodes.values()
                if node.status == StageStatus.IN_PROGRESS
            ],
            "completed_stages": [
                node.stage.value
                for node in workflow.nodes.values()
                if node.status == StageStatus.COMPLETED
            ],
            "failed_stages": [
                {
                    "stage": node.stage.value,
                    "error": node.error
                }
                for node in workflow.nodes.values()
                if node.status == StageStatus.FAILED
            ]
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*70)
        print("WORKFLOW ORCHESTRATOR DEMO")
        print("="*70)
        from datascienceagent.core.context_management import ContextManager
        
        # Create orchestrator (without context manager for demo)
        orchestrator = WorkflowOrchestrator(context_manager=ContextManager(model="openai:gpt-4.1-mini"), model="openai:gpt-4.1-mini")
        
        # Create analysis request
        request = AnalysisRequest(
            query="""
            I have sales data with marketing spend, seasonality, and competitor actions.
            I want to understand what drives sales and build a predictive model.
            """,
            workflow_type="standard_regression",
            objectives=[
                "Identify key drivers of sales",
                "Quantify effect of marketing spend",
                "Account for seasonality",
                "Build predictive model"
            ],
            data_source={
                "type": "csv",
                "path": "data/sales_data.csv"
            }
        )
        
        print(f"\n📊 Analysis Request:")
        print(f"   Query: {request.query.strip()}")
        print(f"   Workflow: {request.workflow_type}")
        print(f"   Objectives: {len(request.objectives)} objectives")
        
        # Execute analysis
        result = await orchestrator.execute_analysis(request)
        
        print(f"\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_seconds:.1f}s")
        print(f"Stages completed: {len(result.stage_results)}")
        
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        console.print(Markdown(f"### Final Interpretation:\n{result.interpretation.get('raw_result', '')}" if result.interpretation else "No interpretation available."))
        console.print(Markdown(f"### Final Report:\n{result.report.get('raw_result', '')}" if result.report else "No report available."))
        

        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
        
        print("\n✅ Demo completed")
    
    asyncio.run(demo())
