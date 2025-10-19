"""
Orchestrator for Data Science Workflow Execution - REFACTORED

Key improvements:
1. Properly passes initial request data (including data_source) to workflow
2. Uses executor's prepare_node_input to combine dependency outputs and initial data
3. Ensures DATA_ACQUISITION node receives data_source information
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from pydantic import BaseModel, Field
from loguru import logger

from datascienceagent.core.workflow_graph import (
    WorkflowGraph, WorkflowExecutor, WorkflowStage,
    StageStatus, create_standard_regression_workflow,
    create_exploratory_workflow, create_modeling_focused_workflow
)
from datascienceagent.agents.specialist_agents import AgentFactory
from datascienceagent.utils.model_utils import process_model

# ============================================================================
# ORCHESTRATOR - REFACTORED
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request for analysis"""
    query: str
    workflow_type: str = "standard_regression"
    data_source: Optional[Dict[str, Any]] = None  # CRITICAL: Must be passed to workflow!
    objectives: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Complete analysis result"""
    workflow_id: str
    query: str
    status: str
    
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
    
    REFACTORED: Now properly passes initial request data to workflow nodes.
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
        
        logger.info(f"✅ Initialized {len(self.agents)} specialist agents")
    
    async def execute_analysis(
        self,
        request: AnalysisRequest
    ) -> AnalysisResult:
        """
        Execute complete analysis workflow.
        
        REFACTORED: Properly passes request data to workflow.
        """
        start_time = datetime.now()
        
        # Store user query in context
        if self.context_manager:
            self.context_manager.store_user_query(request.query)
        
        # REFACTORED: Create workflow WITH initial data from request
        workflow = self._create_workflow(request)
        self.active_workflows[workflow.id] = workflow
        
        logger.info(f"📊 Created workflow: {workflow.name}")
        logger.info(f"   Initial data keys: {list(workflow.initial_data.keys())}")
        
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
                logger.info(f"\n{'='*70}")
                logger.info(f"EXECUTING LEVEL {level_idx + 1}/{len(execution_levels)}")
                logger.info(f"{'='*70}")
                
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
            logger.error(f"❌ Orchestration error: {str(e)}")
            result.status = "failed"
            result.errors.append(f"Orchestration error: {str(e)}")
        
        finally:
            result.completed_at = datetime.now()
            result.execution_time_seconds = (
                result.completed_at - start_time
            ).total_seconds()
        
        return result
    
    def _create_workflow(self, request: AnalysisRequest) -> WorkflowGraph:
        """
        Create appropriate workflow graph for request.
        
        REFACTORED: Now passes initial data from request to workflow!
        This is KEY to fixing the data_source passing issue.
        """
        # Prepare initial data from request
        initial_data = {
            "query": request.query,
            "data_source": request.data_source,  # CRITICAL!
            "objectives": request.objectives,
            "metadata": request.metadata
        }
        
        logger.info(f"📋 Creating workflow with initial data:")
        logger.info(f"   - query: {request.query[:50]}...")
        logger.info(f"   - data_source: {request.data_source}")
        logger.info(f"   - objectives: {len(request.objectives)} objectives")
        
        # Create workflow based on type, passing initial_data
        if request.workflow_type == "standard_regression":
            return create_standard_regression_workflow(initial_data=initial_data)
        elif request.workflow_type == "exploratory":
            return create_exploratory_workflow(initial_data=initial_data)
        elif request.workflow_type == "modeling_focused":
            return create_modeling_focused_workflow(initial_data=initial_data)
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
        
        REFACTORED: Uses executor's prepare_node_input to get both
        dependency outputs AND initial data.
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
            logger.info(f"\n▶️  Executing: {node.stage.value}")
            logger.info(f"   Description: {node.description}")
            
            try:
                node.status = StageStatus.IN_PROGRESS
                node.started_at = datetime.now()
                
                # Get agent for this node
                agent = self.agents.get(node.agent_name)
                if not agent:
                    raise ValueError(f"Agent not found: {node.agent_name}")
                
                # REFACTORED: Use executor's prepare_node_input
                # This combines dependency outputs AND initial data!
                input_data = executor.prepare_node_input(node_id)
                node.input_data = input_data
                
                # Log what input the node is receiving
                logger.info(f"   Input data keys: {list(input_data.keys())}")
                if node.needs_initial_data:
                    logger.info(f"   ✓ Node receives initial data: {node.initial_data_keys}")
                
                # Execute with agent
                output = await self._execute_node(agent, node)
                
                # Mark completed
                executor.mark_completed(node_id, output)
                results[node_id] = output
                
                elapsed = (datetime.now() - node.started_at).total_seconds()
                logger.info(f"   ✅ Completed in {elapsed:.1f}s")
            
            except Exception as e:
                error_msg = f"Error executing {node.stage.value}: {str(e)}"
                logger.error(f"   ❌ Failed: {error_msg}")
                
                executor.mark_failed(node_id, error_msg)
                results[node_id] = {"error": error_msg}
        
        return results
    
    async def _execute_node(
        self,
        agent: Any,
        node: Any
    ) -> Dict[str, Any]:
        """
        Execute a single workflow node with appropriate agent.
        
        The node.input_data now contains BOTH dependency outputs AND initial data!
        """
        stage = node.stage
        
        if stage == WorkflowStage.RESEARCH:
            # Has access to query and objectives from initial_data
            return await agent.execute_with_context(
                f"Research statistical methods for: {node.input_data.get('query', 'the problem')}",
                node
            )
        
        elif stage == WorkflowStage.STATISTICAL_PLANNING:
            return await agent.plan_analysis(
                node.input_data.get("query", ""),
                node.input_data.get("data_summary"),
                node
            )
        
        elif stage == WorkflowStage.DATA_ACQUISITION:
            # CRITICAL: Now has access to data_source from initial_data!
            data_source = node.input_data.get("data_source")
            if not data_source:
                logger.warning("⚠️  No data_source in input_data!")
                logger.warning(f"   Available keys: {list(node.input_data.keys())}")
            else:
                logger.info(f"   ✓ Data source: {data_source}")
            
            return await agent.load_data(
                data_source or {},
                node
            )
        
        elif stage == WorkflowStage.DATA_VALIDATION:
            return await agent.execute_with_context(
                "Validate data quality and schema",
                node
            )
        
        elif stage == WorkflowStage.DATA_CLEANING:
            return await agent.clean_data(
                node.input_data.get("data_acquisition", {}),  # From dependency
                node.input_data.get("data_validation", {}),   # From dependency
                node
            )
        
        elif stage == WorkflowStage.EDA:
            # Has access to objectives from initial_data
            return await agent.explore_data(
                node.input_data.get("data_cleaning", {}),
                node.input_data.get("objectives", []),  # From initial_data!
                node
            )
        
        elif stage == WorkflowStage.FEATURE_ENGINEERING:
            return await agent.engineer_features(
                node.input_data.get("eda", {}),
                node
            )
        
        elif stage == WorkflowStage.POST_EDA_PROCESSING:
            return await agent.execute_with_context(
                "Process data based on EDA findings",
                node
            )
        
        elif stage == WorkflowStage.MODEL_SPECIFICATION:
            return await agent.specify_model(
                node.input_data.get("statistical_planning", {}),
                node.input_data.get("data_cleaning", {}),
                node.input_data.get("eda", {}),
                node
            )
        
        elif stage == WorkflowStage.MODEL_FITTING:
            return await agent.fit_model(
                node.input_data.get("model_specification", {}),
                node
            )
        
        elif stage == WorkflowStage.MODEL_DIAGNOSTICS:
            return await agent.run_diagnostics(
                node.input_data.get("model_fitting", {}),
                node
            )
        
        elif stage == WorkflowStage.INTERPRETATION:
            # Has access to query and objectives from initial_data
            return await agent.interpret_results(
                node.input_data.get("model_fitting", {}),
                node.input_data.get("model_diagnostics", {}),
                node.input_data.get("objectives", []),  # From initial_data!
                node
            )
        
        elif stage == WorkflowStage.REPORT_GENERATION:
            return await agent.generate_report(
                node.input_data.get("interpretation", {}),
                node.input_data,  # Pass all available data
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
                    "description": node.description,
                    "has_initial_data": node.needs_initial_data,
                    "input_keys": list(node.input_data.keys()) if node.input_data else []
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
# EXAMPLE USAGE WITH PROPER DATA PASSING
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*70)
        print("REFACTORED WORKFLOW ORCHESTRATOR DEMO")
        print("Testing proper data_source passing to DATA_ACQUISITION node")
        print("="*70)
        
        from datascienceagent.core.context_management import ContextManager
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator(
            context_manager=ContextManager(model="openai:gpt-4.1-mini"),
            model="openai:gpt-4.1-mini"
        )
        
        # Create analysis request with data_source
        request = AnalysisRequest(
            query="""
            Analyze the relationship between marketing spend and sales.
            Control for seasonality and competitor actions.
            """,
            workflow_type="standard_regression",
            data_source={
                "type": "csv",
                "path": "data/sales_data.csv",
                "parameters": {
                    "delimiter": ",",
                    "header": True
                }
            },
            objectives=[
                "Identify key drivers of sales",
                "Quantify effect of marketing spend",
                "Account for seasonality",
                "Build predictive model"
            ]
        )
        
        print(f"\n📊 Analysis Request:")
        print(f"   Query: {request.query.strip()}")
        print(f"   Data source type: {request.data_source['type']}")
        print(f"   Data source path: {request.data_source['path']}")
        print(f"   Objectives: {len(request.objectives)}")
        
        # Execute analysis
        result = await orchestrator.execute_analysis(request)
        
        print(f"\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_seconds:.1f}s")
        print(f"Stages completed: {len(result.stage_results)}")
        
        # Check if DATA_ACQUISITION stage received data_source
        if "data_acquisition" in result.stage_results:
            print(f"\n✅ DATA_ACQUISITION stage executed")
            print(f"   Result: {result.stage_results['data_acquisition']}")
        
        if result.errors:
            print(f"\n❌ Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        print("\n✅ Demo completed")
        print("\n💡 Key improvements:")
        print("  ✓ data_source passed from request to workflow")
        print("  ✓ workflow stores initial_data")
        print("  ✓ nodes flagged with needs_initial_data get the data")
        print("  ✓ executor.prepare_node_input combines all sources")
        print("  ✓ DATA_ACQUISITION node receives data_source!")
    
    asyncio.run(demo())