"""
Output-Aware Workflow Orchestrator

Enhanced orchestrator that combines:
- Context-aware specialist agents
- Systematic output management
- Organized results storage
- Complete execution tracking

This orchestrator ensures all agent outputs are captured and organized
in a structured results directory for easy review and sharing.

Author: Data Science Agent System
Created: 2025-10-19
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from loguru import logger

from datascienceagent.core.workflow_graph import (
    WorkflowGraph,
    WorkflowExecutor,
    WorkflowStage,
    StageStatus,
    create_standard_regression_workflow,
    create_exploratory_workflow,
    create_modeling_focused_workflow,
)
from datascienceagent.core.context_management import ContextManager
from datascienceagent.core.output_manager import OutputManager
from datascienceagent.agents.output_aware_specialist_agents import OutputAwareAgentFactory
from datascienceagent.utils.model_utils import process_model


# ============================================================================
# REQUEST AND RESULT MODELS
# ============================================================================


class AnalysisRequest(BaseModel):
    """Request for analysis with output management"""
    
    query: str
    workflow_type: str = "standard_regression"
    data_source: Optional[Dict[str, Any]] = None
    objectives: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Output options
    results_base_dir: str = "results"
    workflow_id: Optional[str] = None


class AnalysisResult(BaseModel):
    """Complete analysis result with output tracking"""
    
    workflow_id: str
    query: str
    status: str
    
    # Results by stage
    stage_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Final outputs
    interpretation: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    
    # Output tracking
    results_directory: Optional[str] = None
    manifest_path: Optional[str] = None
    
    # Execution metadata
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_seconds: float = 0
    
    # Errors
    errors: List[str] = Field(default_factory=list)


# ============================================================================
# OUTPUT-AWARE ORCHESTRATOR
# ============================================================================


class OutputAwareOrchestrator:
    """
    Orchestrates complete data science workflows with output management.
    
    Features:
    - Context-aware specialist agents
    - Systematic output capture and organization
    - Structured results directory
    - Complete execution manifest
    - Easy result sharing and review
    
    Usage:
        orchestrator = OutputAwareOrchestrator(
            context_manager=context_mgr,
            model="openai:gpt-4"
        )
        
        request = AnalysisRequest(
            query="Analyze sales data",
            data_source={"type": "csv", "path": "data.csv"},
            objectives=["Find key drivers", "Build model"]
        )
        
        result = await orchestrator.execute_analysis(request)
        
        # Results saved to: result.results_directory
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        model: str = "openai:gpt-4",
    ):
        """
        Initialize orchestrator.
        
        Args:
            context_manager: Context manager for agent memory
            model: LLM model to use for all agents
        """
        self.context_manager = context_manager
        self.model = process_model(model)
        self.agents: Dict[str, Any] = {}
        
        # Output manager will be created per workflow
        self.output_manager: Optional[OutputManager] = None
        
        # Track active workflows
        self.active_workflows: Dict[str, WorkflowGraph] = {}
        self.workflow_executors: Dict[str, WorkflowExecutor] = {}
        
        logger.info("🎯 OutputAwareOrchestrator initialized")
    
    def _initialize_agents(self, output_manager: OutputManager):
        """
        Initialize all specialist agents with output management.
        
        Args:
            output_manager: Output manager for this workflow
        """
        agent_types = [
            "statistician",
            "data_engineer",
            "eda",
            "modeling",
            "interpreter",
        ]
        
        self.agents = {}
        for agent_type in agent_types:
            self.agents[f"{agent_type}_agent"] = OutputAwareAgentFactory.create_agent(
                agent_type,
                model=self.model,
                context_manager=self.context_manager,
                output_manager=output_manager,
            )
        
        logger.info(f"✅ Initialized {len(self.agents)} output-aware agents")
    
    async def execute_analysis(
        self, request: AnalysisRequest
    ) -> AnalysisResult:
        """
        Execute complete analysis workflow with output management.
        
        Args:
            request: Analysis request with query, data source, etc.
            
        Returns:
            AnalysisResult with status and output locations
        """
        start_time = datetime.now()
        
        # Create output manager for this workflow
        self.output_manager = OutputManager(
            base_results_dir=request.results_base_dir,
            workflow_id=request.workflow_id,
        )
        
        # Initialize agents with output manager
        self._initialize_agents(self.output_manager)
        
        # Store user query in context
        if self.context_manager:
            self.context_manager.store_user_query(request.query)
        
        # Save initial request
        self.output_manager.save_json(
            "orchestrator",
            {
                "query": request.query,
                "workflow_type": request.workflow_type,
                "data_source": request.data_source,
                "objectives": request.objectives,
                "metadata": request.metadata,
                "timestamp": start_time.isoformat(),
            },
            "analysis_request.json",
            stage_order=0,
        )
        
        # Create workflow with initial data
        workflow = self._create_workflow(request)
        self.active_workflows[workflow.id] = workflow
        
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING WORKFLOW: {workflow.name}")
        logger.info(f"{'='*70}")
        logger.info(f"📋 Query: {request.query}")
        logger.info(f"📁 Results will be saved to: {self.output_manager.workflow_dir}")
        logger.info(f"{'='*70}\n")
        
        # Create executor
        executor = WorkflowExecutor(workflow)
        self.workflow_executors[workflow.id] = executor
        
        # Initialize result
        result = AnalysisResult(
            workflow_id=workflow.id,
            query=request.query,
            status="in_progress",
            started_at=start_time,
            results_directory=str(self.output_manager.workflow_dir),
        )
        
        try:
            # Execute in topological order
            execution_levels = executor.get_execution_order()
            
            # Save execution plan
            self.output_manager.save_json(
                "orchestrator",
                {
                    "execution_levels": [
                        [workflow.nodes[nid].stage.value for nid in level]
                        for level in execution_levels
                    ],
                    "total_levels": len(execution_levels),
                    "total_nodes": len(workflow.nodes),
                },
                "execution_plan.json",
                stage_order=0,
            )
            
            for level_idx, node_ids in enumerate(execution_levels):
                logger.info(f"\n{'='*70}")
                logger.info(f"LEVEL {level_idx + 1}/{len(execution_levels)}")
                logger.info(f"{'='*70}")
                
                # Execute nodes in this level
                level_results = await self._execute_level(
                    workflow, executor, node_ids, level_idx
                )
                
                # Store results
                for node_id, node_result in level_results.items():
                    node = workflow.nodes[node_id]
                    result.stage_results[node.stage.value] = node_result
            
            # Check completion status
            if all(
                node.status == StageStatus.COMPLETED
                for node in workflow.nodes.values()
            ):
                result.status = "completed"
                
                # Get final outputs
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
            
            # Save final result
            self.output_manager.save_json(
                "orchestrator",
                result.dict(),
                "analysis_result.json",
                stage_order=999,
            )
            
            # Generate manifest and summary
            manifest_path = self.output_manager.save_manifest()
            result.manifest_path = str(manifest_path)
            
            self.output_manager.create_summary_report()
            
            # Log completion
            logger.info(f"\n{'='*70}")
            logger.info(f"WORKFLOW COMPLETED: {result.status.upper()}")
            logger.info(f"{'='*70}")
            logger.info(f"⏱️  Execution time: {result.execution_time_seconds:.1f}s")
            logger.info(f"📁 Results saved to: {result.results_directory}")
            logger.info(f"📋 Manifest: {result.manifest_path}")
            logger.info(f"{'='*70}\n")
        
        return result
    
    def _create_workflow(self, request: AnalysisRequest) -> WorkflowGraph:
        """
        Create appropriate workflow graph for request.
        
        Args:
            request: Analysis request
            
        Returns:
            WorkflowGraph with initial data
        """
        # Prepare initial data from request
        initial_data = {
            "query": request.query,
            "data_source": request.data_source,
            "objectives": request.objectives,
            "metadata": request.metadata,
        }
        
        logger.info(f"📋 Creating {request.workflow_type} workflow")
        logger.info(f"   Initial data keys: {list(initial_data.keys())}")
        
        # Create workflow based on type
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
        node_ids: List[str],
        level_idx: int,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute all nodes in a level.
        
        Args:
            workflow: Workflow graph
            executor: Workflow executor
            node_ids: Node IDs to execute
            level_idx: Level index for ordering
            
        Returns:
            Dict of results by node ID
        """
        results = {}
        
        # Execute sequentially (could parallelize in production)
        for node_id in node_ids:
            node = workflow.nodes[node_id]
            
            # Check if node should be skipped
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
                
                # Prepare node input
                input_data = executor.prepare_node_input(node_id)
                node.input_data = input_data
                
                logger.info(f"   Input keys: {list(input_data.keys())}")
                
                # Calculate stage order (level * 10 + position in level)
                stage_order = (level_idx * 10) + node_ids.index(node_id) + 1
                
                # Execute with agent
                output = await self._execute_node(
                    agent, node, stage_order
                )
                
                # Mark completed
                executor.mark_completed(node_id, output)
                results[node_id] = output
                
                elapsed = (datetime.now() - node.started_at).total_seconds()
                logger.info(f"   ✅ Completed in {elapsed:.1f}s")
            
            except Exception as e:
                error_msg = f"Error executing {node.stage.value}: {str(e)}"
                logger.error(f"   ❌ {error_msg}")
                
                executor.mark_failed(node_id, error_msg)
                results[node_id] = {"error": error_msg}
        
        return results
    
    async def _execute_node(
        self, agent: Any, node: Any, stage_order: int
    ) -> Dict[str, Any]:
        """
        Execute a single workflow node with appropriate agent.
        
        Args:
            agent: Agent to execute
            node: Workflow node
            stage_order: Stage order number
            
        Returns:
            Dict with execution results
        """
        stage = node.stage
        
        # Route to appropriate agent method based on stage
        if stage == WorkflowStage.RESEARCH:
            return await agent.execute_with_outputs(
                f"Research statistical methods for: {node.input_data.get('query', 'the problem')}",
                node,
                stage_order=stage_order,
            )
        
        elif stage == WorkflowStage.STATISTICAL_PLANNING:
            return await agent.plan_analysis(
                node.input_data.get("query", ""),
                node.input_data.get("data_summary"),
                node,
                stage_order=stage_order,
            )
        
        elif stage == WorkflowStage.DATA_ACQUISITION:
            data_source = node.input_data.get("data_source")
            if not data_source:
                logger.warning("⚠️  No data_source in input_data!")
            
            return await agent.load_data(
                data_source or {}, node, stage_order=stage_order
            )
        
        elif stage == WorkflowStage.DATA_VALIDATION:
            return await agent.execute_with_outputs(
                "Validate data quality and schema", node, stage_order=stage_order
            )
        
        elif stage == WorkflowStage.DATA_CLEANING:
            return await agent.clean_data(
                node.input_data.get("data_acquisition", {}),
                node.input_data.get("data_validation", {}),
                node,
                stage_order=stage_order,
            )
        
        elif stage == WorkflowStage.EDA:
            return await agent.explore_data(
                node.input_data.get("data_cleaning", {}),
                node.input_data.get("objectives", []),
                node,
                stage_order=stage_order,
            )
        
        elif stage == WorkflowStage.FEATURE_ENGINEERING:
            return await agent.engineer_features(
                node.input_data.get("eda", {}), node, stage_order=stage_order
            )
        
        elif stage == WorkflowStage.POST_EDA_PROCESSING:
            return await agent.execute_with_outputs(
                "Process data based on EDA findings", node, stage_order=stage_order
            )
        
        elif stage == WorkflowStage.MODEL_SPECIFICATION:
            return await agent.specify_model(
                node.input_data.get("statistical_planning", {}),
                node.input_data.get("data_cleaning", {}),
                node.input_data.get("eda", {}),
                node,
                stage_order=stage_order,
            )
        
        elif stage == WorkflowStage.MODEL_FITTING:
            return await agent.fit_model(
                node.input_data.get("model_specification", {}),
                node,
                stage_order=stage_order,
            )
        
        elif stage == WorkflowStage.MODEL_DIAGNOSTICS:
            return await agent.run_diagnostics(
                node.input_data.get("model_fitting", {}),
                node,
                stage_order=stage_order,
            )
        
        elif stage == WorkflowStage.INTERPRETATION:
            return await agent.interpret_results(
                node.input_data.get("model_fitting", {}),
                node.input_data.get("model_diagnostics", {}),
                node.input_data.get("objectives", []),
                node,
                stage_order=stage_order,
            )
        
        elif stage == WorkflowStage.REPORT_GENERATION:
            return await agent.generate_report(
                node.input_data.get("interpretation", {}),
                node.input_data,
                node,
                stage_order=stage_order,
            )
        
        else:
            return await agent.execute_with_outputs(
                node.description, node, stage_order=stage_order
            )
    
    def _find_stage(
        self, workflow: WorkflowGraph, stage: WorkflowStage
    ) -> Optional[Any]:
        """Find node by stage"""
        for node in workflow.nodes.values():
            if node.stage == stage:
                return node
        return None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


async def demo():
    """Demonstrate the output-aware orchestrator"""
    print("\n" + "="*70)
    print("OUTPUT-AWARE WORKFLOW ORCHESTRATOR DEMO")
    print("="*70)
    
    # Create context manager
    context_mgr = ContextManager(
        persist_directory="./demo_context",
        model="openai:gpt-4.1-mini"
    )
    
    # Create orchestrator
    orchestrator = OutputAwareOrchestrator(
        context_manager=context_mgr,
        model="openai:gpt-4.1-mini"
    )
    
    # Create analysis request
    request = AnalysisRequest(
        query="""
        Analyze the relationship between marketing spend and sales.
        Control for seasonality and competitor actions.
        """,
        workflow_type="standard_regression",
        data_source={
            "type": "csv",
            "path": "data/sales_data.csv",
            "parameters": {"delimiter": ",", "header": True},
        },
        objectives=[
            "Identify key drivers of sales",
            "Quantify effect of marketing spend",
            "Account for seasonality",
            "Build predictive model",
        ],
        results_base_dir="demo_results",
    )
    
    print(f"\n📊 Analysis Request:")
    print(f"   Query: {request.query.strip()}")
    print(f"   Workflow: {request.workflow_type}")
    print(f"   Data source: {request.data_source['path']}")
    print(f"   Objectives: {len(request.objectives)}")
    
    # Execute analysis
    result = await orchestrator.execute_analysis(request)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Status: {result.status}")
    print(f"Execution time: {result.execution_time_seconds:.1f}s")
    print(f"Stages completed: {len(result.stage_results)}")
    print(f"\n📁 Results directory: {result.results_directory}")
    print(f"📋 Manifest: {result.manifest_path}")
    
    if result.errors:
        print(f"\n❌ Errors:")
        for error in result.errors:
            print(f"  - {error}")
    
    print("\n✅ Demo completed!")
    print(f"\n💡 Review results in: {result.results_directory}")


if __name__ == "__main__":
    asyncio.run(demo())