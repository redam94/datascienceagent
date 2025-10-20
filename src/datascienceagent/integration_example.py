"""
Agentic Context-Aware Orchestrator with Error Handling

Version: 1.0.1 (Fixed ContextManager API compatibility)

This is a complete rework of integration_example.py that combines:
1. Context-aware specialist agents (learn from past work)
2. Output-aware workflow management (organized result storage)
3. Comprehensive error handling with automatic retry
4. Agentic workflow execution (LLM decides which nodes to run)

Key Innovation: The orchestrator uses an LLM to dynamically decide which
workflow nodes to execute based on the query, data characteristics, and
results from previous nodes. Not all nodes need to run every time.

Author: Data Science Agent System
Created: 2025-10-19
"""

import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from pathlib import Path
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from loguru import logger

# Import core components
from datascienceagent.core.workflow_graph import (
    WorkflowGraph,
    WorkflowExecutor,
    WorkflowStage,
    StageStatus,
    create_standard_regression_workflow,
    create_exploratory_workflow,
    create_modeling_focused_workflow,
)
from datascienceagent.core.context_management import ContextManager, ContextType
from datascienceagent.core.output_manager import OutputManager

# Import agents
from datascienceagent.agents.output_aware_specialist_agents import OutputAwareAgentFactory

# Import error handling
from datascienceagent.core.error_handling import ErrorAnalyzer, ErrorCategory, ErrorSeverity
from datascienceagent.core.retry_strategies import (
    RetryManager,
    RetryConfig,
    RetryResult,
    create_standard_retry_config,
    create_aggressive_retry_config,
)
from datascienceagent.core.enhanced_workflow import (
    EnhancedWorkflowExecutor,
    WorkflowRecoveryPolicy,
    NodeExecutionRecord,
    WorkflowExecutionReport,
)

# Import utilities
from datascienceagent.utils.model_utils import process_model


# ============================================================================
# AGENTIC WORKFLOW CONTROLLER
# ============================================================================


class NodeExecutionDecision(BaseModel):
    """Decision about whether to execute a node"""
    
    node_id: str
    should_execute: bool
    reasoning: str
    priority: int = 5  # 1-10, higher = more important
    alternative_approach: Optional[str] = None


class WorkflowPlan(BaseModel):
    """Plan for executing workflow nodes"""
    
    nodes_to_execute: List[str] = Field(default_factory=list)
    nodes_to_skip: List[str] = Field(default_factory=list)
    execution_order: List[str] = Field(default_factory=list)
    reasoning: str
    estimated_stages: int = 0


class AgenticWorkflowController:
    """
    LLM-powered controller that decides which workflow nodes to execute.
    
    This is the key innovation: instead of always executing all nodes,
    the controller analyzes the query, data, and intermediate results to
    determine the optimal execution path.
    
    Features:
    - Query analysis to determine required analyses
    - Data-driven decision making
    - Adaptive workflows based on intermediate results
    - Skip unnecessary nodes to save time
    - Re-plan workflow when errors occur
    """
    
    def __init__(
        self,
        model: str = "openai:gpt-4",
        context_manager: Optional[ContextManager] = None
    ):
        """Initialize agentic workflow controller"""
        self.model = process_model(model)
        self.context_manager = context_manager
        
        # Create planning agent
        self.planner = Agent(
            self.model,
            system_prompt=self._get_planner_prompt(),
        )
        
        logger.info("✅ AgenticWorkflowController initialized")
    
    def _get_planner_prompt(self) -> str:
        """Get system prompt for planning agent"""
        return """You are an expert data science workflow planner.

Your role is to analyze analysis requests and determine which workflow
stages are actually needed to answer the user's question effectively.

You have these available stages:
- RESEARCH: Statistical research and method selection
- DATA_ACQUISITION: Load and validate data
- DATA_ENGINEERING: Clean, transform, prepare data
- EXPLORATORY_ANALYSIS: Understand data patterns and relationships
- STATISTICAL_TESTING: Hypothesis testing and validation
- FEATURE_ENGINEERING: Create features for modeling
- MODEL_BUILDING: Train and tune models
- MODEL_VALIDATION: Evaluate model performance
- INTERPRETATION: Interpret and explain results

Guidelines:
1. Not every analysis needs all stages
2. Simple queries may only need 2-3 stages
3. Complex modeling requires the full pipeline
4. Consider data quality - poor data needs more engineering
5. If errors occur, suggest alternative approaches
6. Be efficient - don't waste computation on unnecessary steps

Always provide clear reasoning for your decisions."""
    
    async def create_initial_plan(
        self,
        query: str,
        data_source: Dict[str, Any],
        workflow: WorkflowGraph
    ) -> WorkflowPlan:
        """
        Create initial workflow execution plan based on query and data.
        
        Args:
            query: User's analysis request
            data_source: Information about the data
            workflow: Available workflow graph
            
        Returns:
            WorkflowPlan with nodes to execute
        """
        logger.info("\n🧠 Creating agentic workflow plan...")
        
        # Prepare context for planner
        available_nodes = {
            node_id: {
                "stage": node.stage.value,
                "description": node.description,
                "depends_on": node.depends_on,
            }
            for node_id, node in workflow.nodes.items()
        }
        
        # Get past similar analyses if available
        past_context = ""
        if self.context_manager:
            similar = await self.context_manager.get_relevant_context(query, limit=2)
            if similar:
                past_context = "\n\nPast similar analyses:\n"
                for ex in similar:
                    past_context += f"- {ex['content'][:200]}...\n"
        
        # Ask planner to create execution plan
        planning_prompt = f"""Analyze this data science request and create an execution plan.

USER QUERY:
{query}

DATA SOURCE:
{data_source}

AVAILABLE WORKFLOW NODES:
{available_nodes}
{past_context}

Determine:
1. Which nodes are essential for this query?
2. Which nodes can be skipped?
3. What's the optimal execution order?
4. Why is this plan appropriate?

Consider:
- Query complexity (simple vs complex)
- Whether modeling is explicitly requested
- Data quality requirements
- Time efficiency

Return a JSON object with:
{{
    "nodes_to_execute": ["node_id1", "node_id2", ...],
    "nodes_to_skip": ["node_id3", ...],
    "execution_order": ["node_id1", "node_id2", ...],
    "reasoning": "Explanation of plan",
    "estimated_stages": <number>
}}"""
        
        try:
            # Get plan from LLM
            result = await self.planner.run(planning_prompt)
            
            # Parse result
            plan_data = result.data
            if isinstance(plan_data, str):
                import json
                plan_data = json.loads(plan_data)
            
            plan = WorkflowPlan(**plan_data)
            
            logger.info(f"📋 Agentic plan created:")
            logger.info(f"   Nodes to execute: {len(plan.nodes_to_execute)}")
            logger.info(f"   Nodes to skip: {len(plan.nodes_to_skip)}")
            logger.info(f"   Reasoning: {plan.reasoning[:100]}...")
            
            return plan
            
        except Exception as e:
            logger.warning(f"⚠️  Planning failed, using default plan: {e}")
            # Fallback: execute all nodes
            return WorkflowPlan(
                nodes_to_execute=list(workflow.nodes.keys()),
                nodes_to_skip=[],
                execution_order=list(workflow.nodes.keys()),
                reasoning="Using default full workflow due to planning error",
                estimated_stages=len(workflow.nodes)
            )
    
    async def should_continue_after_node(
        self,
        node_id: str,
        node_result: Dict[str, Any],
        remaining_nodes: List[str],
        workflow: WorkflowGraph
    ) -> Dict[str, Any]:
        """
        Decide whether to continue workflow after seeing node results.
        
        This allows adaptive workflows that change based on intermediate results.
        
        Args:
            node_id: Just-completed node ID
            node_result: Result from the node
            remaining_nodes: Nodes still in plan
            workflow: Workflow graph
            
        Returns:
            Dict with: continue (bool), skip_nodes (list), reasoning (str)
        """
        logger.info(f"\n🤔 Evaluating whether to continue after {node_id}...")
        
        # Check if node failed
        if not node_result.get("success", True):
            error = node_result.get("error", "Unknown error")
            logger.warning(f"   Node failed: {error}")
            
            # Ask planner for recovery strategy
            recovery_prompt = f"""A workflow node failed:

NODE: {node_id} ({workflow.nodes[node_id].stage.value})
ERROR: {error}
REMAINING NODES: {remaining_nodes}

Should we:
1. Continue with remaining nodes?
2. Skip some nodes?
3. Stop the workflow?

Return JSON:
{{
    "continue": true/false,
    "skip_nodes": ["node_ids to skip"],
    "reasoning": "explanation"
}}"""
            
            try:
                result = await self.planner.run(recovery_prompt)
                decision = result.data
                if isinstance(decision, str):
                    import json
                    decision = json.loads(decision)
                
                logger.info(f"   Decision: {'Continue' if decision['continue'] else 'Stop'}")
                logger.info(f"   Reasoning: {decision['reasoning'][:100]}...")
                return decision
                
            except Exception as e:
                logger.warning(f"   Decision failed: {e}, continuing by default")
                return {"continue": True, "skip_nodes": [], "reasoning": "Default continue"}
        
        # Node succeeded - continue normally
        return {"continue": True, "skip_nodes": [], "reasoning": "Node succeeded"}
    
    async def adapt_plan_for_error(
        self,
        failed_node_id: str,
        error: str,
        original_plan: WorkflowPlan,
        workflow: WorkflowGraph
    ) -> WorkflowPlan:
        """
        Adapt workflow plan when a node fails.
        
        The planner suggests alternative approaches or node skipping.
        
        Args:
            failed_node_id: Node that failed
            error: Error message
            original_plan: Original execution plan
            workflow: Workflow graph
            
        Returns:
            Adapted WorkflowPlan
        """
        logger.info(f"\n🔄 Adapting plan after {failed_node_id} failure...")
        
        adaptation_prompt = f"""A workflow node failed. Suggest an adapted plan.

FAILED NODE: {failed_node_id}
ERROR: {error}
ORIGINAL PLAN: {original_plan.dict()}
REMAINING NODES: {[n for n in original_plan.nodes_to_execute 
                   if n != failed_node_id]}

Suggest:
1. Can we skip the failed node and continue?
2. Should we try alternative nodes?
3. Is this a critical failure?

Return adapted plan as JSON:
{{
    "nodes_to_execute": [...],
    "nodes_to_skip": [...],
    "execution_order": [...],
    "reasoning": "...",
    "estimated_stages": N
}}"""
        
        try:
            result = await self.planner.run(adaptation_prompt)
            adapted_data = result.data
            if isinstance(adapted_data, str):
                import json
                adapted_data = json.loads(adapted_data)
            
            adapted_plan = WorkflowPlan(**adapted_data)
            logger.info(f"   Adapted plan: {adapted_plan.reasoning[:100]}...")
            return adapted_plan
            
        except Exception as e:
            logger.warning(f"   Adaptation failed: {e}, keeping original plan")
            return original_plan


# ============================================================================
# AGENTIC CONTEXT-AWARE ORCHESTRATOR
# ============================================================================


class AgenticAnalysisRequest(BaseModel):
    """Request for agentic analysis"""
    
    query: str
    workflow_type: str = "standard_regression"
    data_source: Optional[Dict[str, Any]] = None
    objectives: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Agentic options
    enable_adaptive_workflow: bool = True
    enable_error_recovery: bool = True
    max_execution_time_minutes: int = 30
    
    # Output options
    results_base_dir: str = "results"
    workflow_id: Optional[str] = None


class AgenticAnalysisResult(BaseModel):
    """Result from agentic analysis"""
    
    workflow_id: str
    query: str
    status: str  # "completed", "partial", "failed"
    
    # Agentic execution info
    initial_plan: Optional[WorkflowPlan] = None
    final_plan: Optional[WorkflowPlan] = None
    nodes_executed: List[str] = Field(default_factory=list)
    nodes_skipped: List[str] = Field(default_factory=list)
    
    # Results by stage
    stage_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Error handling
    execution_report: Optional[WorkflowExecutionReport] = None
    recovery_actions: List[str] = Field(default_factory=list)
    
    # Outputs
    results_directory: Optional[str] = None
    manifest_path: Optional[str] = None
    
    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_seconds: float = 0
    
    # Summary
    interpretation: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)


class AgenticContextAwareOrchestrator:
    """
    Advanced orchestrator combining:
    - Agentic workflow planning (LLM decides which nodes to run)
    - Context-aware agents (learn from past work)
    - Output management (organized result storage)
    - Error handling with retry (automatic recovery)
    
    This is the recommended orchestrator for production use.
    """
    
    def __init__(
        self,
        model: str = "openai:gpt-4",
        context_manager: Optional[ContextManager] = None,
        output_manager: Optional[OutputManager] = None,
        enable_agentic_workflow: bool = True,
        enable_error_recovery: bool = True,
        recovery_policy: Optional[WorkflowRecoveryPolicy] = None
    ):
        """
        Initialize agentic orchestrator.
        
        Args:
            model: LLM model to use for agents and planning
            context_manager: Context manager for memory (created if None)
            output_manager: Output manager for results (created per workflow)
            enable_agentic_workflow: Enable LLM-driven workflow planning
            enable_error_recovery: Enable automatic error recovery
            recovery_policy: Custom recovery policy
        """
        self.model = model
        
        # Initialize context manager if not provided
        if context_manager is None:
            self.context_manager = ContextManager(
                persist_directory="./context_storage",
                model=model
            )
            logger.info("✅ Created new ContextManager")
        else:
            self.context_manager = context_manager
        
        # Output manager is created per workflow
        self.base_output_manager = output_manager
        
        # Agentic controller
        self.enable_agentic_workflow = enable_agentic_workflow
        if enable_agentic_workflow:
            self.workflow_controller = AgenticWorkflowController(
                model=model,
                context_manager=self.context_manager
            )
        else:
            self.workflow_controller = None
        
        # Error recovery
        self.enable_error_recovery = enable_error_recovery
        self.recovery_policy = recovery_policy or self._create_default_recovery_policy()
        self.error_analyzer = ErrorAnalyzer()
        self.retry_manager = RetryManager(create_standard_retry_config())
        
        # Agents (created per workflow)
        self.agents: Dict[str, Any] = {}
        
        logger.info("🚀 AgenticContextAwareOrchestrator initialized")
        logger.info(f"   Model: {model}")
        logger.info(f"   Agentic workflow: {enable_agentic_workflow}")
        logger.info(f"   Error recovery: {enable_error_recovery}")
    
    def _create_default_recovery_policy(self) -> WorkflowRecoveryPolicy:
        """Create default recovery policy"""
        return WorkflowRecoveryPolicy(
            enable_node_retry=True,
            max_node_retries=3,
            allow_node_skip=True,
            use_cached_results=False,  # Fresh execution
            enable_fallback_agents=False,  # Single agent per type
            stop_on_critical_error=False,  # Try to continue
            max_failed_nodes=5,  # Allow some failures
            enable_checkpoints=True,
            checkpoint_interval=2,
            detailed_error_logging=True,
            save_error_reports=True
        )
    
    async def execute_analysis(
        self,
        request: AgenticAnalysisRequest
    ) -> AgenticAnalysisResult:
        """
        Execute analysis with agentic workflow planning.
        
        This is the main entry point that orchestrates:
        1. Session initialization
        2. Agentic workflow planning
        3. Agent creation
        4. Adaptive execution with error recovery
        5. Result collection and storage
        
        Args:
            request: Analysis request with query and options
            
        Returns:
            AgenticAnalysisResult with comprehensive execution info
        """
        start_time = datetime.now()
        workflow_id = request.workflow_id or f"analysis_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING AGENTIC ANALYSIS")
        logger.info("="*80)
        logger.info(f"Workflow ID: {workflow_id}")
        logger.info(f"Query: {request.query}")
        logger.info(f"Agentic workflow: {request.enable_adaptive_workflow}")
        logger.info(f"Error recovery: {request.enable_error_recovery}")
        
        # Initialize result
        result = AgenticAnalysisResult(
            workflow_id=workflow_id,
            query=request.query,
            status="in_progress",
            started_at=start_time
        )
        
        try:
            # 1. Start context session
            self.context_manager.start_session(
                topic=f"{request.query[:50]}..."
            )
            logger.info("✅ Context session started")
            
            # 2. Create output manager for this workflow
            output_manager = OutputManager(
                workflow_id=workflow_id,
                base_results_dir=request.results_base_dir
            )
            result.results_directory = str(output_manager.workflow_dir)
            logger.info(f"📁 Results directory: {output_manager.workflow_dir}")
            
            # 3. Create workflow graph
            workflow = self._create_workflow(request)
            logger.info(f"📊 Workflow created: {workflow.name} ({len(workflow.nodes)} nodes)")
            
            # 4. Create agentic execution plan
            if self.enable_agentic_workflow and request.enable_adaptive_workflow:
                initial_plan = await self.workflow_controller.create_initial_plan(
                    request.query,
                    request.data_source or {},
                    workflow
                )
                result.initial_plan = initial_plan
                current_plan = initial_plan
            else:
                # Non-agentic: execute all nodes
                current_plan = WorkflowPlan(
                    nodes_to_execute=list(workflow.nodes.keys()),
                    nodes_to_skip=[],
                    execution_order=list(workflow.nodes.keys()),
                    reasoning="Executing all nodes (non-agentic mode)",
                    estimated_stages=len(workflow.nodes)
                )
                result.initial_plan = current_plan
            
            # 5. Create agents
            self.agents = self._create_agents(self.context_manager, output_manager)
            logger.info(f"🤖 Created {len(self.agents)} agents")
            
            # 6. Execute workflow with agentic control
            execution_result = await self._execute_agentic_workflow(
                workflow,
                current_plan,
                request,
                output_manager
            )
            
            # 7. Update result
            result.status = execution_result["status"]
            result.nodes_executed = execution_result["nodes_executed"]
            result.nodes_skipped = execution_result["nodes_skipped"]
            result.stage_results = execution_result["stage_results"]
            result.execution_report = execution_result.get("execution_report")
            result.recovery_actions = execution_result.get("recovery_actions", [])
            result.final_plan = execution_result.get("final_plan")
            result.errors = execution_result.get("errors", [])
            
            # 8. Save manifest and create summary
            output_manager.save_manifest()
            output_manager.create_summary_report()
            result.manifest_path = str(output_manager.workflow_dir / "manifest.json")
            
            # 9. Get interpretation if available
            if "interpretation" in result.stage_results:
                result.interpretation = result.stage_results["interpretation"]
            
            # 10. Finalize timing
            result.completed_at = datetime.now()
            result.execution_time_seconds = (
                result.completed_at - start_time
            ).total_seconds()
            
            logger.info("\n" + "="*80)
            logger.info(f"✅ ANALYSIS COMPLETE: {result.status.upper()}")
            logger.info("="*80)
            logger.info(f"Execution time: {result.execution_time_seconds:.1f}s")
            logger.info(f"Nodes executed: {len(result.nodes_executed)}")
            logger.info(f"Nodes skipped: {len(result.nodes_skipped)}")
            logger.info(f"Results: {result.results_directory}")
            
            return result
            
        except Exception as e:
            error_msg = f"Critical error in orchestrator: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            result.status = "failed"
            result.errors.append(error_msg)
            result.completed_at = datetime.now()
            result.execution_time_seconds = (
                result.completed_at - start_time
            ).total_seconds()
            
            return result
    
    def _create_workflow(self, request: AgenticAnalysisRequest) -> WorkflowGraph:
        """Create workflow graph from request"""
        initial_data = {
            "query": request.query,
            "data_source": request.data_source,
            "objectives": request.objectives,
            "metadata": request.metadata
        }
        
        if request.workflow_type == "standard_regression":
            return create_standard_regression_workflow(initial_data=initial_data)
        elif request.workflow_type == "exploratory":
            return create_exploratory_workflow(initial_data=initial_data)
        elif request.workflow_type == "modeling_focused":
            return create_modeling_focused_workflow(initial_data=initial_data)
        else:
            raise ValueError(f"Unknown workflow type: {request.workflow_type}")
    
    def _create_agents(
        self,
        context_manager: ContextManager,
        output_manager: OutputManager
    ) -> Dict[str, Any]:
        """Create output-aware specialist agents"""
        agent_types = [
            "statistician",
            "data_engineer",
            "eda",
            "modeling",
            "interpreter"
        ]
        
        agents = {}
        for agent_type in agent_types:
            agent = OutputAwareAgentFactory.create_agent(
                agent_type,
                model=self.model,
                context_manager=context_manager,
                output_manager=output_manager
            )
            agents[f"{agent_type}_agent"] = agent
        
        return agents
    
    async def _execute_agentic_workflow(
        self,
        workflow: WorkflowGraph,
        plan: WorkflowPlan,
        request: AgenticAnalysisRequest,
        output_manager: OutputManager
    ) -> Dict[str, Any]:
        """
        Execute workflow with agentic control and error recovery.
        
        This is where the magic happens:
        - Nodes are executed according to the agentic plan
        - After each node, the controller decides whether to continue
        - Errors trigger retry and plan adaptation
        - Results are continuously evaluated
        """
        logger.info("\n" + "="*80)
        logger.info("🎯 EXECUTING AGENTIC WORKFLOW")
        logger.info("="*80)
        
        nodes_executed = []
        nodes_skipped = list(plan.nodes_to_skip)
        stage_results = {}
        recovery_actions = []
        errors = []
        current_plan = plan
        
        # Create executor for tracking
        executor = WorkflowExecutor(workflow)
        
        # Execute nodes according to plan
        for node_id in plan.execution_order:
            node = workflow.nodes[node_id]
            
            # Check if this node should be skipped per plan
            if node_id in plan.nodes_to_skip:
                logger.info(f"\n⏭️  Skipping {node.stage.value} (per agentic plan)")
                node.status = StageStatus.SKIPPED
                continue
            
            # Check dependencies
            if executor.can_skip_node(node_id):
                logger.warning(f"\n⏭️  Skipping {node.stage.value} (dependency failed)")
                node.status = StageStatus.SKIPPED
                nodes_skipped.append(node_id)
                continue
            
            # Execute node with retry
            logger.info(f"\n{'='*80}")
            logger.info(f"▶️  EXECUTING: {node.stage.value}")
            logger.info(f"{'='*80}")
            
            node_result = await self._execute_node_with_recovery(
                workflow, executor, node_id, len(nodes_executed), output_manager
            )
            
            # Track result
            stage_results[node.stage.value] = node_result
            
            if node_result.get("success", True):
                nodes_executed.append(node_id)
                logger.info(f"✅ {node.stage.value} completed successfully")
            else:
                error_msg = node_result.get("error", "Unknown error")
                errors.append(f"{node.stage.value}: {error_msg}")
                logger.error(f"❌ {node.stage.value} failed: {error_msg}")
                
                # Record recovery actions
                if "recovery_strategy" in node_result:
                    recovery_actions.append(
                        f"{node.stage.value}: {node_result['recovery_strategy']}"
                    )
                
                # If agentic, ask controller what to do
                if (self.enable_agentic_workflow and 
                    request.enable_adaptive_workflow):
                    
                    remaining_nodes = [
                        n for n in plan.execution_order 
                        if n not in nodes_executed and n not in nodes_skipped
                    ]
                    
                    if remaining_nodes:
                        # Should we adapt the plan?
                        adapted_plan = await self.workflow_controller.adapt_plan_for_error(
                            node_id, error_msg, current_plan, workflow
                        )
                        
                        if adapted_plan != current_plan:
                            logger.info("🔄 Plan adapted due to error")
                            current_plan = adapted_plan
                            recovery_actions.append(f"Adapted plan after {node.stage.value} failure")
            
            # After each node, check if we should continue
            if (self.enable_agentic_workflow and 
                request.enable_adaptive_workflow):
                
                remaining = [
                    n for n in plan.execution_order 
                    if n not in nodes_executed and n not in nodes_skipped
                ]
                
                if remaining:
                    decision = await self.workflow_controller.should_continue_after_node(
                        node_id, node_result, remaining, workflow
                    )
                    
                    if not decision["continue"]:
                        logger.info(f"🛑 Stopping workflow: {decision['reasoning']}")
                        recovery_actions.append(f"Stopped after {node.stage.value}: {decision['reasoning']}")
                        break
                    
                    # Skip nodes per decision
                    if decision["skip_nodes"]:
                        nodes_skipped.extend(decision["skip_nodes"])
                        recovery_actions.append(f"Skipping {len(decision['skip_nodes'])} nodes: {decision['reasoning']}")
        
        # Determine final status
        if errors:
            if nodes_executed:
                status = "partial"
            else:
                status = "failed"
        else:
            status = "completed"
        
        return {
            "status": status,
            "nodes_executed": nodes_executed,
            "nodes_skipped": nodes_skipped,
            "stage_results": stage_results,
            "recovery_actions": recovery_actions,
            "errors": errors,
            "final_plan": current_plan
        }
    
    async def _execute_node_with_recovery(
        self,
        workflow: WorkflowGraph,
        executor: WorkflowExecutor,
        node_id: str,
        stage_order: int,
        output_manager: OutputManager
    ) -> Dict[str, Any]:
        """
        Execute a single node with comprehensive error recovery.
        
        This wraps node execution with:
        - Automatic retry on failure
        - Error analysis and diagnosis
        - Recovery strategy selection
        - Result validation
        """
        node = workflow.nodes[node_id]
        
        # Define execution function for retry
        async def execute_node():
            node.status = StageStatus.IN_PROGRESS
            node.started_at = datetime.now()
            
            # Get agent
            agent = self.agents.get(node.agent_name)
            if not agent:
                raise ValueError(f"Agent not found: {node.agent_name}")
            
            # Prepare input
            input_data = executor.prepare_node_input(node_id)
            node.input_data = input_data
            
            # Execute agent
            output = await agent.execute_with_outputs(
                task=node.description,
                workflow_node=node,
                stage_order=stage_order,
                save_outputs=True
            )
            
            # Mark completed if successful
            if output.get("success", True):
                executor.mark_completed(node_id, output)
            else:
                executor.mark_failed(node_id, output.get("error", "Unknown error"))
            
            return output
        
        # Execute with retry if enabled
        if self.enable_error_recovery:
            try:
                retry_result = await self.retry_manager.execute_with_retry(
                    execute_node,
                    error_context={
                        "node_id": node_id,
                        "stage": node.stage.value,
                        "agent": node.agent_name
                    }
                )
                
                if retry_result.success:
                    result = retry_result.result
                    result["total_attempts"] = retry_result.total_attempts
                    if retry_result.total_attempts > 1:
                        result["recovery_strategy"] = "retry_successful"
                    return result
                else:
                    # Retry failed, return error
                    return {
                        "success": False,
                        "error": retry_result.final_error,
                        "total_attempts": retry_result.total_attempts,
                        "recovery_strategy": "retry_exhausted",
                        "agent": node.agent_name,
                        "stage": node.stage.value
                    }
                    
            except Exception as e:
                error_msg = f"Error executing node {node_id}: {str(e)}"
                logger.error(error_msg)
                executor.mark_failed(node_id, error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "agent": node.agent_name,
                    "stage": node.stage.value
                }
        else:
            # No retry, execute once
            try:
                return await execute_node()
            except Exception as e:
                error_msg = f"Error executing node {node_id}: {str(e)}"
                logger.error(error_msg)
                executor.mark_failed(node_id, error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "agent": node.agent_name,
                    "stage": node.stage.value
                }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


async def example_simple_agentic_analysis():
    """Example 1: Simple analysis with agentic workflow"""
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 1: Simple Agentic Analysis")
    logger.info("="*80)
    
    # Create orchestrator
    orchestrator = AgenticContextAwareOrchestrator(
        model="openai:gpt-4.1-mini",
        enable_agentic_workflow=True,
        enable_error_recovery=True
    )
    
    # Simple query - agentic controller will determine minimal workflow
    request = AgenticAnalysisRequest(
        query="What are the basic statistics of the sales data?",
        data_source={"type": "csv", "path": "data/sales_data.csv"},
        enable_adaptive_workflow=True,
        workflow_type="exploratory"
    )
    
    result = await orchestrator.execute_analysis(request)
    
    print(f"\n{'='*80}")
    print(f"RESULT: {result.status}")
    print(f"{'='*80}")
    print(f"Nodes executed: {len(result.nodes_executed)}")
    print(f"Nodes skipped: {len(result.nodes_skipped)}")
    print(f"Execution time: {result.execution_time_seconds:.1f}s")
    print(f"Results saved to: {result.results_directory}")
    
    if result.initial_plan:
        print(f"\nAgentic reasoning: {result.initial_plan.reasoning}")


async def example_complex_modeling_with_recovery():
    """Example 2: Complex modeling with error recovery"""
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: Complex Modeling with Recovery")
    logger.info("="*80)
    
    # Create orchestrator with aggressive recovery
    recovery_policy = WorkflowRecoveryPolicy(
        enable_node_retry=True,
        max_node_retries=5,
        allow_node_skip=True,
        stop_on_critical_error=False,
        max_failed_nodes=10
    )
    
    orchestrator = AgenticContextAwareOrchestrator(
        model="openai:gpt-4.1-mini",
        enable_agentic_workflow=True,
        enable_error_recovery=True,
        recovery_policy=recovery_policy
    )
    
    # Complex modeling query - will use full workflow
    request = AgenticAnalysisRequest(
        query="""Build a predictive model for sales revenue based on advertising spend.
        Include feature engineering, model comparison, and performance validation.""",
        data_source={"type": "csv", "path": "data/sales_data.csv"},
        enable_adaptive_workflow=True,
        enable_error_recovery=True,
        workflow_type="modeling_focused"
    )
    
    result = await orchestrator.execute_analysis(request)
    
    print(f"\n{'='*80}")
    print(f"RESULT: {result.status}")
    print(f"{'='*80}")
    print(f"Nodes executed: {len(result.nodes_executed)}")
    print(f"Recovery actions: {len(result.recovery_actions)}")
    
    if result.recovery_actions:
        print("\nRecovery actions taken:")
        for action in result.recovery_actions:
            print(f"  - {action}")


async def example_non_agentic_mode():
    """Example 3: Traditional non-agentic mode (execute all nodes)"""
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 3: Non-Agentic Mode (All Nodes)")
    logger.info("="*80)
    
    # Create orchestrator with agentic mode disabled
    orchestrator = AgenticContextAwareOrchestrator(
        model="openai:gpt-4.1-mini",
        enable_agentic_workflow=False,  # Traditional mode
        enable_error_recovery=True
    )
    
    request = AgenticAnalysisRequest(
        query="Analyze sales trends",
        data_source={"type": "csv", "path": "data/sales_data.csv"},
        enable_adaptive_workflow=False,  # Execute all nodes
        workflow_type="standard_regression"
    )
    
    result = await orchestrator.execute_analysis(request)
    
    print(f"\n{'='*80}")
    print(f"RESULT: {result.status}")
    print(f"{'='*80}")
    print(f"Nodes executed: {len(result.nodes_executed)} (all nodes)")


async def main():
    """Run examples"""
    
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # Run examples
    # await example_simple_agentic_analysis()
    # await example_complex_modeling_with_recovery()
    await example_non_agentic_mode()


if __name__ == "__main__":
    """
    USAGE GUIDE:
    
    1. Import the orchestrator:
       from datascienceagent.integration_example_agentic import (
           AgenticContextAwareOrchestrator,
           AgenticAnalysisRequest
       )
    
    2. Create orchestrator:
       orchestrator = AgenticContextAwareOrchestrator(
           model="openai:gpt-4",
           enable_agentic_workflow=True,  # LLM decides which nodes to run
           enable_error_recovery=True     # Automatic retry on errors
       )
    
    3. Run analysis:
       request = AgenticAnalysisRequest(
           query="Your analysis question",
           data_source={"type": "csv", "path": "data.csv"},
           enable_adaptive_workflow=True
       )
       result = await orchestrator.execute_analysis(request)
    
    FEATURES:
    - 🧠 Agentic: LLM intelligently decides which workflow nodes to execute
    - 🔄 Adaptive: Workflow changes based on intermediate results
    - 💾 Context-aware: Agents learn from past analyses
    - 📁 Output management: All results systematically organized
    - 🛡️ Error recovery: Automatic retry with exponential backoff
    - ⚡ Efficient: Skip unnecessary nodes to save time
    
    BENEFITS:
    - Simple queries execute in 2-3 stages instead of 8+
    - Failed nodes automatically retry with corrections
    - Context from past work improves results
    - All outputs saved and organized
    - Adaptive workflow adjusts to errors and results
    """
    
    asyncio.run(main())