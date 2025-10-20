"""
Integration Example: Enhanced Orchestrator with Error Handling

This example shows how to integrate the new error handling and retry system
with your existing workflow orchestrator and agents.

Author: Data Science Agent System
Created: 2025-10-19
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

# Import your existing components
from datascienceagent.context_aware_orchestrator import OutputAwareOrchestrator as WorkflowOrchestrator, AnalysisRequest
from datascienceagent.core.workflow_graph import WorkflowGraph, WorkflowNode

# Import new error handling components
from datascienceagent.core.error_handling import ErrorAnalyzer
from datascienceagent.core.retry_strategies import (
    RetryManager,
    RetryConfig,
    create_standard_retry_config,
    create_aggressive_retry_config
)
from datascienceagent.core.enhanced_code_executor import EnhancedCodeExecutor
from datascienceagent.core.enhanced_workflow import (
    EnhancedWorkflowExecutor,
    WorkflowRecoveryPolicy
)


# ============================================================================
# OPTION 1: ENHANCE EXISTING ORCHESTRATOR (MINIMAL CHANGES)
# ============================================================================

class EnhancedOrchestrator(WorkflowOrchestrator):
    """
    Drop-in replacement for WorkflowOrchestrator with error handling.
    
    This extends your existing orchestrator to add:
    - Automatic retry for failed nodes
    - Error diagnosis and recovery
    - Comprehensive execution reports
    """
    
    def __init__(
        self,
        context_manager: Optional[Any] = None,
        model: str = "openai:gpt-4",
        enable_error_recovery: bool = True,
        recovery_policy: Optional[WorkflowRecoveryPolicy] = None
    ):
        """
        Initialize enhanced orchestrator.
        
        Args:
            context_manager: Context manager instance
            model: LLM model to use
            enable_error_recovery: Enable automatic error recovery
            recovery_policy: Custom recovery policy (uses defaults if None)
        """
        # Initialize parent
        super().__init__(context_manager, model)
        
        # Add error handling components
        self.enable_error_recovery = enable_error_recovery
        self.recovery_policy = recovery_policy or self._create_default_policy()
        self.error_analyzer = ErrorAnalyzer()
        self.retry_manager = RetryManager(create_standard_retry_config())
        
        logger.info("EnhancedOrchestrator initialized with error recovery")
    
    def _create_default_policy(self) -> WorkflowRecoveryPolicy:
        """Create default recovery policy"""
        return WorkflowRecoveryPolicy(
            enable_node_retry=True,
            max_node_retries=3,
            allow_node_skip=True,
            use_cached_results=True,
            enable_fallback_agents=True,
            stop_on_critical_error=True,
            max_failed_nodes=3,
            enable_checkpoints=True,
            checkpoint_interval=2
        )
    
    async def _execute_level(
        self,
        workflow: WorkflowGraph,
        executor: Any,  # WorkflowExecutor
        node_ids: list,
        level_idx: int
    ) -> Dict[str, Any]:
        """
        Override to add error recovery for node execution.
        
        This wraps the original _execute_level with retry logic.
        """
        results = {}
        
        for node_id in node_ids:
            node = workflow.nodes[node_id]
            
            # Check if should skip
            if executor.can_skip_node(node_id):
                logger.warning(f"⏭️  Skipping {node.stage.value} (dependency failed)")
                results[node_id] = {"error": "Skipped due to dependency"}
                continue
            
            logger.info(f"\n▶️  Executing: {node.stage.value}")
            
            # Execute with error recovery
            if self.enable_error_recovery:
                result = await self._execute_node_with_recovery(
                    workflow, executor, node_id, level_idx
                )
            else:
                result = await self._execute_node_original(
                    workflow, executor, node_id, level_idx
                )
            
            results[node_id] = result
        
        return results
    
    async def _execute_node_with_recovery(
        self,
        workflow: WorkflowGraph,
        executor: Any,
        node_id: str,
        level_idx: int
    ) -> Dict[str, Any]:
        """
        Execute node with automatic retry and error recovery.
        """
        node = workflow.nodes[node_id]
        
        # Define execution function for retry manager
        async def execute_fn():
            return await self._execute_node_once(workflow, executor, node_id, level_idx)
        
        # Execute with retry
        retry_result = await self.retry_manager.execute_with_retry(
            execute_fn,
            error_context={
                "node_id": node_id,
                "stage": node.stage.value,
                "workflow_id": workflow.id
            }
        )
        
        if retry_result.success:
            logger.info(f"   ✅ Success (attempts: {retry_result.total_attempts})")
            return retry_result.result
        else:
            # Log error analysis
            logger.error(f"   ❌ Failed after {retry_result.total_attempts} attempts")
            logger.error(f"   Reason: {retry_result.gave_up_reason}")
            
            # Analyze final error
            if retry_result.final_error:
                error_analysis = self.error_analyzer.analyze_error(
                    Exception(retry_result.final_error),
                    context={"node_id": node_id, "stage": node.stage.value}
                )
                
                logger.error(f"   Category: {error_analysis.category.value}")
                logger.error(f"   Severity: {error_analysis.severity.value}")
                logger.info(f"   💡 Suggested fixes:")
                for fix in error_analysis.suggested_fixes:
                    logger.info(f"      - {fix}")
            
            # Mark as failed
            executor.mark_failed(node_id, retry_result.final_error)
            
            return {"error": retry_result.final_error}
    
    async def _execute_node_once(
        self,
        workflow: WorkflowGraph,
        executor: Any,
        node_id: str,
        level_idx: int
    ) -> Dict[str, Any]:
        """
        Execute node once without retry (called by retry manager).
        
        Raises exception if fails.
        """
        node = workflow.nodes[node_id]
        
        try:
            node.status = "in_progress"
            node.started_at = asyncio.get_event_loop().time()
            
            # Get agent
            agent = self.agents.get(node.agent_name)
            if not agent:
                raise ValueError(f"Agent not found: {node.agent_name}")
            
            # Prepare input
            input_data = executor.prepare_node_input(node_id)
            node.input_data = input_data
            
            # Calculate stage order
            stage_order = (level_idx * 10) + 1
            
            # Execute with agent
            output = await self._execute_node(agent, node)
            
            # Mark completed
            executor.mark_completed(node_id, output)
            
            return output
        
        except Exception as e:
            # Re-raise for retry manager to handle
            raise


# ============================================================================
# OPTION 2: USE ENHANCED CODE EXECUTOR IN AGENTS
# ============================================================================

def create_enhanced_agents(context_manager, output_manager, model):
    """
    Create agents with enhanced code execution capabilities.
    
    This replaces the default code executor with the enhanced version
    that includes automatic error correction and retry.
    """
    from datascienceagent.agents.specialist_agents import AgentFactory
    
    agents = {}
    agent_types = ["statistician", "data_engineer", "eda", "modeling", "interpreter"]
    
    for agent_type in agent_types:
        # Create agent
        agent = AgentFactory.create_agent(
            agent_type,
            model=model,
            context_manager=context_manager
        )
        
        # Replace code executor with enhanced version
        if hasattr(agent, 'output_executor'):
            enhanced_executor = EnhancedCodeExecutor(
                timeout_seconds=300,
                enable_auto_correction=True,
                max_correction_attempts=2
            )
            agent.enhanced_executor = enhanced_executor
            logger.info(f"✅ {agent_type} agent enhanced with error recovery")
        
        agents[f"{agent_type}_agent"] = agent
    
    return agents


# ============================================================================
# OPTION 3: COMPLETE INTEGRATION EXAMPLE
# ============================================================================

async def run_analysis_with_error_handling(
    query: str,
    data_source: str,
    results_dir: str = "./results"
):
    """
    Complete example of running analysis with comprehensive error handling.
    
    This shows:
    1. Creating enhanced orchestrator
    2. Configuring error recovery
    3. Running analysis
    4. Handling and reporting errors
    """
    
    logger.info("="*70)
    logger.info("STARTING ANALYSIS WITH ERROR HANDLING")
    logger.info("="*70)
    
    # 1. Create enhanced orchestrator with custom recovery policy
    recovery_policy = WorkflowRecoveryPolicy(
        enable_node_retry=True,
        max_node_retries=3,
        node_retry_config=create_aggressive_retry_config(),
        allow_node_skip=True,
        use_cached_results=True,
        enable_fallback_agents=True,
        stop_on_critical_error=True,
        max_failed_nodes=3,
        enable_checkpoints=True,
        checkpoint_interval=1,
        detailed_error_logging=True,
        save_error_reports=True
    )
    
    orchestrator = EnhancedOrchestrator(
        model="openai:gpt-4.1-mini",
        enable_error_recovery=True,
        recovery_policy=recovery_policy
    )
    
    # 2. Create analysis request
    request = AnalysisRequest(
        query=query,
        workflow_type="standard_regression",
        data_source={"type": "csv", "path": data_source},
        objectives=["understand_patterns", "build_model", "predict"],
        metadata={"results_dir": results_dir}
    )
    
    # 3. Execute with comprehensive error handling
    try:
        result = await orchestrator.execute_analysis(request)
        
        # 4. Report results
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*70)
        logger.info(f"Status: {result.status}")
        logger.info(f"Workflow ID: {result.workflow_id}")
        
        # Success metrics
        if result.status == "completed":
            logger.info("✅ All stages completed successfully")
        elif result.status == "partial":
            logger.warning("⚠️  Partial completion - some stages failed")
        else:
            logger.error("❌ Analysis failed")
        
        # Error summary
        if result.errors:
            logger.warning(f"\n⚠️  Encountered {len(result.errors)} errors:")
            for i, error in enumerate(result.errors, 1):
                logger.warning(f"  {i}. {error}")
        
        # Stage results
        logger.info(f"\nStage Results:")
        for stage, stage_result in result.stage_results.items():
            status = "✅" if "error" not in stage_result else "❌"
            logger.info(f"  {status} {stage}")
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Unexpected error in orchestrator: {e}")
        
        # Analyze error
        error_analyzer = ErrorAnalyzer()
        analysis = error_analyzer.analyze_error(e)
        
        logger.error(f"Error category: {analysis.category.value}")
        logger.error(f"Severity: {analysis.severity.value}")
        logger.error(f"Root cause: {analysis.root_cause}")
        
        if analysis.suggested_fixes:
            logger.info("\n💡 Suggested fixes:")
            for fix in analysis.suggested_fixes:
                logger.info(f"  - {fix}")
        
        raise


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def example_1_basic_usage():
    """Example 1: Basic usage with default settings"""
    
    # Create enhanced orchestrator (uses default recovery policy)
    orchestrator = EnhancedOrchestrator(model="openai:gpt-4")
    
    # Run analysis
    request = AnalysisRequest(
        query="Analyze sales data and predict future revenue",
        data_source={"type": "csv", "path": "data/sales_data.csv"}
    )
    
    result = await orchestrator.execute_analysis(request)
    print(f"Status: {result.status}")


async def example_2_custom_policy():
    """Example 2: Custom recovery policy"""
    
    # Create strict recovery policy (no skipping, stop on first error)
    strict_policy = WorkflowRecoveryPolicy(
        enable_node_retry=True,
        max_node_retries=5,
        allow_node_skip=False,
        stop_on_critical_error=True,
        max_failed_nodes=1
    )
    
    orchestrator = EnhancedOrchestrator(
        model="openai:gpt-4.1-mini",
        recovery_policy=strict_policy
    )
    
    # Run analysis
    request = AnalysisRequest(
        query="Critical analysis - must complete all stages",
        data_source={"type": "csv", "path": "data/sales_data.csv"}
    )
    
    result = await orchestrator.execute_analysis(request)
    print(f"Status: {result.status}")


async def example_3_with_agents():
    """Example 3: Using enhanced agents with auto-correction"""
    
    from datascienceagent.core.context_management import ContextManager
    from datascienceagent.core.output_manager import OutputManager
    
    # Create managers
    context_manager = ContextManager(persist_directory="./context_storage")
    output_manager = OutputManager(base_results_dir="./results")
    
    # Create enhanced agents
    agents = create_enhanced_agents(context_manager, output_manager, "openai:gpt-4.1-mini")
    
    # Create orchestrator with enhanced agents
    orchestrator = EnhancedOrchestrator(
        context_manager=context_manager,
        model="openai:gpt-4.1-mini",
    )
    orchestrator.agents = agents  # Use enhanced agents
    
    # Run analysis
    request = AnalysisRequest(
        query="Analyze with auto-correcting code execution",
        data_source={"type": "csv", "path": "data/sales_data.csv"}
    )
    
    result = await orchestrator.execute_analysis(request)
    print(f"Status: {result.status}")


async def main():
    """Main example runner"""
    
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # Run complete example
    await run_analysis_with_error_handling(
        query="Analyze media effect on sales and build predictive model",
        data_source="data/sales_data.csv",
        results_dir="./results/media_analysis"
    )


if __name__ == "__main__":
    """
    To use this integration:
    
    1. Import the EnhancedOrchestrator instead of WorkflowOrchestrator:
       
       from datascienceagent.examples.integration_example import EnhancedOrchestrator
    
    2. Use it as a drop-in replacement:
       
       orchestrator = EnhancedOrchestrator(
           model="openai:gpt-4",
           enable_error_recovery=True
       )
    
    3. Run your analysis as usual:
       
       result = await orchestrator.execute_analysis(request)
    
    The enhanced orchestrator will automatically:
    - Retry failed nodes
    - Analyze and diagnose errors
    - Apply recovery strategies
    - Provide detailed error reports
    """
    
    asyncio.run(main())