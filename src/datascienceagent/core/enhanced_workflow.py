"""
Enhanced Workflow Orchestrator with Error Recovery

This module extends the base workflow orchestrator with:
- Node-level retry and recovery
- Workflow-level error handling
- Automatic fallback strategies
- Comprehensive error tracking and reporting
- Recovery checkpoints and resume capability

Author: Data Science Agent System
Created: 2025-10-19
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger

from datascienceagent.core.error_handling import (
    ErrorAnalyzer,
    ErrorCategory,
    ErrorSeverity,
    RecoveryStrategy as ErrorRecoveryStrategy
)
from datascienceagent.core.retry_strategies import (
    RetryManager,
    RetryConfig,
    RetryResult,
    create_standard_retry_config,
    create_aggressive_retry_config
)


class NodeRecoveryAction(str, Enum):
    """Actions to take when a node fails"""
    
    RETRY = "retry"                      # Retry the node
    SKIP = "skip"                        # Skip and continue
    USE_CACHED = "use_cached"            # Use cached result from previous run
    USE_FALLBACK = "use_fallback"        # Use fallback node/agent
    PROPAGATE_FAILURE = "propagate_failure"  # Stop workflow
    MANUAL_INTERVENTION = "manual_intervention"  # Wait for user input


class WorkflowRecoveryPolicy(BaseModel):
    """Policy for workflow-level error recovery"""
    
    # Node retry settings
    enable_node_retry: bool = True
    max_node_retries: int = 3
    node_retry_config: RetryConfig = Field(default_factory=create_standard_retry_config)
    
    # Recovery strategies
    allow_node_skip: bool = True
    use_cached_results: bool = True
    enable_fallback_agents: bool = True
    
    # Workflow control
    stop_on_critical_error: bool = True
    max_failed_nodes: int = 3  # Stop if more than N nodes fail
    
    # Checkpointing
    enable_checkpoints: bool = True
    checkpoint_interval: int = 1  # Checkpoint after every N nodes
    
    # Reporting
    detailed_error_logging: bool = True
    save_error_reports: bool = True


class NodeExecutionRecord(BaseModel):
    """Record of node execution attempts"""
    
    node_id: str
    stage_name: str
    attempts: int = 0
    success: bool = False
    
    # Timing
    first_attempt_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_time_seconds: float = 0
    
    # Results
    output_data: Optional[Dict[str, Any]] = None
    
    # Errors
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    final_error: Optional[str] = None
    
    # Recovery
    recovery_actions_taken: List[str] = Field(default_factory=list)
    used_fallback: bool = False
    used_cached_result: bool = False


class WorkflowExecutionReport(BaseModel):
    """Comprehensive workflow execution report"""
    
    workflow_id: str
    workflow_name: str
    
    # Status
    status: str  # "completed", "partial", "failed"
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_time_seconds: float = 0
    
    # Node statistics
    total_nodes: int
    successful_nodes: int = 0
    failed_nodes: int = 0
    skipped_nodes: int = 0
    
    # Execution records
    node_records: Dict[str, NodeExecutionRecord] = Field(default_factory=dict)
    
    # Errors
    critical_errors: List[Dict[str, Any]] = Field(default_factory=list)
    recoverable_errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Recovery statistics
    total_retries: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)


class EnhancedWorkflowExecutor:
    """
    Enhanced workflow executor with comprehensive error handling and recovery.
    
    Features:
    - Automatic node retry with exponential backoff
    - Intelligent error analysis and recovery decisions
    - Fallback agent strategies
    - Checkpoint and resume capability
    - Detailed execution reporting
    - Error aggregation and analysis
    
    This executor wraps the base WorkflowExecutor and adds robust
    error handling at both the node and workflow levels.
    """
    
    def __init__(
        self,
        workflow: Any,  # WorkflowGraph
        recovery_policy: Optional[WorkflowRecoveryPolicy] = None,
        agents: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced workflow executor.
        
        Args:
            workflow: WorkflowGraph to execute
            recovery_policy: Recovery policy configuration
            agents: Dictionary of available agents
        """
        self.workflow = workflow
        self.recovery_policy = recovery_policy or WorkflowRecoveryPolicy()
        self.agents = agents or {}
        
        # Initialize error analyzer
        self.error_analyzer = ErrorAnalyzer()
        
        # Initialize retry manager for nodes
        self.retry_manager = RetryManager(self.recovery_policy.node_retry_config)
        
        # Execution tracking
        self.execution_report = WorkflowExecutionReport(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            started_at=datetime.now(),
            total_nodes=len(workflow.nodes)
        )
        
        # Cached results for recovery
        self.result_cache: Dict[str, Any] = {}
        
        # Checkpoints
        self.checkpoints: List[Dict[str, Any]] = []
        
        logger.info(f"EnhancedWorkflowExecutor initialized for workflow: {workflow.name}")
        logger.info(f"  Recovery policy: node_retry={self.recovery_policy.enable_node_retry}, "
                   f"fallback={self.recovery_policy.enable_fallback_agents}")
    
    async def execute_workflow(self) -> WorkflowExecutionReport:
        """
        Execute the workflow with comprehensive error handling.
        
        Returns:
            WorkflowExecutionReport with detailed execution information
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING ENHANCED WORKFLOW EXECUTION: {self.workflow.name}")
        logger.info(f"{'='*70}")
        
        start_time = datetime.now()
        
        try:
            # Get execution order (topological sort)
            execution_levels = self._get_execution_order()
            
            logger.info(f"📋 Workflow has {len(execution_levels)} execution levels")
            
            # Execute each level
            for level_idx, node_ids in enumerate(execution_levels):
                logger.info(f"\n{'='*70}")
                logger.info(f"LEVEL {level_idx + 1}/{len(execution_levels)}: "
                           f"{len(node_ids)} node(s)")
                logger.info(f"{'='*70}")
                
                # Execute nodes in this level
                level_results = await self._execute_level(
                    node_ids,
                    level_idx
                )
                
                # Check if we should stop
                if self._should_stop_workflow():
                    logger.warning("⚠️  Stopping workflow due to too many failures")
                    break
                
                # Create checkpoint
                if self.recovery_policy.enable_checkpoints:
                    if (level_idx + 1) % self.recovery_policy.checkpoint_interval == 0:
                        self._create_checkpoint(level_idx)
            
            # Workflow completed
            self.execution_report.completed_at = datetime.now()
            self.execution_report.total_time_seconds = (
                self.execution_report.completed_at - start_time
            ).total_seconds()
            
            # Determine final status
            if self.execution_report.failed_nodes == 0:
                self.execution_report.status = "completed"
            elif self.execution_report.successful_nodes > 0:
                self.execution_report.status = "partial"
            else:
                self.execution_report.status = "failed"
            
            # Generate recommendations
            self._generate_recommendations()
            
            logger.info(f"\n{'='*70}")
            logger.info(f"WORKFLOW EXECUTION COMPLETE: {self.execution_report.status.upper()}")
            logger.info(f"{'='*70}")
            logger.info(f"  Successful: {self.execution_report.successful_nodes}/{self.execution_report.total_nodes}")
            logger.info(f"  Failed: {self.execution_report.failed_nodes}/{self.execution_report.total_nodes}")
            logger.info(f"  Skipped: {self.execution_report.skipped_nodes}/{self.execution_report.total_nodes}")
            logger.info(f"  Total retries: {self.execution_report.total_retries}")
            logger.info(f"  Time: {self.execution_report.total_time_seconds:.1f}s")
            
            return self.execution_report
        
        except Exception as e:
            logger.error(f"❌ Workflow execution failed with unexpected error: {e}")
            self.execution_report.status = "failed"
            self.execution_report.critical_errors.append({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return self.execution_report
    
    async def _execute_level(
        self,
        node_ids: List[str],
        level_idx: int
    ) -> Dict[str, Any]:
        """
        Execute all nodes in a level.
        
        Args:
            node_ids: IDs of nodes to execute
            level_idx: Level index
            
        Returns:
            Dictionary of results by node ID
        """
        results = {}
        
        for node_id in node_ids:
            node = self.workflow.nodes[node_id]
            
            logger.info(f"\n▶️  Executing node: {node.stage.value}")
            logger.info(f"   Description: {node.description}")
            
            # Execute node with error handling
            result = await self._execute_node_with_recovery(node_id, level_idx)
            results[node_id] = result
        
        return results
    
    async def _execute_node_with_recovery(
        self,
        node_id: str,
        level_idx: int
    ) -> Dict[str, Any]:
        """
        Execute a single node with comprehensive error handling and recovery.
        
        Args:
            node_id: Node ID to execute
            level_idx: Level index
            
        Returns:
            Node execution result
        """
        node = self.workflow.nodes[node_id]
        
        # Create execution record
        record = NodeExecutionRecord(
            node_id=node_id,
            stage_name=node.stage.value,
            first_attempt_at=datetime.now()
        )
        
        # Check if we should skip this node
        if self._should_skip_node(node_id):
            logger.warning(f"⏭️  Skipping {node.stage.value} (dependency failed)")
            record.recovery_actions_taken.append("skipped_due_to_dependency")
            self.execution_report.skipped_nodes += 1
            self.execution_report.node_records[node_id] = record
            return {"error": "Skipped due to failed dependency"}
        
        # Try to execute with retry
        if self.recovery_policy.enable_node_retry:
            result = await self._execute_with_retry(node_id, record)
        else:
            result = await self._execute_node_once(node_id, record)
        
        # Update record
        record.completed_at = datetime.now()
        record.total_time_seconds = (
            record.completed_at - record.first_attempt_at
        ).total_seconds()
        
        # Update workflow statistics
        if record.success:
            self.execution_report.successful_nodes += 1
            logger.info(f"   ✅ Completed in {record.total_time_seconds:.1f}s")
        else:
            self.execution_report.failed_nodes += 1
            logger.error(f"   ❌ Failed after {record.attempts} attempts")
            
            # Analyze final error
            if record.final_error:
                error_analysis = self.error_analyzer.analyze_error(
                    Exception(record.final_error),
                    context={"node_id": node_id, "stage": node.stage.value}
                )
                
                if error_analysis.severity == ErrorSeverity.CRITICAL:
                    self.execution_report.critical_errors.append({
                        "node_id": node_id,
                        "stage": node.stage.value,
                        "error": record.final_error,
                        "category": error_analysis.category.value,
                        "severity": error_analysis.severity.value
                    })
                else:
                    self.execution_report.recoverable_errors.append({
                        "node_id": node_id,
                        "stage": node.stage.value,
                        "error": record.final_error,
                        "category": error_analysis.category.value
                    })
        
        # Store record
        self.execution_report.node_records[node_id] = record
        
        return record.output_data or {"error": record.final_error}
    
    async def _execute_with_retry(
        self,
        node_id: str,
        record: NodeExecutionRecord
    ) -> bool:
        """Execute node with retry logic"""
        
        node = self.workflow.nodes[node_id]
        
        # Define execution function for retry manager
        async def execute_fn():
            return await self._execute_node_once(node_id, record)
        
        # Execute with retry
        retry_result = await self.retry_manager.execute_with_retry(
            execute_fn,
            error_context={
                "node_id": node_id,
                "stage": node.stage.value
            }
        )
        
        # Update record
        record.attempts = retry_result.total_attempts
        self.execution_report.total_retries += (retry_result.total_attempts - 1)
        
        if retry_result.success:
            record.success = True
            record.output_data = retry_result.result
            self.execution_report.successful_recoveries += 1
            record.recovery_actions_taken.append("retry_successful")
            return True
        else:
            record.success = False
            record.final_error = retry_result.final_error
            self.execution_report.failed_recoveries += 1
            
            # Try additional recovery strategies
            recovered = await self._try_recovery_strategies(node_id, record, retry_result)
            return recovered
    
    async def _execute_node_once(
        self,
        node_id: str,
        record: NodeExecutionRecord
    ) -> Dict[str, Any]:
        """
        Execute node once without retry.
        
        This is called by the retry manager.
        
        Raises:
            Exception if execution fails
        """
        node = self.workflow.nodes[node_id]
        
        # Get agent for this node
        agent = self.agents.get(node.agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {node.agent_name}")
        
        # Prepare node input
        input_data = self._prepare_node_input(node_id)
        node.input_data = input_data
        
        # Execute with agent
        try:
            output = await self._execute_agent_task(agent, node)
            record.errors.append({
                "attempt": record.attempts + 1,
                "success": True,
                "timestamp": datetime.now().isoformat()
            })
            return output
        
        except Exception as e:
            record.errors.append({
                "attempt": record.attempts + 1,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
    
    async def _try_recovery_strategies(
        self,
        node_id: str,
        record: NodeExecutionRecord,
        retry_result: RetryResult
    ) -> bool:
        """
        Try additional recovery strategies after retry fails.
        
        Args:
            node_id: Node ID
            record: Execution record
            retry_result: Result from retry attempts
            
        Returns:
            True if recovery successful, False otherwise
        """
        node = self.workflow.nodes[node_id]
        
        # Strategy 1: Use cached result
        if self.recovery_policy.use_cached_results and node_id in self.result_cache:
            logger.info("💾 Using cached result from previous successful run")
            record.output_data = self.result_cache[node_id]
            record.success = True
            record.used_cached_result = True
            record.recovery_actions_taken.append("used_cached_result")
            return True
        
        # Strategy 2: Use fallback agent
        if self.recovery_policy.enable_fallback_agents:
            fallback_result = await self._try_fallback_agent(node_id, record)
            if fallback_result:
                return True
        
        # Strategy 3: Skip if allowed and not critical
        if self.recovery_policy.allow_node_skip:
            error_analysis = self.error_analyzer.analyze_error(
                Exception(retry_result.final_error),
                context={"node_id": node_id}
            )
            
            if error_analysis.severity != ErrorSeverity.CRITICAL:
                logger.warning(f"⏭️  Skipping non-critical node {node.stage.value}")
                record.output_data = {"skipped": True, "reason": retry_result.final_error}
                record.recovery_actions_taken.append("skipped_non_critical")
                self.execution_report.skipped_nodes += 1
                return False  # Not successful, but workflow can continue
        
        return False
    
    async def _try_fallback_agent(
        self,
        node_id: str,
        record: NodeExecutionRecord
    ) -> bool:
        """Try using a fallback agent"""
        # Placeholder for fallback agent logic
        # In production, this would try alternative agents
        return False
    
    async def _execute_agent_task(
        self,
        agent: Any,
        node: Any
    ) -> Dict[str, Any]:
        """Execute agent task (placeholder - override with actual logic)"""
        # This would call the actual agent execution method
        # For now, just a placeholder
        raise NotImplementedError("Override with actual agent execution logic")
    
    def _prepare_node_input(self, node_id: str) -> Dict[str, Any]:
        """Prepare input data for node from dependencies and initial data"""
        node = self.workflow.nodes[node_id]
        input_data = {}
        
        # Add outputs from dependencies
        for dep_id in node.depends_on:
            dep_node = self.workflow.nodes[dep_id]
            if dep_node.output_data:
                input_data[f"{dep_node.stage.value}_output"] = dep_node.output_data
        
        # Add initial workflow data
        if node.needs_initial_data and self.workflow.initial_data:
            for key in node.initial_data_keys:
                if key in self.workflow.initial_data:
                    input_data[key] = self.workflow.initial_data[key]
        
        return input_data
    
    def _should_skip_node(self, node_id: str) -> bool:
        """Check if node should be skipped due to failed dependencies"""
        node = self.workflow.nodes[node_id]
        
        for dep_id in node.depends_on:
            dep_record = self.execution_report.node_records.get(dep_id)
            if dep_record and not dep_record.success:
                return True
        
        return False
    
    def _should_stop_workflow(self) -> bool:
        """Check if workflow should be stopped"""
        
        # Stop if too many nodes failed
        if (self.recovery_policy.stop_on_critical_error and 
            self.execution_report.failed_nodes >= self.recovery_policy.max_failed_nodes):
            return True
        
        # Stop if there's a critical error
        if self.recovery_policy.stop_on_critical_error and self.execution_report.critical_errors:
            return True
        
        return False
    
    def _get_execution_order(self) -> List[List[str]]:
        """Get execution order (topological sort by levels)"""
        from collections import deque
        
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
    
    def _create_checkpoint(self, level_idx: int):
        """Create a checkpoint for recovery"""
        checkpoint = {
            "level": level_idx,
            "timestamp": datetime.now().isoformat(),
            "completed_nodes": list(self.execution_report.node_records.keys()),
            "result_cache": self.result_cache.copy()
        }
        self.checkpoints.append(checkpoint)
        logger.info(f"💾 Checkpoint created at level {level_idx}")
    
    def _generate_recommendations(self):
        """Generate recommendations based on execution results"""
        recommendations = []
        
        # Check for repeated failures
        if self.execution_report.failed_recoveries > 2:
            recommendations.append(
                "Multiple retry failures detected. Consider reviewing error logs and "
                "adjusting retry configuration or fixing underlying issues."
            )
        
        # Check for critical errors
        if self.execution_report.critical_errors:
            recommendations.append(
                f"Found {len(self.execution_report.critical_errors)} critical error(s). "
                "These require immediate attention and manual intervention."
            )
        
        # Check execution time
        if self.execution_report.total_time_seconds > 600:  # > 10 minutes
            recommendations.append(
                "Workflow execution took longer than expected. Consider optimizing "
                "slow nodes or increasing resource allocation."
            )
        
        # Check success rate
        success_rate = (self.execution_report.successful_nodes / 
                       self.execution_report.total_nodes * 100)
        if success_rate < 80:
            recommendations.append(
                f"Low success rate ({success_rate:.1f}%). Review failed nodes and "
                "consider adjusting workflow configuration or input data."
            )
        
        self.execution_report.recommendations = recommendations


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logger.info("Enhanced Workflow Orchestrator module loaded")
    logger.info("Use this module to execute workflows with automatic error recovery")