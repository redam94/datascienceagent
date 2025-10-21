"""
Output-Context-Aware Base Agent

This module provides an enhanced agent base class that combines:
- Context awareness (retrieves relevant past work)
- Output management (saves all outputs systematically)
- Code execution with output capture
- Automatic result tracking

This is the recommended base class for all specialist agents.

Author: Data Science Agent System
Created: 2025-10-19
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import traceback

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from loguru import logger

from datascienceagent.core.context_aware_agent import (
    ContextAwareAgent,
    TargetedContext,
)
from datascienceagent.core.context_management import (
    ContextManager,
    ContextType,
)
from datascienceagent.core.output_manager import OutputManager
from datascienceagent.core.output_capture_code_executor import (
    OutputCapturingExecutor,
    ExecutionResult,
)
from datascienceagent.utils.model_utils import process_model


# ============================================================================
# ENHANCED AGENT WITH OUTPUT MANAGEMENT
# ============================================================================


class OutputContextAwareAgent(ContextAwareAgent):
    """
    Enhanced agent that combines context awareness with output management.
    
    Features:
    - All features from ContextAwareAgent (context retrieval, code execution)
    - Automatic output capture and saving via OutputManager
    - Systematic storage of code, console output, plots, data files
    - Execution tracking and manifest generation
    - Data passing between agents via saved outputs
    
    Usage:
        agent = OutputContextAwareAgent(
            name="my_agent",
            model="ollama:llama3.2",
            context_manager=context_mgr,
            output_manager=output_mgr
        )
        
        result = await agent.execute_with_outputs(
            task="Analyze the data",
            workflow_node=node,
            stage_order=1
        )
    """
    
    def __init__(
        self,
        name: str,
        model: str = "ollama:llama3.2",
        context_manager: Optional[ContextManager] = None,
        output_manager: Optional[OutputManager] = None,
        enable_code_execution: bool = True,
    ):
        """
        Initialize output-context-aware agent.
        
        Args:
            name: Agent name
            model: LLM model to use
            context_manager: Context manager for retrieving past work
            output_manager: Output manager for saving results
            enable_code_execution: Whether to enable code execution
        """
        # Initialize parent (ContextAwareAgent)
        super().__init__(
            name=name,
            model=model,
            context_manager=context_manager,
            enable_code_execution=enable_code_execution,
        )
        
        # Add output management
        self.output_manager = output_manager
        
        # Use output-capturing executor instead of basic executor
        if enable_code_execution and output_manager:
            self.output_executor = OutputCapturingExecutor(
                timeout_seconds=300,
                max_output_lines=10000
            )
            logger.info(f"✅ {name} initialized with output capture")
        else:
            self.output_executor = None
    
    async def execute_with_outputs(
        self,
        task: str,
        workflow_node: Any,
        stage_order: Optional[int] = None,
        save_outputs: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute agent task with full output management.
        
        This is the main method that combines:
        1. Context retrieval (from ContextAwareAgent)
        2. Task execution
        3. Output capture and saving (via OutputManager)
        
        Args:
            task: Description of task to execute
            workflow_node: Current workflow node
            stage_order: Optional stage order number for file organization
            save_outputs: Whether to save outputs (default True)
            **kwargs: Additional parameters
            
        Returns:
            Dict with:
                - success: bool
                - result: Task result
                - code: Generated code (if any)
                - execution: Execution result (if code was run)
                - outputs_saved_to: Path where outputs were saved
                - error: Error message (if failed)
        """
        stage_name = workflow_node.stage.value if hasattr(workflow_node, 'stage') else self.name
        
        logger.info(f"\n{'='*70}")
        logger.info(f"{self.name.upper()}: {task}")
        logger.info(f"{'='*70}")
        
        try:
            # 1. Get targeted context
            context = await self.get_targeted_context(
                task,
                workflow_node,
                include_history=(
                    workflow_node.context_scope == "include_history"
                    if hasattr(workflow_node, 'context_scope')
                    else False
                ),
            )
            
            # 2. Store that we're starting this task
            if self.context_manager:
                self.context_manager.store_agent_thought(
                    self.name,
                    f"Starting: {task}",
                    metadata={
                        "stage": stage_name,
                        "stage_order": stage_order,
                    },
                )
            
            # 3. Build prompt with context
            prompt = self._build_prompt_with_context(task, context)
            
            # 4. Execute with LLM
            result = await self.agent.run(prompt)
            result_text = str(result.output)
            
            # 5. Process result - extract and execute code if present
            output = {
                "success": True,
                "raw_result": result_text,
                "timestamp": datetime.now().isoformat(),
                "agent": self.name,
                "stage": stage_name,
            }
            
            # Check if result contains code
            code = self._extract_code(result_text)
            
            if code and self.enable_code_execution and self.output_executor:
                logger.info("📝 Code detected - executing with output capture...")
                
                # Prepare input data from workflow node
                input_data = self._prepare_execution_input_data(workflow_node)
                
                # Execute code with output capture
                if self.output_manager and save_outputs:
                    # Use integrated execution with automatic output saving
                    exec_result = await self.output_executor.execute_with_output_manager(
                        code=code,
                        stage_name=stage_name,
                        output_manager=self.output_manager,
                        stage_order=stage_order,
                        input_data=input_data,
                        code_filename=f"{stage_name}_code.py"
                    )
                else:
                    # Execute without output saving
                    exec_result = await self.output_executor.execute_code(
                        code=code,
                        input_data=input_data
                    )
                
                # Add execution results to output
                output["code"] = code
                output["execution"] = {
                    "success": exec_result.success,
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                    "error": exec_result.error,
                    "execution_time_seconds": exec_result.execution_time_seconds,
                    "generated_files": exec_result.generated_files,
                }
                
                # Store successful code in context
                if exec_result.success and self.context_manager:
                    self.context_manager.store_code(
                        code,
                        language="python",
                        agent_name=self.name,
                        purpose=task,
                        metadata={
                            "stage": stage_name,
                            "execution_time": exec_result.execution_time_seconds,
                            "files_generated": len(exec_result.generated_files),
                        },
                    )
                
                # Update success status
                output["success"] = exec_result.success
                if not exec_result.success:
                    output["error"] = exec_result.error
            
            # 6. Save additional outputs if output manager is available
            if self.output_manager and save_outputs:
                # Save the raw result as text
                self.output_manager.save_text(
                    stage_name,
                    result_text,
                    f"{stage_name}_result.txt",
                    stage_order
                )
                
                # Save structured output as JSON
                self.output_manager.save_json(
                    stage_name,
                    output,
                    f"{stage_name}_output.json",
                    stage_order
                )
                
                # Track where outputs were saved
                stage_dir = self.output_manager.get_stage_dir(stage_name, stage_order)
                output["outputs_saved_to"] = str(stage_dir)
                
                logger.info(f"💾 All outputs saved to: {stage_dir}")
            
            # 7. Store output in context
            if self.context_manager:
                self.context_manager.store_agent_output(
                    self.name,
                    output,
                    metadata={
                        "stage": stage_name,
                        "stage_order": stage_order,
                    },
                )
            
            logger.info(f"✅ {self.name} completed successfully")
            return output
            
        except Exception as e:
            error_msg = f"Error in {self.name}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"❌ {error_msg}")
            
            # Store error in context
            if self.context_manager:
                self.context_manager.store_agent_thought(
                    self.name,
                    error_msg,
                    metadata={"error": True, "stage": stage_name},
                )
            
            # Save error to outputs if manager available
            if self.output_manager and save_outputs:
                self.output_manager.save_text(
                    stage_name,
                    error_msg,
                    f"{stage_name}_error.txt",
                    stage_order
                )
            
            # Attempt diagnosis
            diagnosed = await self._diagnose_error(str(e), task, context)
            
            return {
                "success": False,
                "error": error_msg,
                "diagnosis": diagnosed,
                "agent": self.name,
                "stage": stage_name,
                "timestamp": datetime.now().isoformat(),
            }
    
    def _prepare_execution_input_data(
        self,
        workflow_node: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare input data for code execution from workflow node.
        
        This extracts data from previous stages that might be needed
        for code execution (e.g., DataFrames, model objects, etc.)
        
        Args:
            workflow_node: Workflow node with input_data
            
        Returns:
            Dictionary of variables to make available in code execution
        """
        if not hasattr(workflow_node, 'input_data'):
            return None
        
        input_data = {}
        node_input = workflow_node.input_data
        
        # Look for common data structures to pass through
        for key, value in node_input.items():
            # Look for DataFrames from previous stages
            if isinstance(value, dict):
                if 'data' in value:
                    # Previous stage might have saved a DataFrame
                    input_data[f"{key}_data"] = value['data']
                if 'metadata' in value:
                    input_data[f"{key}_metadata"] = value['metadata']
        
        return input_data if input_data else None
    
    async def execute_with_context(
        self,
        task: str,
        workflow_node: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Override parent method to use execute_with_outputs by default.
        
        This maintains backward compatibility while adding output management.
        """
        return await self.execute_with_outputs(
            task=task,
            workflow_node=workflow_node,
            **kwargs
        )


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


async def demo_output_context_aware_agent():
    """Demonstrate the output-context-aware agent"""
    from datascienceagent.core.context_management import ContextManager
    from datascienceagent.core.output_manager import OutputManager
    
    print("\n" + "="*70)
    print("OUTPUT-CONTEXT-AWARE AGENT DEMO")
    print("="*70)
    
    # Initialize managers
    context_mgr = ContextManager(
        persist_directory="./demo_context",
        model="ollama:llama3.2"
    )
    context_mgr.start_session(topic="demo_analysis")
    
    output_mgr = OutputManager(workflow_id="demo_workflow")
    
    # Create agent
    agent = OutputContextAwareAgent(
        name="demo_agent",
        model="ollama:llama3.2",
        context_manager=context_mgr,
        output_manager=output_mgr,
        enable_code_execution=True
    )
    
    # Mock workflow node
    class MockNode:
        def __init__(self):
            self.stage = type('Stage', (), {'value': 'demo_stage'})()
            self.input_data = {}
            self.required_context_types = [ContextType.PAST_ANALYSES]
            self.context_scope = "current_session"
    
    node = MockNode()
    
    # Execute a task with output capture
    print("\n📊 Executing task with output capture...")
    result = await agent.execute_with_outputs(
        task="""
        Generate a simple analysis:
        1. Create a sample dataset for regression modeling
        2. Calculate statistics and correlations between covariates
        3. Create a visualization
        4. Implement a regression model
        5. Evaluate model performance
        6. Summarize findings
        7. Save all outputs (data, stats, plot)
        
        Write Python code to do this.
        """,
        workflow_node=node,
        stage_order=1
    )
    
    print(f"\n✅ Task completed!")
    print(f"   Success: {result['success']}")
    print(f"   Outputs saved to: {result.get('outputs_saved_to', 'N/A')}")
    
    if result.get('execution'):
        exec_result = result['execution']
        print(f"   Code executed: {exec_result['success']}")
        print(f"   Files generated: {len(exec_result.get('generated_files', []))}")
        print(f"   Execution time: {exec_result['execution_time_seconds']:.2f}s")
    
    # Save manifest
    output_mgr.save_manifest()
    output_mgr.create_summary_report()
    
    print(f"\n📁 Complete results saved to: {output_mgr.workflow_dir}")
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_output_context_aware_agent())