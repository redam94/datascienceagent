"""
Context-Aware Base Agent with Code Execution

This module provides the base class for all specialist agents with:
- Targeted context retrieval (avoids overwhelming context window)
- Code execution capabilities with sandboxing
- Automatic result tracking and storage
- Error diagnosis and recovery
"""

import asyncio
import tempfile
import os
import sys
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import traceback

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from loguru import logger


from datascienceagent.core.code_executor import CodeExecutor, ExecutionResult
from datascienceagent.utils.model_utils import process_model
from datascienceagent.core.context_management import ContextManager, ContextType


# ============================================================================
# CONTEXT-AWARE BASE AGENT
# ============================================================================


# class ContextType(str, Enum):
#     """Types of context"""

#     USER_QUERY = "user_query"
#     AGENT_THOUGHT = "agent_thought"
#     AGENT_OUTPUT = "agent_output"
#     CODE_CHUNK = "code_chunk"
#     DATA_SUMMARY = "data_summary"
#     MODEL_RESULT = "model_result"
#     INTERPRETATION = "interpretation"
#     ERROR = "error"
#     EDA_INSIGHT = "eda_insight"
#     STATISTICAL_PLAN = "statistical_plan"


class TargetedContext(BaseModel):
    """
    Targeted context for an agent - only includes relevant information
    to avoid overwhelming the context window.
    """

    # Core information
    task_description: str
    session_summary: str = ""

    # Targeted context (limited to most relevant)
    relevant_findings: List[str] = Field(default_factory=list, max_length=5)
    relevant_code_patterns: List[str] = Field(default_factory=list, max_length=3)
    relevant_errors: List[str] = Field(default_factory=list, max_length=3)

    # Stage-specific context
    previous_stage_output: Optional[Dict[str, Any]] = None

    # Metadata
    total_context_chunks: int = 0
    context_retrieved_at: datetime = Field(default_factory=datetime.now)


class ContextAwareAgent:
    """
    Base class for all specialist agents with context awareness and code execution.

    Features:
    - Targeted context retrieval (only gets relevant context)
    - Code execution with sandboxing
    - Automatic error diagnosis
    - Result tracking and storage
    """

    def __init__(
        self,
        name: str,
        model: str = "openai:gpt-4",
        context_manager: Optional[ContextManager] = None,
        enable_code_execution: bool = True,
    ):
        self.name = name
        self.model = process_model(model)
        self.context_manager = context_manager
        self.enable_code_execution = enable_code_execution

        # Initialize code executor
        if enable_code_execution:
            self.code_executor = CodeExecutor(timeout_seconds=90, max_memory_mb=1024)

        # Initialize agent with system prompt
        self.agent = Agent(self.model, system_prompt=self.get_system_prompt())

    def get_system_prompt(self) -> str:
        """
        Override this in subclasses to provide agent-specific prompts.
        """
        return f"""You are {self.name}, a specialist data science agent.
        
You have access to:
- Targeted context from previous work
- Code execution capabilities
- Error diagnosis tools

When you need to write code:
1. Generate clean, well-documented code
2. Use the execute_code function to run it
3. If errors occur, diagnose and fix them
4. Store successful patterns for future use

Keep responses focused and concise."""

    async def get_targeted_context(
        self, task: str, workflow_node: Any, include_history: bool = False
    ) -> TargetedContext:
        """
        Retrieve only the context relevant for this specific task.

        This prevents overwhelming the agent's context window with
        irrelevant information.

        Args:
            task: Description of current task
            workflow_node: Current workflow node with context requirements
            include_history: Whether to include historical context

        Returns:
            TargetedContext with only relevant information
        """
        if self.context_manager is None:
            return TargetedContext(task_description=task)

        # Build targeted context
        targeted_context = TargetedContext(task_description=task)

        # Get session summary (brief overview)
        if hasattr(self.context_manager, "get_session_summary"):
            summary = await self.context_manager.get_session_summary()
            # Truncate to avoid overwhelming
            targeted_context.session_summary = summary[:500] if summary else ""

        # Get only relevant findings for this stage
        required_types = workflow_node.required_context_types if workflow_node else []

        if required_types:
            for ctx_type in required_types[:3]:  # Limit to top 3 types
                results = await self.context_manager.get_relevant_context(
                    query_text=task,
                    context_types=[ctx_type],
                    limit=2,  # Only get top 2 most relevant per type
                )

                for result in results:
                    content = result.get("content", "")
                    if len(targeted_context.relevant_findings) < 5:
                        # Truncate long content
                        truncated = content[:200] if len(content) > 200 else content
                        targeted_context.relevant_findings.append(truncated)

        # Get relevant code patterns if this agent writes code
        if self.enable_code_execution and include_history:
            code_results = await self.context_manager.get_relevant_context(
                query_text=task, context_types=[ContextType.CODE_CHUNK], limit=3
            )

            for result in code_results:
                pattern = result.get("content", "")
                if len(targeted_context.relevant_code_patterns) < 3:
                    # Extract key pattern (truncate)
                    truncated = pattern[:150] if len(pattern) > 150 else pattern
                    targeted_context.relevant_code_patterns.append(truncated)

        # Get previous stage output if available
        if workflow_node and workflow_node.input_data:
            targeted_context.previous_stage_output = workflow_node.input_data

        return targeted_context

    async def execute_with_context(
        self, task: str, workflow_node: Any, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute agent task with targeted context.

        Args:
            task: Description of task to execute
            workflow_node: Current workflow node
            **kwargs: Additional parameters

        Returns:
            Dict with results
        """
        # Get targeted context (only relevant information)
        context = await self.get_targeted_context(
            task,
            workflow_node,
            include_history=(workflow_node.context_scope == "include_history"),
        )

        # Store that we're starting this task
        if self.context_manager:
            self.context_manager.store_agent_thought(
                self.name,
                f"Starting: {task}",
                metadata={
                    "stage": workflow_node.stage.value if workflow_node else "unknown"
                },
            )

        # Build prompt with targeted context
        prompt = self._build_prompt_with_context(task, context)

        # Execute
        try:
            result = await self.agent.run(prompt)

            # Parse result
            output = await self._process_result(str(result.output), workflow_node)

            # Store output
            if self.context_manager:
                self.context_manager.store_agent_output(
                    self.name,
                    output,
                    metadata={
                        "stage": (
                            workflow_node.stage.value if workflow_node else "unknown"
                        )
                    },
                )

            return output

        except Exception as e:
            error_msg = f"Error in {self.name}: {str(e)}\n{traceback.format_exc()}"

            # Store error
            if self.context_manager:
                self.context_manager.store_agent_thought(
                    self.name, error_msg, metadata={"error": True}
                )

            # Attempt diagnosis
            diagnosed = await self._diagnose_error(str(e), task, context)

            return {"success": False, "error": error_msg, "diagnosis": diagnosed}

    def _build_prompt_with_context(self, task: str, context: TargetedContext) -> str:
        """Build prompt with targeted context"""
        prompt_parts = [f"TASK: {task}\n"]

        if context.session_summary:
            prompt_parts.append(f"SESSION CONTEXT:\n{context.session_summary}\n")

        if context.relevant_findings:
            prompt_parts.append("RELEVANT FINDINGS:")
            for finding in context.relevant_findings:
                prompt_parts.append(f"- {finding}")
            prompt_parts.append("")

        if context.relevant_code_patterns:
            prompt_parts.append("SUCCESSFUL CODE PATTERNS:")
            for i, pattern in enumerate(context.relevant_code_patterns, 1):
                prompt_parts.append(f"{i}. {pattern}")
            prompt_parts.append("")

        if context.previous_stage_output:
            prompt_parts.append(f"PREVIOUS STAGE OUTPUT:")
            prompt_parts.append(str(context.previous_stage_output)[:300])
            prompt_parts.append("")

        return "\n".join(prompt_parts)

    async def _process_result(
        self, result_text: str, workflow_node: Any
    ) -> Dict[str, Any]:
        """
        Process agent result - extract code if present and execute it.
        """
        output = {
            "raw_result": result_text,
            "success": True,
            "timestamp": datetime.now().isoformat(),
        }

        # Check if result contains code
        code = self._extract_code(result_text)
        logger.debug(f"Extracted code: {code}")
        if code and self.enable_code_execution:
            # Execute code
            exec_result = await self.code_executor.execute_code(code)

            output["code"] = code
            output["execution_result"] = {
                "success": exec_result.success,
                "stdout": exec_result.stdout,
                "stderr": exec_result.stderr,
                "error": exec_result.error,
                "execution_time": exec_result.execution_time_seconds,
            }

            # Store successful code
            if exec_result.success and self.context_manager:
                self.context_manager.store_code(
                    code,
                    language="python",
                    agent_name=self.name,
                    purpose=workflow_node.description if workflow_node else "unknown",
                    metadata={"execution_time": exec_result.execution_time_seconds},
                )

        return output

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from markdown code blocks"""
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
        return None

    async def _diagnose_error(
        self, error: str, task: str, context: TargetedContext
    ) -> str:
        """
        Diagnose an error and suggest fixes.
        """
        diagnosis_prompt = f"""An error occurred while executing: {task}

ERROR:
{error}

CONTEXT:
{context.session_summary[:200]}

Diagnose the error and suggest how to fix it.
Be specific and actionable."""

        try:
            result = await self.agent.run(diagnosis_prompt)
            return str(result.output)
        except:
            return "Unable to diagnose error automatically"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    async def demo():
        # Create agent
        agent = ContextAwareAgent(
            name="demo_agent", model="ollama:llama3.2", enable_code_execution=True,
            context_manager=ContextManager(persist_directory="./context_storage", chunk_size=500, model="ollama:llama3.2")
        )

        # Test code execution
        code = """
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

print(f"Created dataframe with shape: {data.shape}")
print(f"Correlation: {data.corr().iloc[0, 1]:.3f}")
"""

        result = await agent.code_executor.execute_code(code)
        print(f"Execution success: {result.success}")
        print(f"Output:\n{result.stdout}")

    asyncio.run(demo())
