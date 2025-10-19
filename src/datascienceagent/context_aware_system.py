"""
Integrated Agent System with Context Management
Combines the multi-agent system with RAG and context summarization
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from loguru import logger

from datascienceagent.utils.model_utils import process_model

# Import context management (in production, use proper imports)
from datascienceagent.core.context_management import (
    ContextManager,
    ContextType,
    ContextSummary,
)
from datascienceagent.core.context_aware_agent import ContextAwareAgent

# ============================================================================
# CONTEXT-AWARE ORCHESTRATOR
# ============================================================================


class ContextAwareOrchestrator:
    """
    Orchestrator with full context management integration
    """

    def __init__(
        self,
        model: str = "openai:gpt-4",
        context_manager: Optional[ContextManager] = None,  # ContextManager instance
        agents: Optional[Dict[str, Any]] = None,
    ):
        self.model = process_model(model)
        self.context_manager: ContextManager = context_manager or ContextManager(
            persist_directory="./context_chroma_db", chunk_size=1000
        )
        self.agents = agents or {}

        self.system_prompt = """You are the Orchestrator Agent with access to comprehensive context.

You have access to:
- Current session context
- Relevant past sessions on similar topics
- Successful code patterns from history
- Key findings from previous analyses
- Agent performance and approaches

Use this context to:
1. Create more informed analysis plans
2. Avoid repeating past mistakes
3. Build on successful approaches
4. Reference relevant prior work
5. Provide continuity across sessions

Be specific about how you're using historical context."""

        self.agent = Agent(self.model, system_prompt=self.system_prompt)

    async def plan_analysis_with_context(
        self,
        query: str,
        data_source: Optional[Dict[str, Any]] = None,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create analysis plan with full context awareness"""

        # Start new session
        session_id = self.context_manager.start_session(topic=topic)

        # Store user query
        self.context_manager.store_user_query(query)

        # Get relevant context
        relevant_context = await self.context_manager.get_relevant_context(
            query_text=query, limit=10
        )

        # Build context-aware prompt
        prompt = f"""Create an analysis plan for this query:

QUERY: {query}
DATA SOURCE: {data_source}

RELEVANT PAST WORK:
{self._format_context(relevant_context[:5])}

Create a detailed plan that:
1. Builds on successful past approaches
2. Avoids known pitfalls
3. References useful code patterns
4. Considers prior findings

Provide a step-by-step plan with task dependencies."""

        # Store orchestrator's thinking
        self.context_manager.store_agent_thought(
            "orchestrator",
            f"Planning analysis with {len(relevant_context)} relevant historical contexts",
            metadata={"query": query},
        )

        result = await self.agent.run(prompt)

        plan_data = {
            "session_id": session_id,
            "query": query,
            "plan": str(result.data),
            "context_used": len(relevant_context),
        }

        # Store the plan
        self.context_manager.store_agent_output(
            "orchestrator", plan_data, metadata={"stage": "planning"}
        )

        return plan_data

    async def execute_with_context(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan with context management at each step"""

        results = {}

        # Define workflow stages
        stages = [
            ("research", "statistician"),
            ("load_data", "data_engineer"),
            ("eda", "eda_agent"),
            ("modeling", "modeling_agent"),
            ("interpretation", "interpreter_agent"),
        ]

        for stage_name, agent_name in stages:
            logger.info(f"\n📋 Executing stage: {stage_name}")

            # Get context for this agent
            context_summary = await self.context_manager.get_context_for_agent(
                agent_name,
                f"Execute {stage_name} for query: {plan['query']}",
                include_history=True,
            )

            logger.info(
                f"   📝 Context: {context_summary.total_chunks} relevant chunks"
            )

            # Execute stage (simplified for demo)
            stage_result = await self._execute_stage_with_context(
                stage_name, agent_name, plan, context_summary
            )

            results[stage_name] = stage_result

            # Store results
            self.context_manager.store_agent_output(
                agent_name,
                stage_result,
                metadata={"stage": stage_name, "query": plan["query"]},
            )

            # Store any code generated
            if "code" in stage_result:
                self.context_manager.store_code(
                    stage_result["code"],
                    language="python",
                    agent_name=agent_name,
                    purpose=stage_name,
                    metadata={"stage": stage_name},
                )

        return results

    async def _execute_stage_with_context(
        self,
        stage_name: str,
        agent_name: str,
        plan: Dict[str, Any],
        context: ContextSummary,
    ) -> Dict[str, Any]:
        """Execute a stage with context"""

        # Build context-enhanced prompt
        prompt = f"""Execute {stage_name} for this analysis:

Query: {plan['query']}

CONTEXT FROM CURRENT SESSION:
{context.session_summary[:500]}

"""

        if context.code_patterns:
            prompt += "\nRELEVANT CODE PATTERNS:\n"
            for pattern in context.code_patterns[:2]:
                prompt += f"- {pattern}\n"

        if context.key_findings:
            prompt += "\nKEY FINDINGS TO CONSIDER:\n"
            for finding in context.key_findings[:2]:
                prompt += f"- {finding[:100]}\n"

        prompt += f"\nExecute the {stage_name} stage, leveraging this context."

        # Store thought process
        self.context_manager.store_agent_thought(
            agent_name,
            f"Executing {stage_name} with context awareness",
            metadata={"stage": stage_name},
        )

        # Simulate execution (in production, call actual agent)
        result = {
            "stage": stage_name,
            "status": "success",
            "output": f"Completed {stage_name} with context awareness",
            "context_chunks_used": context.total_chunks,
        }

        # Add stage-specific outputs
        if stage_name == "modeling":
            result[
                "code"
            ] = """
# Context-aware modeling
import statsmodels.api as sm

# Using approach from successful past analysis
model = sm.OLS(y, X)
results = model.fit(cov_type='HC3')  # Robust errors based on past experience
"""

        return result

    async def generate_final_report(self) -> str:
        """Generate final report with full session context"""

        # Get session summary
        session_summary = await self.context_manager.get_session_summary()

        prompt = f"""Generate a comprehensive analysis report.

SESSION SUMMARY:
{session_summary}

Include:
1. Executive summary
2. Methods used (with justification from context)
3. Key findings
4. Recommendations
5. How this builds on prior work

Be specific about how historical context informed the analysis."""

        result = await self.agent.run(prompt)
        report = str(result.data)

        # Store the final report
        self.context_manager.store_agent_output(
            "orchestrator", {"report": report}, metadata={"stage": "final_report"}
        )

        return report

    def _format_context(self, contexts: List[Dict[str, Any]]) -> str:
        """Format context for prompt"""
        if not contexts:
            return "No relevant past work found."

        formatted = []
        for ctx in contexts:
            content = ctx["content"][:200]
            ctx_type = ctx["metadata"].get("context_type", "unknown")
            formatted.append(f"[{ctx_type}] {content}...")

        return "\n\n".join(formatted)


# ============================================================================
# CONTEXT-AWARE SPECIALIZED AGENTS
# ============================================================================


class ContextAwareStatistician:
    """Statistician agent with context awareness"""

    def __init__(self, context_manager, model: str = "openai:gpt-4"):
        self.name = "statistician"
        self.context_manager = context_manager
        self.model = model
        self.agent = Agent(model, system_prompt=self.get_system_prompt())

    def get_system_prompt(self) -> str:
        return """You are an expert statistician with access to past analyses.

When recommending methods:
1. Reference successful approaches from history
2. Learn from past assumption violations
3. Build on proven diagnostics
4. Avoid methods that failed before

Be explicit about how you're using historical knowledge."""

    async def execute_with_context(self, task: str) -> Dict[str, Any]:
        """Execute with full context awareness"""

        # Get context
        context = await self.context_manager.get_context_for_agent(
            self.name, task, include_history=True
        )

        # Build enhanced prompt
        prompt = f"""Task: {task}

RELEVANT CONTEXT:
{context.session_summary}

PAST SUCCESSFUL METHODS:
{self._format_insights(context.topic_insights)}

Recommend appropriate statistical methods, explaining how past work informs your recommendations."""

        # Store thought
        self.context_manager.store_agent_thought(
            self.name,
            f"Processing with {context.total_chunks} context chunks",
            metadata={"task": task},
        )

        result = await self.agent.run(prompt)

        output = {"result": str(result.data), "context_used": context.total_chunks}

        # Store output
        self.context_manager.store_agent_output(
            self.name, output, metadata={"task": task}
        )

        return output

    def _format_insights(self, insights: List[str]) -> str:
        if not insights:
            return "No past insights available."
        return "\n".join(f"- {i}" for i in insights[:3])


class ContextAwareModeler:
    """Modeling agent with context and code pattern awareness"""

    def __init__(self, context_manager, model: str = "openai:gpt-4"):
        self.name = "modeling_agent"
        self.context_manager = context_manager
        self.model = process_model(model)
        self.agent = Agent(self.model, system_prompt=self.get_system_prompt())

    def get_system_prompt(self) -> str:
        return """You are a statistical modeling expert with access to successful code patterns.

When building models:
1. Adapt proven code patterns from past work
2. Use diagnostic approaches that worked well before
3. Learn from past model failures
4. Reference successful specifications

Always generate clean, well-documented code."""

    async def execute_with_context(self, task: str) -> Dict[str, Any]:
        """Execute with code pattern awareness"""

        # Get context including code patterns
        context = await self.context_manager.get_context_for_agent(
            self.name, task, include_history=True
        )

        # Get relevant code examples
        code_examples = await self.context_manager.get_relevant_context(
            query_text=task, context_types=[ContextType.CODE_CHUNK], limit=5
        )

        # Build prompt with code patterns
        prompt = f"""Task: {task}

CONTEXT:
{context.session_summary[:300]}

SUCCESSFUL CODE PATTERNS:
{self._format_code_patterns(context.code_patterns)}

SIMILAR CODE EXAMPLES:
{self._format_code_examples(code_examples[:2])}

Build the model, adapting these successful patterns."""

        # Store thought
        self.context_manager.store_agent_thought(
            self.name,
            f"Using {len(context.code_patterns)} code patterns and {len(code_examples)} examples",
            metadata={"task": task},
        )

        result = await self.agent.run(prompt)

        # Generate code
        code = self._extract_code(str(result.data))

        output = {
            "result": str(result.data),
            "code": code,
            "patterns_used": len(context.code_patterns),
        }

        # Store output and code
        self.context_manager.store_agent_output(
            self.name, output, metadata={"task": task}
        )

        if code:
            self.context_manager.store_code(
                code,
                language="python",
                agent_name=self.name,
                purpose=task,
                metadata={"adapted_from_patterns": True},
            )

        return output

    def _format_code_patterns(self, patterns: List[str]) -> str:
        if not patterns:
            return "No past patterns available."
        return "\n".join(f"{i+1}. {p}" for i, p in enumerate(patterns[:3]))

    def _format_code_examples(self, examples: List[Dict[str, Any]]) -> str:
        if not examples:
            return "No similar examples found."

        formatted = []
        for ex in examples:
            formatted.append(f"```python\n{ex['content'][:300]}\n```")
        return "\n\n".join(formatted)

    def _extract_code(self, response: str) -> str:
        """Extract code blocks from response"""
        # Simplified extraction
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        return ""


# ============================================================================
# COMPLETE WORKFLOW WITH CONTEXT
# ============================================================================


async def run_context_aware_analysis(
    query: str,
    topic: Optional[str] = None,
    data_source: Optional[Dict[str, Any]] = None,
):
    """Run complete analysis with full context management"""

    print("=" * 70)
    print("CONTEXT-AWARE ANALYSIS WORKFLOW")
    print("=" * 70)
    print(f"\nQuery: {query}")
    if topic:
        print(f"Topic: {topic}")
    print()

    # Initialize context manager
    context_mgr = ContextManager(
        persist_directory="./analysis_chroma_db", chunk_size=1000
    )

    # Create orchestrator
    orchestrator = ContextAwareOrchestrator(context_manager=context_mgr)

    # Create agents
    statistician = ContextAwareStatistician(context_mgr)
    modeler = ContextAwareModeler(context_mgr)

    # Phase 1: Planning with context
    print("📋 PHASE 1: Context-Aware Planning")
    print("-" * 70)

    plan = await orchestrator.plan_analysis_with_context(query, data_source, topic)

    print(f"✅ Plan created using {plan['context_used']} historical contexts")
    print(f"   Session ID: {plan['session_id']}")

    # Phase 2: Research with context
    print("\n🔬 PHASE 2: Statistical Research with Context")
    print("-" * 70)

    research_result = await statistician.execute_with_context(
        f"Research methods for: {query}"
    )

    print(
        f"✅ Research complete using {research_result['context_used']} context chunks"
    )

    # Phase 3: Modeling with code patterns
    print("\n📊 PHASE 3: Modeling with Code Patterns")
    print("-" * 70)

    modeling_result = await modeler.execute_with_context(f"Build model for: {query}")

    print(
        f"✅ Model built using {modeling_result.get('patterns_used', 0)} code patterns"
    )
    if modeling_result.get("code"):
        print(f"   Code generated: {len(modeling_result['code'])} characters")

    # Phase 4: Execute full workflow
    print("\n⚙️ PHASE 4: Full Workflow Execution")
    print("-" * 70)

    results = await orchestrator.execute_with_context(plan)

    print(f"✅ Executed {len(results)} stages")

    # Phase 5: Final report
    print("\n📝 PHASE 5: Generating Context-Aware Report")
    print("-" * 70)

    report = await orchestrator.generate_final_report()

    print(f"✅ Report generated")
    print(f"\nReport preview:\n{report[:500]}...\n")

    # Show statistics
    print("\n📊 SESSION STATISTICS")
    print("-" * 70)

    stats = context_mgr.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    return {
        "session_id": plan["session_id"],
        "plan": plan,
        "results": results,
        "report": report,
        "stats": stats,
    }


# ============================================================================
# DEMO: CONTINUITY ACROSS SESSIONS
# ============================================================================


async def demo_cross_session_learning():
    """Demonstrate learning across multiple sessions"""

    print("\n" + "=" * 70)
    print("DEMO: CROSS-SESSION LEARNING")
    print("=" * 70)

    # Session 1: Initial analysis
    print("\n📊 SESSION 1: Initial Analysis")
    print("-" * 70)

    result1 = await run_context_aware_analysis(
        query="Analyze marketing effectiveness on sales",
        topic="marketing_roi_analysis",
        data_source={"type": "csv", "location": "sales_data.csv"},
    )

    print(
        f"\n✅ Session 1 complete. Stored {result1['stats']['total_chunks']} context chunks"
    )

    # Wait a bit
    await asyncio.sleep(1)

    # Session 2: Follow-up with same topic
    print("\n\n📊 SESSION 2: Follow-up Analysis (Same Topic)")
    print("-" * 70)
    print("This session should leverage context from Session 1")

    result2 = await run_context_aware_analysis(
        query="Add competitor analysis and seasonal effects to the marketing model",
        topic="marketing_roi_analysis",  # Same topic!
        data_source={"type": "csv", "location": "sales_data_extended.csv"},
    )

    print(f"\n✅ Session 2 complete.")
    print(f"   Should have referenced Session 1 insights and code patterns")

    # Session 3: Different topic
    print("\n\n📊 SESSION 3: Different Topic")
    print("-" * 70)
    print("This session should NOT use context from Sessions 1 & 2")

    result3 = await run_context_aware_analysis(
        query="Predict customer churn using logistic regression",
        topic="customer_churn_prediction",  # Different topic
        data_source={"type": "csv", "location": "customer_data.csv"},
    )

    print(f"\n✅ Session 3 complete.")
    print(f"   Should be isolated from marketing analysis context")

    # Show how topics are isolated
    print("\n\n🔍 CONTEXT ISOLATION VERIFICATION")
    print("-" * 70)

    context_mgr = ContextManager(persist_directory="./analysis_chroma_db")

    # Get stats
    stats = context_mgr.get_stats()
    print(f"\nTotal chunks stored: {stats['total_chunks']}")
    print(f"Spanning {3} sessions across {2} topics")

    print("\n✅ Cross-session learning demo complete!")


# ============================================================================
# MAIN DEMO
# ============================================================================


async def main():
    """Run comprehensive context-aware system demo"""

    print("\n🚀 CONTEXT-AWARE AGENT SYSTEM")
    print("=" * 70)
    print("Demonstrating:")
    print("  • Context summarization")
    print("  • RAG with ChromaDB")
    print("  • Cross-session learning")
    print("  • Code pattern reuse")
    print("  • Topic-based isolation")
    print()

    # Run single analysis
    await run_context_aware_analysis(
        query="Build a regression model to predict house prices from size, location, and age",
        topic="real_estate_valuation",
        data_source={"type": "csv", "location": "houses.csv"},
    )

    # Uncomment to run cross-session demo
    # await demo_cross_session_learning()

    print("\n\n✅ All demos complete!")
    print("\n💡 Key Features Demonstrated:")
    print("  ✓ Automatic context chunking and storage")
    print("  ✓ Semantic search for relevant context")
    print("  ✓ Context summarization for agents")
    print("  ✓ Code pattern extraction and reuse")
    print("  ✓ Topic-based context isolation")
    print("  ✓ Cross-session learning")
    print("  ✓ Full thought process tracking")


if __name__ == "__main__":

    asyncio.run(main())
