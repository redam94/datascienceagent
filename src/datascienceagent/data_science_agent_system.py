"""
Data Science Multi-Agent System
Main implementation file with orchestrator and core infrastructure
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


# ============================================================================
# CORE DATA MODELS
# ============================================================================


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskType(str, Enum):
    RESEARCH = "research"
    LOAD_DATA = "load_data"
    EDA = "eda"
    MODELING = "modeling"
    INTERPRETATION = "interpretation"


class Task(BaseModel):
    """Represents a single task in the analysis workflow"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: TaskType
    description: str
    dependencies: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    agent: Optional[str] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AnalysisPlan(BaseModel):
    """Complete analysis workflow plan"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    tasks: List[Task]
    workflow_graph: Dict[str, List[str]] = Field(default_factory=dict)
    estimated_duration_minutes: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Response from an agent"""

    agent: str
    status: str  # "success", "failure", "partial"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    artifacts: List[str] = Field(default_factory=list)
    next_suggestions: List[str] = Field(default_factory=list)


class DataSource(BaseModel):
    """Data source specification"""

    type: str  # csv, excel, sql, api, parquet
    location: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Problem(BaseModel):
    """Problem specification"""

    type: str  # regression, classification, time_series, etc.
    description: str
    target_variable: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    research_question: Optional[str] = None


class AnalysisResult(BaseModel):
    """Final analysis results"""

    analysis_id: str
    query: str
    plan: AnalysisPlan
    success: bool
    execution_time_seconds: float
    results: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


# ============================================================================
# BASE AGENT CLASS
# ============================================================================


class BaseDataScienceAgent:
    """Base class for all specialized agents"""

    def __init__(
        self,
        name: str,
        model: str = "openai:gpt-4",
        system_prompt: Optional[str] = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt or self.default_system_prompt()
        self.agent = Agent(model, system_prompt=self.system_prompt)
        self.capabilities: List[TaskType] = []
        self.tools: Dict[str, Any] = {}

    def default_system_prompt(self) -> str:
        """Default system prompt - override in subclasses"""
        return f"You are {self.name}, a specialized AI agent for data science."

    async def execute_task(self, task: Task) -> AgentResponse:
        """Execute a task - override in subclasses"""
        raise NotImplementedError

    async def process(self, task: Task) -> AgentResponse:
        """Process task with error handling"""
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()

            response = await self.execute_task(task)

            task.status = (
                TaskStatus.COMPLETED
                if response.status == "success"
                else TaskStatus.FAILED
            )
            task.completed_at = datetime.now()
            task.output_data = response.result

            return response

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return AgentResponse(agent=self.name, status="failure", error=str(e))

    def can_handle(self, task: Task) -> bool:
        """Check if agent can handle this task"""
        return task.type in self.capabilities


# ============================================================================
# ORCHESTRATOR AGENT
# ============================================================================


class OrchestratorAgent:
    """
    Master orchestrator that coordinates all specialized agents
    Plans workflows and routes tasks to appropriate agents
    """

    def __init__(
        self,
        model: str = "openai:gpt-4",
        agents: Optional[Dict[str, BaseDataScienceAgent]] = None,
    ):
        self.model = model
        self.agents = agents or {}

        self.system_prompt = """You are the Orchestrator Agent for a sophisticated data science system.

Your responsibilities:
1. Understand user requests and break them into actionable tasks
2. Create execution plans with proper task dependencies
3. Route tasks to appropriate specialized agents:
   - Research/Statistician Agent: Statistical planning and method research
   - Data Engineer Agent: Data loading and preparation
   - EDA Agent: Exploratory analysis and visualization
   - Modeling Agent: Statistical modeling and analysis
   - Interpreter Agent: Result interpretation and reporting

4. Coordinate agent execution in the correct order
5. Aggregate results and provide comprehensive reports

When planning:
- Consider data availability and quality
- Ensure statistical assumptions are validated
- Include diagnostic checks
- Plan for visualization and interpretation
- Think about reproducibility

Be methodical and thorough. Always validate dependencies before execution."""

        self.agent = Agent(model, system_prompt=self.system_prompt)

    def register_agent(self, agent: BaseDataScienceAgent):
        """Register a specialized agent"""
        self.agents[agent.name] = agent
        print(f"✅ Registered agent: {agent.name}")

    async def plan_analysis(
        self, query: str, data_source: Optional[DataSource] = None
    ) -> AnalysisPlan:
        """Create an execution plan from user query"""

        prompt = f"""Create a detailed analysis plan for this request:

Query: {query}
Data Source: {data_source.model_dump() if data_source else "To be determined"}

Create a step-by-step plan with these task types:
- research: Statistical planning and method selection
- load_data: Data loading and validation
- eda: Exploratory data analysis
- modeling: Statistical modeling
- interpretation: Result interpretation

For each task, specify:
1. Task description
2. Dependencies (which tasks must complete first)
3. Required inputs
4. Expected outputs

Return a structured plan with proper task ordering."""

        result = await self.agent.run(prompt)
        response = str(result.data)

        # Parse response into tasks (simplified for demo)
        # In production, use structured output or JSON parsing
        tasks = self._parse_plan_response(response, query)

        # Build dependency graph
        workflow_graph = self._build_workflow_graph(tasks)

        return AnalysisPlan(
            query=query,
            tasks=tasks,
            workflow_graph=workflow_graph,
            estimated_duration_minutes=len(tasks) * 2.0,
        )

    def _parse_plan_response(self, response: str, query: str) -> List[Task]:
        """Parse LLM response into structured tasks"""
        # Simplified parsing - in production use structured output
        tasks = []

        # Default workflow for demonstration
        task_sequence = [
            ("research", "Research appropriate statistical methods", []),
            ("load_data", "Load and validate data", ["research"]),
            ("eda", "Perform exploratory data analysis", ["load_data"]),
            ("modeling", "Fit statistical models", ["eda"]),
            ("interpretation", "Interpret results and create report", ["modeling"]),
        ]

        for task_type, description, deps in task_sequence:
            task = Task(
                type=TaskType(task_type),
                description=description,
                dependencies=[
                    t.id for t in tasks if any(d in t.type.value for d in deps)
                ],
                input_data={"query": query},
            )
            tasks.append(task)

        return tasks

    def _build_workflow_graph(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """Build adjacency list representation of task dependencies"""
        graph = {task.id: task.dependencies for task in tasks}
        return graph

    async def execute_plan(self, plan: AnalysisPlan) -> AnalysisResult:
        """Execute the analysis plan"""

        print("\n" + "=" * 70)
        print(f"🚀 EXECUTING ANALYSIS: {plan.query}")
        print("=" * 70)

        start_time = datetime.now()
        results = {}

        # Execute tasks in dependency order
        executed_tasks = set()

        while len(executed_tasks) < len(plan.tasks):
            # Find tasks ready to execute (all dependencies met)
            ready_tasks = [
                task
                for task in plan.tasks
                if task.id not in executed_tasks
                and all(dep in executed_tasks for dep in task.dependencies)
            ]

            if not ready_tasks:
                # No tasks ready - check for circular dependencies
                remaining = [t for t in plan.tasks if t.id not in executed_tasks]
                raise ValueError(
                    f"Circular dependency detected. Remaining tasks: {[t.description for t in remaining]}"
                )

            # Execute ready tasks
            for task in ready_tasks:
                print(f"\n📋 Executing: {task.description}")
                result = await self._execute_task(task)
                results[task.type.value] = result
                executed_tasks.add(task.id)

        execution_time = (datetime.now() - start_time).total_seconds()

        print("\n✅ Analysis complete!")
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")

        return AnalysisResult(
            analysis_id=plan.id,
            query=plan.query,
            plan=plan,
            success=True,
            execution_time_seconds=execution_time,
            results=results,
        )

    async def _execute_task(self, task: Task) -> AgentResponse:
        """Route task to appropriate agent"""

        # Find agent that can handle this task
        agent = None
        for ag in self.agents.values():
            if ag.can_handle(task):
                agent = ag
                break

        if not agent:
            # Fallback to orchestrator handling
            return await self._handle_task_directly(task)

        task.agent = agent.name
        return await agent.process(task)

    async def _handle_task_directly(self, task: Task) -> AgentResponse:
        """Handle task directly when no specialized agent available"""

        prompt = f"""Execute this task:

Task Type: {task.type}
Description: {task.description}
Input: {task.input_data}

Provide a detailed response including:
1. What was done
2. Key findings
3. Next steps or recommendations"""

        result = await self.agent.run(prompt)

        return AgentResponse(
            agent="orchestrator", status="success", result={"output": str(result.data)}
        )

    async def synthesize_results(self, analysis_result: AnalysisResult) -> str:
        """Create final synthesis of all results"""

        prompt = f"""Synthesize these analysis results into a comprehensive report:

Query: {analysis_result.query}

Results from each stage:
{self._format_results(analysis_result.results)}

Create a clear, actionable report including:
1. Executive Summary
2. Key Findings
3. Statistical Results
4. Recommendations
5. Limitations and Caveats

Format for a business audience."""

        result = await self.agent.run(prompt)
        return str(result.data)

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format results for synthesis"""
        formatted = []
        for stage, data in results.items():
            formatted.append(f"\n{stage.upper()}:")
            formatted.append(f"  {data}")
        return "\n".join(formatted)


# ============================================================================
# WORKFLOW ENGINE
# ============================================================================


class WorkflowEngine:
    """Manages workflow execution with state persistence"""

    def __init__(self, orchestrator: OrchestratorAgent):
        self.orchestrator = orchestrator
        self.active_workflows: Dict[str, AnalysisPlan] = {}

    async def start_analysis(
        self, query: str, data_source: Optional[DataSource] = None
    ) -> str:
        """Start a new analysis workflow"""

        # Create plan
        plan = await self.orchestrator.plan_analysis(query, data_source)

        # Store workflow
        self.active_workflows[plan.id] = plan

        return plan.id

    async def execute_workflow(self, workflow_id: str) -> AnalysisResult:
        """Execute a workflow by ID"""

        plan = self.active_workflows.get(workflow_id)
        if not plan:
            raise ValueError(f"Workflow {workflow_id} not found")

        result = await self.orchestrator.execute_plan(plan)
        return result

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""

        plan = self.active_workflows.get(workflow_id)
        if not plan:
            return {"error": "Workflow not found"}

        return {
            "id": plan.id,
            "query": plan.query,
            "total_tasks": len(plan.tasks),
            "completed_tasks": sum(
                1 for t in plan.tasks if t.status == TaskStatus.COMPLETED
            ),
            "failed_tasks": sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED),
            "current_tasks": [
                {"description": t.description, "status": t.status}
                for t in plan.tasks
                if t.status == TaskStatus.IN_PROGRESS
            ],
        }


# ============================================================================
# DEMO SPECIALIZED AGENTS
# ============================================================================


class StatisticianAgent(BaseDataScienceAgent):
    """Research and statistical planning agent"""

    def __init__(self, model: str = "openai:gpt-4"):
        super().__init__("statistician", model)
        self.capabilities = [TaskType.RESEARCH]

    def default_system_prompt(self) -> str:
        return """You are an expert statistician and research methodologist.

Your expertise includes:
- Recommending appropriate statistical methods
- Identifying assumptions and their diagnostics
- Planning rigorous analyses
- Understanding research design
- Interpreting statistical theory

When planning analyses:
1. Consider the research question carefully
2. Identify the appropriate statistical framework
3. List key assumptions to validate
4. Recommend diagnostics and sensitivity analyses
5. Consider power and sample size
6. Suggest visualization strategies

Be thorough and scientifically rigorous."""

    async def execute_task(self, task: Task) -> AgentResponse:
        """Research statistical methods"""

        query = task.input_data.get("query", "")

        prompt = f"""Plan the statistical analysis for this request:

{query}

Provide:
1. Problem type (regression, classification, time series, etc.)
2. Recommended statistical methods (with justification)
3. Key assumptions to validate
4. Required diagnostics
5. Suggested visualizations
6. Potential challenges

Be specific and actionable."""

        result = await self.agent.run(prompt)

        return AgentResponse(
            agent=self.name,
            status="success",
            result={
                "plan": str(result.data),
                "methods_recommended": ["linear_regression", "diagnostics"],
                "assumptions": [
                    "linearity",
                    "independence",
                    "normality",
                    "homoscedasticity",
                ],
            },
            next_suggestions=["Proceed to data loading and validation"],
        )


class DataEngineerAgent(BaseDataScienceAgent):
    """Data loading and preparation agent"""

    def __init__(self, model: str = "openai:gpt-4"):
        super().__init__("data_engineer", model)
        self.capabilities = [TaskType.LOAD_DATA]

    def default_system_prompt(self) -> str:
        return """You are an expert data engineer specializing in data loading and preparation.

Your skills include:
- Loading data from various sources (CSV, Excel, SQL, APIs)
- Writing robust data loading scripts
- Data validation and quality checks
- Handling missing data
- Data type conversions
- Performance optimization

Always ensure:
1. Data integrity
2. Proper error handling
3. Clear documentation
4. Reproducible scripts"""

    async def execute_task(self, task: Task) -> AgentResponse:
        """Load and prepare data"""

        # In production, actually load data
        # For demo, simulate data loading

        await asyncio.sleep(0.5)  # Simulate I/O

        return AgentResponse(
            agent=self.name,
            status="success",
            result={
                "data_loaded": True,
                "n_rows": 1000,
                "n_cols": 10,
                "columns": ["x1", "x2", "y"],
                "missing_values": {"x1": 0, "x2": 5, "y": 0},
                "loading_script": "# pandas.read_csv('data.csv')\n# data validation complete",
            },
            next_suggestions=["Data is ready for EDA"],
        )


# ============================================================================
# MAIN DEMO
# ============================================================================


async def main():
    """Demonstrate the multi-agent system"""

    print("🤖 DATA SCIENCE MULTI-AGENT SYSTEM")
    print("=" * 70)
    print()

    # Create orchestrator
    orchestrator = OrchestratorAgent()

    # Register specialized agents
    orchestrator.register_agent(StatisticianAgent())
    orchestrator.register_agent(DataEngineerAgent())

    # Create workflow engine
    engine = WorkflowEngine(orchestrator)

    # Example analysis request
    query = """
    I have sales data with marketing spend, seasonality, and competitor actions.
    I want to understand what drives sales and predict future performance.
    """

    print(f"📊 Analysis Request: {query}\n")

    # Start workflow
    workflow_id = await engine.start_analysis(query)
    print(f"✅ Workflow created: {workflow_id}\n")

    # Check status
    status = engine.get_workflow_status(workflow_id)
    print(f"📋 Initial Status: {status['total_tasks']} tasks planned\n")

    # Execute workflow
    result = await engine.execute_workflow(workflow_id)

    # Synthesize results
    print("\n" + "=" * 70)
    print("📊 SYNTHESIZING RESULTS")
    print("=" * 70)

    final_report = await orchestrator.synthesize_results(result)
    print("\n" + final_report)

    print("\n\n✅ Complete system demonstration finished!")
    print("\n💡 Next steps:")
    print("  1. Implement remaining specialized agents (EDA, Modeling, Interpreter)")
    print("  2. Add actual data processing capabilities")
    print("  3. Implement code execution sandbox")
    print("  4. Add FastAPI REST API layer")
    print("  5. Create visualization tools")
    print("  6. Add result caching and persistence")


if __name__ == "__main__":
    asyncio.run(main())
