"""
Refactored Specialist Agents with Context Management

Each agent:
- Receives only targeted context for its specific task
- Has code execution capabilities
- Can diagnose and fix errors
- Stores successful patterns for future use
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from pydantic import BaseModel, Field

# Import base agent
from datascienceagent.core.context_aware_agent import (
    ContextAwareAgent,
    TargetedContext,
    ContextType,
)
from datascienceagent.agents.data_engineer_helper import DataEngineerAgentHelper


# ============================================================================
# STATISTICIAN AGENT - Planning and Research
# ============================================================================


class StatisticianAgent(ContextAwareAgent):
    """
    Statistical planning and research agent.

    Responsibilities:
    - Research appropriate statistical methods
    - Plan analysis approaches
    - Recommend validation strategies
    - Specify assumptions to check

    Context needs:
    - User query
    - Past successful analyses (similar problems)
    - Available methods (from knowledge base)
    """

    def __init__(
        self, model: str = "openai:gpt-4", context_manager: Optional[Any] = None
    ):
        super().__init__(
            name="statistician_agent",
            model=model,
            context_manager=context_manager,
            enable_code_execution=False,  # Mainly planning, not coding
        )

    def get_system_prompt(self) -> str:
        return """You are an expert statistician and research methodologist.

Your role is to:
1. Research and recommend appropriate statistical methods
2. Plan comprehensive analysis strategies
3. Identify key assumptions to validate
4. Specify diagnostic procedures
5. Suggest power calculations and sample size considerations

When planning:
- Consider the research question and data characteristics
- Recommend multiple approaches (classical and Bayesian when appropriate)
- Identify potential confounders and biases
- Specify validation procedures
- Learn from past successful analyses

Provide clear, structured plans that guide the modeling agent."""

    async def plan_analysis(
        self,
        problem_description: str,
        data_summary: Optional[Dict[str, Any]],
        workflow_node: Any,
    ) -> Dict[str, Any]:
        """
        Plan a comprehensive statistical analysis.

        Returns:
            Dict with recommended methods, assumptions, diagnostics
        """
        task = f"Plan statistical analysis for: {problem_description}"

        # Add data summary to task context
        if data_summary:
            task += f"\n\nData characteristics: {data_summary}"

        result = await self.execute_with_context(task, workflow_node)

        # Parse and structure the plan
        structured_plan = self._structure_plan(result)

        return structured_plan

    def _structure_plan(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the raw analysis plan"""
        return {
            "plan_text": raw_result.get("raw_result", ""),
            "recommended_methods": [],  # Extract from text
            "assumptions_to_check": [],
            "diagnostic_procedures": [],
            "validation_strategy": "",
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# DATA ENGINEER AGENT - Data Processing
# ============================================================================


class DataEngineerAgent(ContextAwareAgent):
    """
    Data loading, cleaning, and preparation specialist.

    Responsibilities:
    - Load data from various sources
    - Validate data quality
    - Clean and preprocess data
    - Feature engineering (post-EDA)

    Context needs:
    - Data source specifications
    - Validation requirements
    - Cleaning strategies from past work
    - EDA insights (for post-EDA processing)
    """

    def __init__(
        self, model: str = "openai:gpt-4", context_manager: Optional[Any] = None
    ):
        super().__init__(
            name="data_engineer_agent",
            model=model,
            context_manager=context_manager,
            enable_code_execution=True,
        )
        self.helper = DataEngineerAgentHelper()

    def get_system_prompt(self) -> str:
        return """You are a data engineering specialist.

Your role is to:
1. Load data from various sources (CSV, Excel, SQL, APIs)
2. Validate data quality and schema
3. Clean and preprocess data
4. Engineer features based on domain knowledge or EDA insights
5. Handle missing data appropriately

When writing data processing code:
- Generate clean, well-documented code
- Handle errors gracefully
- Validate data at each step
- Use vectorized operations for efficiency
- Log data transformations for reproducibility

Tools available:
- pandas, polars for data manipulation
- SQL for database queries
- Data validation frameworks

Always test code on sample data first."""

    async def load_data(
        self, data_source: Dict[str, Any], workflow_node: Any
    ) -> Dict[str, Any]:
        """
        Load data from specified source.

        Args:
            data_source: Dict with type, location, parameters
            workflow_node: Current workflow node

        Returns:
            Dict with loaded data info and loading code
        """
        

        return self.helper.load_data_from_source(data_source)

    async def clean_data(
        self,
        data_info: Dict[str, Any],
        validation_report: Dict[str, Any],
        workflow_node: Any,
    ) -> Dict[str, Any]:
        """
        Clean data based on validation report.

        Args:
            data_info: Information about the data
            validation_report: Issues found during validation
            workflow_node: Current workflow node

        Returns:
            Dict with cleaning code and results
        """
        task = f"""Clean data based on validation findings:

Data info: {data_info}

Validation issues:
{validation_report}

Generate Python code to:
1. Handle missing values appropriately
2. Fix data type issues
3. Remove or handle outliers
4. Address any quality issues

Use appropriate strategies from past successful work."""

        return await self.execute_with_context(task, workflow_node)

    async def engineer_features(
        self, eda_insights: Dict[str, Any], workflow_node: Any
    ) -> Dict[str, Any]:
        """
        Engineer features based on EDA insights.

        This happens POST-EDA to exploit findings.

        Args:
            eda_insights: Insights from exploratory analysis
            workflow_node: Current workflow node

        Returns:
            Dict with feature engineering code
        """
        task = f"""Engineer features based on EDA insights:

{eda_insights}

Generate Python code to:
1. Create interaction terms for interesting relationships
2. Transform variables as suggested by EDA
3. Create polynomial features if needed
4. Handle categorical variables appropriately
5. Scale/normalize as needed

Focus on features that will improve model performance."""

        return await self.execute_with_context(task, workflow_node)


# ============================================================================
# EDA AGENT - Exploratory Analysis
# ============================================================================


class EDAAgent(ContextAwareAgent):
    """
    Exploratory data analysis and visualization specialist.

    Responsibilities:
    - Generate appropriate visualizations
    - Identify patterns and relationships
    - Detect outliers and anomalies
    - Suggest feature engineering
    - Assess distribution properties

    Context needs:
    - Clean data
    - Analysis objectives
    - Past successful visualization patterns
    """

    def __init__(
        self, model: str = "openai:gpt-4", context_manager: Optional[Any] = None
    ):
        super().__init__(
            name="eda_agent",
            model=model,
            context_manager=context_manager,
            enable_code_execution=True,
        )

    def get_system_prompt(self) -> str:
        return """You are an exploratory data analysis specialist.

Your role is to:
1. Create insightful visualizations
2. Identify patterns, trends, and relationships
3. Detect outliers and anomalies
4. Assess distribution properties
5. Suggest feature engineering opportunities

When creating visualizations:
- Use matplotlib, seaborn, or plotly appropriately
- Create publication-quality plots
- Add clear labels and titles
- Use appropriate color schemes
- Save plots to files

When analyzing:
- Look for non-linear relationships
- Check for multicollinearity
- Identify potential transformations
- Suggest interaction terms
- Note any data quality issues

Provide actionable insights that guide modeling."""

    async def explore_data(
        self, data_info: Dict[str, Any], objectives: List[str], workflow_node: Any
    ) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory analysis.

        Args:
            data_info: Information about the data
            objectives: Specific objectives (e.g., "understand X-Y relationship")
            workflow_node: Current workflow node

        Returns:
            Dict with insights, visualizations, and code
        """
        task = f"""Perform exploratory data analysis:

Data: {data_info}

Objectives:
{chr(10).join(f'- {obj}' for obj in objectives)}

Generate Python code to:
1. Create univariate distributions for key variables
2. Create scatter plots / correlation matrices
3. Check for outliers
4. Assess normality and other distributional assumptions
5. Identify interesting patterns or relationships

Provide insights and recommendations based on findings."""

        result = await self.execute_with_context(task, workflow_node)

        # Extract insights from result
        insights = self._extract_insights(result)

        return {
            **result,
            "insights": insights,
            "feature_suggestions": self._extract_feature_suggestions(result),
        }

    def _extract_insights(self, result: Dict[str, Any]) -> List[str]:
        """Extract key insights from EDA result"""
        # Simple extraction - in production, parse more carefully
        text = result.get("raw_result", "")
        insights = []

        # Look for patterns in text
        for line in text.split("\n"):
            if any(
                keyword in line.lower()
                for keyword in ["insight:", "finding:", "note:", "important:"]
            ):
                insights.append(line.strip())

        return insights[:5]  # Top 5 insights

    def _extract_feature_suggestions(self, result: Dict[str, Any]) -> List[str]:
        """Extract feature engineering suggestions"""
        text = result.get("raw_result", "")
        suggestions = []

        # Look for suggestions in text
        for line in text.split("\n"):
            if any(
                keyword in line.lower()
                for keyword in ["suggest:", "recommend:", "consider:", "try:"]
            ):
                suggestions.append(line.strip())

        return suggestions[:5]  # Top 5 suggestions


# ============================================================================
# MODELING AGENT - Statistical Modeling
# ============================================================================


class ModelingAgent(ContextAwareAgent):
    """
    Statistical modeling specialist.

    Responsibilities:
    - Specify models based on statistical plan
    - Fit models using statsmodels or PyMC
    - Run diagnostics
    - Compare model specifications

    Context needs:
    - Statistical plan
    - Processed data
    - EDA insights
    - Successful model code patterns
    """

    def __init__(
        self, model: str = "openai:gpt-4", context_manager: Optional[Any] = None
    ):
        super().__init__(
            name="modeling_agent",
            model=model,
            context_manager=context_manager,
            enable_code_execution=True,
        )

    def get_system_prompt(self) -> str:
        return """You are a statistical modeling specialist.

Your role is to:
1. Specify models based on statistical plans
2. Implement models in statsmodels or PyMC
3. Fit models and estimate parameters
4. Run comprehensive diagnostics
5. Compare alternative specifications

When building models:
- Follow the statistical plan carefully
- Use appropriate estimation methods
- Check all assumptions
- Run residual diagnostics
- Calculate fit statistics
- Use past successful code patterns

For classical models, use statsmodels.
For Bayesian models, use PyMC with appropriate priors.

Always validate results before finalizing."""

    async def specify_model(
        self,
        statistical_plan: Dict[str, Any],
        data_info: Dict[str, Any],
        eda_insights: Dict[str, Any],
        workflow_node: Any,
    ) -> Dict[str, Any]:
        """
        Specify model structure.

        Args:
            statistical_plan: Plan from statistician
            data_info: Information about data
            eda_insights: Insights from EDA
            workflow_node: Current workflow node

        Returns:
            Dict with model specification
        """
        task = f"""Specify statistical model:

Statistical Plan:
{statistical_plan}

Data:
{data_info}

EDA Insights:
{eda_insights}

Specify:
1. Model formula/structure
2. Estimation method
3. Any transformations needed
4. Priors (if Bayesian)
5. Diagnostics to run"""

        return await self.execute_with_context(task, workflow_node)

    async def fit_model(
        self, model_spec: Dict[str, Any], workflow_node: Any
    ) -> Dict[str, Any]:
        """
        Fit the specified model.

        Args:
            model_spec: Model specification
            workflow_node: Current workflow node

        Returns:
            Dict with fitted model and results
        """
        task = f"""Fit statistical model:

Specification:
{model_spec}

Generate Python code to:
1. Prepare data for modeling
2. Fit the model using appropriate library (statsmodels or PyMC)
3. Extract key results (coefficients, p-values, fit statistics)
4. Save fitted model for later use

Use successful patterns from past modeling work."""

        return await self.execute_with_context(task, workflow_node)

    async def run_diagnostics(
        self, fitted_model_info: Dict[str, Any], workflow_node: Any
    ) -> Dict[str, Any]:
        """
        Run model diagnostics.

        Args:
            fitted_model_info: Information about fitted model
            workflow_node: Current workflow node

        Returns:
            Dict with diagnostic results
        """
        task = f"""Run model diagnostics:

Fitted Model:
{fitted_model_info}

Generate Python code to:
1. Check residual assumptions (normality, homoscedasticity)
2. Look for influential observations
3. Check for multicollinearity (VIF)
4. Create residual plots
5. Run specification tests

Identify any issues and suggest remedies."""

        result = await self.execute_with_context(task, workflow_node)

        # Extract diagnostic findings
        findings = self._extract_diagnostic_findings(result)

        return {**result, "diagnostic_findings": findings}

    def _extract_diagnostic_findings(self, result: Dict[str, Any]) -> List[str]:
        """Extract key diagnostic findings"""
        text = result.get("raw_result", "")
        findings = []

        for line in text.split("\n"):
            if any(
                keyword in line.lower()
                for keyword in ["violation:", "issue:", "problem:", "concern:", "pass"]
            ):
                findings.append(line.strip())

        return findings[:10]


# ============================================================================
# INTERPRETER AGENT - Result Interpretation
# ============================================================================


class InterpreterAgent(ContextAwareAgent):
    """
    Model interpretation and reporting specialist.

    Responsibilities:
    - Interpret model results
    - Assess practical significance
    - Generate natural language summaries
    - Create final reports
    - Provide recommendations

    Context needs:
    - Model results
    - Diagnostics
    - Original objectives
    - Business context
    """

    def __init__(
        self, model: str = "openai:gpt-4", context_manager: Optional[Any] = None
    ):
        super().__init__(
            name="interpreter_agent",
            model=model,
            context_manager=context_manager,
            enable_code_execution=True,  # May generate summary plots
        )

    def get_system_prompt(self) -> str:
        return """You are a results interpretation specialist.

Your role is to:
1. Interpret statistical results for non-technical audiences
2. Assess practical (not just statistical) significance
3. Relate findings to original objectives
4. Provide actionable recommendations
5. Note limitations and caveats

When interpreting:
- Use clear, non-technical language
- Quantify effect sizes meaningfully
- Distinguish correlation from causation
- Acknowledge uncertainty
- Provide context for magnitudes

Generate reports that decision-makers can act on."""

    async def interpret_results(
        self,
        model_results: Dict[str, Any],
        diagnostics: Dict[str, Any],
        objectives: List[str],
        workflow_node: Any,
    ) -> Dict[str, Any]:
        """
        Interpret model results.

        Args:
            model_results: Results from modeling
            diagnostics: Diagnostic findings
            objectives: Original analysis objectives
            workflow_node: Current workflow node

        Returns:
            Dict with interpretation
        """
        task = f"""Interpret statistical results:

Model Results:
{model_results}

Diagnostics:
{diagnostics}

Original Objectives:
{chr(10).join(f'- {obj}' for obj in objectives)}

Provide:
1. Plain-language summary of findings
2. Effect sizes with practical context
3. Assessment of model quality
4. Answers to original questions
5. Limitations and caveats
6. Recommendations"""

        return await self.execute_with_context(task, workflow_node)

    async def generate_report(
        self,
        interpretation: Dict[str, Any],
        all_outputs: Dict[str, Any],
        workflow_node: Any,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report.

        Args:
            interpretation: Interpretation of results
            all_outputs: All outputs from workflow
            workflow_node: Current workflow node

        Returns:
            Dict with formatted report
        """
        task = f"""Generate comprehensive analysis report:

Interpretation:
{interpretation}

Analysis Outputs:
{all_outputs}

Create a well-structured report with:
1. Executive Summary
2. Methods Overview
3. Key Findings (with visualizations)
4. Detailed Results
5. Recommendations
6. Technical Appendix

Format for both technical and non-technical readers."""

        return await self.execute_with_context(task, workflow_node)


# ============================================================================
# AGENT FACTORY
# ============================================================================


class AgentFactory:
    """Factory for creating specialist agents"""

    @staticmethod
    def create_agent(
        agent_type: str,
        model: str = "openai:gpt-4",
        context_manager: Optional[Any] = None,
    ) -> ContextAwareAgent:
        """
        Create a specialist agent by type.

        Args:
            agent_type: Type of agent (statistician, data_engineer, etc.)
            model: LLM model to use
            context_manager: Context manager instance

        Returns:
            Specialist agent instance
        """
        agents = {
            "statistician": StatisticianAgent,
            "data_engineer": DataEngineerAgent,
            "eda": EDAAgent,
            "modeling": ModelingAgent,
            "interpreter": InterpreterAgent,
        }

        agent_class = agents.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent_class(model=model, context_manager=context_manager)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        # Create agents
        statistician = StatisticianAgent()
        data_engineer = DataEngineerAgent()
        eda_agent = EDAAgent()
        modeling_agent = ModelingAgent()
        interpreter = InterpreterAgent()

        print("✅ Created all specialist agents")
        print(f"  - {statistician.name}")
        print(f"  - {data_engineer.name}")
        print(f"  - {eda_agent.name}")
        print(f"  - {modeling_agent.name}")
        print(f"  - {interpreter.name}")

    asyncio.run(demo())
