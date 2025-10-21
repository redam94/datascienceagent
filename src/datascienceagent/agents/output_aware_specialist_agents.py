"""
Output-Aware Specialist Agents

Enhanced specialist agents that combine:
- Context awareness (retrieve relevant past work)
- Output management (save all outputs systematically)
- Code execution with output capture
- Automatic result tracking

These agents should be used instead of the base specialist_agents
when you want comprehensive output capture and organization.

Author: Data Science Agent System
Created: 2025-10-19
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from pydantic import BaseModel, Field
from loguru import logger

from datascienceagent.core.output_context_aware_agent import OutputContextAwareAgent
from datascienceagent.core.context_management import ContextManager, ContextType
from datascienceagent.core.output_manager import OutputManager
from datascienceagent.agents.data_engineer_helper import DataEngineerAgentHelper


# ============================================================================
# OUTPUT-AWARE STATISTICIAN AGENT
# ============================================================================


class OutputAwareStatisticianAgent(OutputContextAwareAgent):
    """
    Statistical planning and research agent with output management.
    
    Responsibilities:
    - Research appropriate statistical methods
    - Plan analysis approaches
    - Recommend validation strategies
    - Specify assumptions to check
    
    All outputs (research findings, analysis plans, recommendations)
    are automatically saved to organized directories.
    """
    
    def __init__(
        self,
        model: str = "openai:gpt-4",
        context_manager: Optional[ContextManager] = None,
        output_manager: Optional[OutputManager] = None,
    ):
        super().__init__(
            name="statistician",
            model=model,
            context_manager=context_manager,
            output_manager=output_manager,
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

Provide clear, structured plans that guide the modeling agent.

Your outputs will be automatically saved and organized for team review."""
    
    async def plan_analysis(
        self,
        problem_description: str,
        data_summary: Optional[Dict[str, Any]],
        workflow_node: Any,
        stage_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Plan a comprehensive statistical analysis with output capture.
        
        Args:
            problem_description: Description of the problem
            data_summary: Summary of available data
            workflow_node: Current workflow node
            stage_order: Stage order number
            
        Returns:
            Dict with analysis plan and metadata
        """
        task = f"Plan statistical analysis for: {problem_description}"
        
        if data_summary:
            task += f"\n\nData characteristics: {data_summary}"
        
        # Execute with full output management
        result = await self.execute_with_outputs(
            task=task,
            workflow_node=workflow_node,
            stage_order=stage_order
        )
        
        # Structure the plan
        if result["success"]:
            structured_plan = self._structure_plan(result)
            result["structured_plan"] = structured_plan
            
            # Save the structured plan
            if self.output_manager:
                stage_name = workflow_node.stage.value if hasattr(workflow_node, 'stage') else "statistical_planning"
                self.output_manager.save_json(
                    stage_name,
                    structured_plan,
                    "analysis_plan.json",
                    stage_order
                )
        
        return result
    
    def _structure_plan(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Structure the raw analysis plan into organized sections"""
        return {
            "plan_text": raw_result.get("raw_result", ""),
            "recommended_methods": [],  # Could extract from text with NLP
            "assumptions_to_check": [],
            "diagnostic_procedures": [],
            "validation_strategy": "",
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# OUTPUT-AWARE DATA ENGINEER AGENT
# ============================================================================


class OutputAwareDataEngineerAgent(OutputContextAwareAgent):
    """
    Data loading, cleaning, and preparation specialist with output management.
    
    Responsibilities:
    - Load data from various sources
    - Validate data quality
    - Clean and preprocess data
    - Feature engineering (post-EDA)
    
    All outputs (loaded data, cleaned data, transformation code, validation reports)
    are automatically saved in organized directories.
    """
    
    def __init__(
        self,
        model: str = "openai:gpt-4",
        context_manager: Optional[ContextManager] = None,
        output_manager: Optional[OutputManager] = None,
    ):
        super().__init__(
            name="data_engineer",
            model=model,
            context_manager=context_manager,
            output_manager=output_manager,
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

Always test code on sample data first.

Your outputs (code, data, reports) will be automatically saved and organized."""
    
    async def load_data(
        self,
        data_source: Dict[str, Any],
        workflow_node: Any,
        stage_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load data from specified source with output capture.
        
        Args:
            data_source: Dict with type, location, parameters
            workflow_node: Current workflow node
            stage_order: Stage order number
            
        Returns:
            Dict with loaded data info and metadata
        """
        logger.info(f"📥 Loading data from source...")
        
        # Use helper to load data
        result = self.helper.load_data_from_source(data_source)
        
        # Save outputs if manager available
        if self.output_manager and result["success"]:
            stage_name = workflow_node.stage.value if hasattr(workflow_node, 'stage') else "data_acquisition"
            
            # Save the loading code
            if result.get("code"):
                self.output_manager.save_code(
                    stage_name,
                    result["code"],
                    "data_loading_code.py",
                    stage_order
                )
            
            # Save metadata
            if result.get("metadata"):
                self.output_manager.save_json(
                    stage_name,
                    result["metadata"],
                    "data_metadata.json",
                    stage_order
                )
            
            # Save the actual data
            if result.get("data") is not None:
                self.output_manager.save_dataframe(
                    stage_name,
                    result["data"],
                    "loaded_data.csv",
                    stage_order,
                    save_format="csv"
                )
            
            # Track output location
            stage_dir = self.output_manager.get_stage_dir(stage_name, stage_order)
            result["outputs_saved_to"] = str(stage_dir)
        
        # Store in context
        if self.context_manager and result["success"]:
            self.context_manager.store_agent_output(
                self.name,
                result,
                metadata={
                    "stage": "data_acquisition",
                    "stage_order": stage_order,
                }
            )
        
        return result
    
    async def clean_data(
        self,
        data_info: Dict[str, Any],
        validation_report: Dict[str, Any],
        workflow_node: Any,
        stage_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Clean data based on validation report with output capture.
        
        Args:
            data_info: Information about the data
            validation_report: Issues found during validation
            workflow_node: Current workflow node
            stage_order: Stage order number
            
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
        
        return await self.execute_with_outputs(
            task=task,
            workflow_node=workflow_node,
            stage_order=stage_order
        )
    
    async def engineer_features(
        self,
        eda_insights: Dict[str, Any],
        workflow_node: Any,
        stage_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Engineer features based on EDA insights with output capture.
        
        Args:
            eda_insights: Insights from exploratory analysis
            workflow_node: Current workflow node
            stage_order: Stage order number
            
        Returns:
            Dict with feature engineering code and results
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
        
        return await self.execute_with_outputs(
            task=task,
            workflow_node=workflow_node,
            stage_order=stage_order
        )


# ============================================================================
# OUTPUT-AWARE EDA AGENT
# ============================================================================


class OutputAwareEDAAgent(OutputContextAwareAgent):
    """
    Exploratory data analysis and visualization specialist with output management.
    
    Responsibilities:
    - Generate appropriate visualizations
    - Identify patterns and relationships
    - Detect outliers and anomalies
    - Suggest feature engineering
    - Assess distribution properties
    
    All outputs (plots, insights, statistical summaries) are automatically
    saved in organized directories.
    """
    
    def __init__(
        self,
        model: str = "openai:gpt-4",
        context_manager: Optional[ContextManager] = None,
        output_manager: Optional[OutputManager] = None,
    ):
        super().__init__(
            name="eda",
            model=model,
            context_manager=context_manager,
            output_manager=output_manager,
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
- Save plots to files with descriptive names (e.g., 'correlation_matrix.png', 'distribution_age.png')

When analyzing:
- Look for non-linear relationships
- Check for multicollinearity
- Identify potential transformations
- Suggest interaction terms
- Note any data quality issues

Provide actionable insights that guide modeling.

Your plots and insights will be automatically saved and organized."""
    
    async def explore_data(
        self,
        data_info: Dict[str, Any],
        objectives: List[str],
        workflow_node: Any,
        stage_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory analysis with output capture.
        
        Args:
            data_info: Information about the data
            objectives: Specific objectives
            workflow_node: Current workflow node
            stage_order: Stage order number
            
        Returns:
            Dict with insights, visualizations, and code
        """
        task = f"""Perform exploratory data analysis:

Data: {data_info}

Objectives:
{chr(10).join(f'- {obj}' for obj in objectives)}

Generate Python code to:
1. Create univariate distributions for key variables (save as 'distribution_*.png')
2. Create scatter plots / correlation matrices (save as 'correlation_matrix.png')
3. Check for outliers (create boxplots, save as 'outliers_*.png')
4. Assess normality and other distributional assumptions (Q-Q plots, save as 'normality_*.png')
5. Identify interesting patterns or relationships

Save all plots with descriptive filenames.

Provide insights and recommendations based on findings."""
        
        result = await self.execute_with_outputs(
            task=task,
            workflow_node=workflow_node,
            stage_order=stage_order
        )
        
        # Extract insights from result
        if result["success"]:
            insights = self._extract_insights(result)
            feature_suggestions = self._extract_feature_suggestions(result)
            
            result["insights"] = insights
            result["feature_suggestions"] = feature_suggestions
            
            # Save insights and suggestions
            if self.output_manager:
                stage_name = workflow_node.stage.value if hasattr(workflow_node, 'stage') else "eda"
                
                self.output_manager.save_json(
                    stage_name,
                    {
                        "insights": insights,
                        "feature_suggestions": feature_suggestions,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "eda_insights.json",
                    stage_order
                )
        
        return result
    
    def _extract_insights(self, result: Dict[str, Any]) -> List[str]:
        """Extract key insights from EDA result"""
        text = result.get("raw_result", "")
        insights = []
        
        for line in text.split("\n"):
            if any(
                keyword in line.lower()
                for keyword in ["insight:", "finding:", "note:", "important:", "observation:"]
            ):
                insights.append(line.strip())
        
        return insights[:10]
    
    def _extract_feature_suggestions(self, result: Dict[str, Any]) -> List[str]:
        """Extract feature engineering suggestions"""
        text = result.get("raw_result", "")
        suggestions = []
        
        for line in text.split("\n"):
            if any(
                keyword in line.lower()
                for keyword in ["suggest:", "recommend:", "consider:", "try:", "feature:"]
            ):
                suggestions.append(line.strip())
        
        return suggestions[:10]


# ============================================================================
# OUTPUT-AWARE MODELING AGENT
# ============================================================================


class OutputAwareModelingAgent(OutputContextAwareAgent):
    """
    Statistical modeling specialist with output management.
    
    Responsibilities:
    - Specify models based on statistical plan
    - Fit models using statsmodels or PyMC
    - Run diagnostics
    - Compare model specifications
    
    All outputs (model code, fitted models, diagnostic plots, results)
    are automatically saved in organized directories.
    """
    
    def __init__(
        self,
        model: str = "openai:gpt-4",
        context_manager: Optional[ContextManager] = None,
        output_manager: Optional[OutputManager] = None,
    ):
        super().__init__(
            name="modeling",
            model=model,
            context_manager=context_manager,
            output_manager=output_manager,
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
- Save diagnostic plots with descriptive names (e.g., 'residual_plot.png', 'qq_plot.png')
- Save model summaries to text files

For classical models, use statsmodels.
For Bayesian models, use PyMC with appropriate priors.

Always validate results before finalizing.

Your models and diagnostics will be automatically saved and organized."""
    
    async def specify_model(
        self,
        statistical_plan: Dict[str, Any],
        data_info: Dict[str, Any],
        eda_insights: Dict[str, Any],
        workflow_node: Any,
        stage_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Specify model structure with output capture.
        
        Args:
            statistical_plan: Plan from statistician
            data_info: Information about data
            eda_insights: Insights from EDA
            workflow_node: Current workflow node
            stage_order: Stage order number
            
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
        
        return await self.execute_with_outputs(
            task=task,
            workflow_node=workflow_node,
            stage_order=stage_order
        )
    
    async def fit_model(
        self,
        model_spec: Dict[str, Any],
        workflow_node: Any,
        stage_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fit the specified model with output capture.
        
        Args:
            model_spec: Model specification
            workflow_node: Current workflow node
            stage_order: Stage order number
            
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
4. Save model summary to 'model_summary.txt'
5. Save fitted model for later use (use pickle)

Use successful patterns from past modeling work."""
        
        return await self.execute_with_outputs(
            task=task,
            workflow_node=workflow_node,
            stage_order=stage_order
        )
    
    async def run_diagnostics(
        self,
        fitted_model_info: Dict[str, Any],
        workflow_node: Any,
        stage_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run model diagnostics with output capture.
        
        Args:
            fitted_model_info: Information about fitted model
            workflow_node: Current workflow node
            stage_order: Stage order number
            
        Returns:
            Dict with diagnostic results
        """
        task = f"""Run model diagnostics:

Fitted Model:
{fitted_model_info}

Generate Python code to:
1. Check residual assumptions (normality, homoscedasticity)
   - Save Q-Q plot as 'qq_plot.png'
   - Save residual plot as 'residual_plot.png'
2. Look for influential observations
   - Save influence plot as 'influence_plot.png'
3. Check for multicollinearity (VIF)
4. Create residual plots
5. Run specification tests

Save all diagnostic plots with descriptive names.

Identify any issues and suggest remedies."""
        
        result = await self.execute_with_outputs(
            task=task,
            workflow_node=workflow_node,
            stage_order=stage_order
        )
        
        # Extract diagnostic findings
        if result["success"]:
            findings = self._extract_diagnostic_findings(result)
            result["diagnostic_findings"] = findings
            
            # Save findings
            if self.output_manager:
                stage_name = workflow_node.stage.value if hasattr(workflow_node, 'stage') else "model_diagnostics"
                
                self.output_manager.save_json(
                    stage_name,
                    {
                        "findings": findings,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "diagnostic_findings.json",
                    stage_order
                )
        
        return result
    
    def _extract_diagnostic_findings(self, result: Dict[str, Any]) -> List[str]:
        """Extract key diagnostic findings"""
        text = result.get("raw_result", "")
        findings = []
        
        for line in text.split("\n"):
            if any(
                keyword in line.lower()
                for keyword in ["violation:", "issue:", "problem:", "concern:", "pass:", "ok:"]
            ):
                findings.append(line.strip())
        
        return findings[:15]


# ============================================================================
# OUTPUT-AWARE INTERPRETER AGENT
# ============================================================================


class OutputAwareInterpreterAgent(OutputContextAwareAgent):
    """
    Model interpretation and reporting specialist with output management.
    
    Responsibilities:
    - Interpret model results
    - Assess practical significance
    - Generate natural language summaries
    - Create final reports
    - Provide recommendations
    
    All outputs (interpretations, reports, summary visualizations) are
    automatically saved in organized directories.
    """
    
    def __init__(
        self,
        model: str = "openai:gpt-4",
        context_manager: Optional[ContextManager] = None,
        output_manager: Optional[OutputManager] = None,
    ):
        super().__init__(
            name="interpreter",
            model=model,
            context_manager=context_manager,
            output_manager=output_manager,
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

Generate reports that decision-makers can act on.

Your interpretations and reports will be automatically saved and organized."""
    
    async def interpret_results(
        self,
        model_results: Dict[str, Any],
        diagnostics: Dict[str, Any],
        objectives: List[str],
        workflow_node: Any,
        stage_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Interpret model results with output capture.
        
        Args:
            model_results: Results from modeling
            diagnostics: Diagnostic findings
            objectives: Original analysis objectives
            workflow_node: Current workflow node
            stage_order: Stage order number
            
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
6. Recommendations

Save any summary visualizations you create."""
        
        return await self.execute_with_outputs(
            task=task,
            workflow_node=workflow_node,
            stage_order=stage_order
        )
    
    async def generate_report(
        self,
        interpretation: Dict[str, Any],
        all_outputs: Dict[str, Any],
        workflow_node: Any,
        stage_order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report with output capture.
        
        Args:
            interpretation: Interpretation of results
            all_outputs: All outputs from workflow
            workflow_node: Current workflow node
            stage_order: Stage order number
            
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
3. Key Findings (reference visualizations created in previous stages)
4. Detailed Results
5. Recommendations
6. Technical Appendix

Format for both technical and non-technical readers.
Save report as markdown file."""
        
        result = await self.execute_with_outputs(
            task=task,
            workflow_node=workflow_node,
            stage_order=stage_order
        )
        
        # Save final report as markdown if generated
        if result["success"] and self.output_manager:
            stage_name = workflow_node.stage.value if hasattr(workflow_node, 'stage') else "report_generation"
            
            # Extract report text from result
            report_text = result.get("raw_result", "")
            
            self.output_manager.save_text(
                stage_name,
                report_text,
                "final_report.md",
                stage_order
            )
        
        return result


# ============================================================================
# AGENT FACTORY
# ============================================================================


class OutputAwareAgentFactory:
    """Factory for creating output-aware specialist agents"""
    
    @staticmethod
    def create_agent(
        agent_type: str,
        model: str = "openai:gpt-4",
        context_manager: Optional[ContextManager] = None,
        output_manager: Optional[OutputManager] = None,
    ) -> OutputContextAwareAgent:
        """
        Create an output-aware specialist agent by type.
        
        Args:
            agent_type: Type of agent (statistician, data_engineer, etc.)
            model: LLM model to use
            context_manager: Context manager instance
            output_manager: Output manager instance
            
        Returns:
            Output-aware specialist agent instance
        """
        agents = {
            "statistician": OutputAwareStatisticianAgent,
            "data_engineer": OutputAwareDataEngineerAgent,
            "eda": OutputAwareEDAAgent,
            "modeling": OutputAwareModelingAgent,
            "interpreter": OutputAwareInterpreterAgent,
        }
        
        agent_class = agents.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return agent_class(
            model=model,
            context_manager=context_manager,
            output_manager=output_manager
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        from datascienceagent.core.context_management import ContextManager
        from datascienceagent.core.output_manager import OutputManager
        
        print("\n" + "="*70)
        print("OUTPUT-AWARE SPECIALIST AGENTS DEMO")
        print("="*70)
        
        # Initialize managers
        context_mgr = ContextManager(model="openai:gpt-4.1-mini")
        context_mgr.start_session(topic="demo_analysis")
        
        output_mgr = OutputManager(workflow_id="demo_agents")
        
        # Create agents using factory
        agents = {
            "statistician": OutputAwareAgentFactory.create_agent(
                "statistician", "openai:gpt-4.1-mini", context_mgr, output_mgr
            ),
            "data_engineer": OutputAwareAgentFactory.create_agent(
                "data_engineer", "openai:gpt-4.1-mini", context_mgr, output_mgr
            ),
            "eda": OutputAwareAgentFactory.create_agent(
                "eda", "openai:gpt-4.1-mini", context_mgr, output_mgr
            ),
        }
        
        print(f"\n✅ Created {len(agents)} output-aware agents")
        for name, agent in agents.items():
            print(f"   - {agent.name}")
        
        print(f"\n📁 Outputs will be saved to: {output_mgr.workflow_dir}")
        
        # Save manifest
        output_mgr.save_manifest()
        output_mgr.create_summary_report()
        
        print("\n✅ Demo complete!")
    
    asyncio.run(demo())