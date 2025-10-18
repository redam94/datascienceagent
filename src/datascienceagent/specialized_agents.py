"""
Specialized Agent Implementations
EDA Agent, Modeling Agent, and Interpreter Agent
"""

import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from pydantic_ai import Agent

# Import base classes (in production, use proper imports)
# from data_science_agent_system import BaseDataScienceAgent, Task, TaskType, AgentResponse


# ============================================================================
# EDA AGENT
# ============================================================================

class PlotSpec(BaseModel):
    """Specification for a plot"""
    plot_type: str  # scatter, histogram, box, line, heatmap
    x: Optional[str] = None
    y: Optional[str] = None
    hue: Optional[str] = None
    title: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EDAReport(BaseModel):
    """Report from exploratory data analysis"""
    summary_statistics: Dict[str, Any]
    distributions: Dict[str, Any]
    correlations: Optional[Dict[str, float]] = None
    outliers_detected: List[str] = Field(default_factory=list)
    missing_data_summary: Dict[str, int] = Field(default_factory=dict)
    recommended_transformations: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    plots_generated: List[str] = Field(default_factory=list)


class EDAAgent:
    """Exploratory Data Analysis Agent"""
    
    def __init__(self, model: str = "openai:gpt-4"):
        self.name = "eda_agent"
        self.model = model
        # self.capabilities = [TaskType.EDA]  # Would use actual TaskType
        
        self.system_prompt = """You are an expert in exploratory data analysis and data visualization.

Your expertise includes:
- Understanding data distributions and patterns
- Creating informative visualizations
- Identifying outliers and anomalies
- Detecting relationships between variables
- Suggesting feature engineering
- Writing clean, efficient plotting code

When analyzing data:
1. Start with univariate analysis (distributions, summary stats)
2. Move to bivariate analysis (relationships, correlations)
3. Identify data quality issues
4. Look for non-linear patterns
5. Check for multicollinearity
6. Suggest transformations if needed

Always explain what you observe and why it matters."""

        self.agent = Agent(model, system_prompt=self.system_prompt)
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute EDA task"""
        
        # In production, would receive actual dataframe
        data_info = task.get("input_data", {})
        
        print(f"  📊 Analyzing data: {data_info.get('n_rows', 0)} rows, {data_info.get('n_cols', 0)} columns")
        
        # Generate EDA plan
        eda_plan = await self._create_eda_plan(data_info)
        
        # Execute analysis
        eda_report = await self._perform_eda(data_info, eda_plan)
        
        # Generate visualizations
        plots = await self._generate_visualizations(data_info, eda_report)
        
        return {
            "agent": self.name,
            "status": "success",
            "result": {
                "eda_report": eda_report.model_dump(),
                "plots": plots,
                "code_generated": self._generate_eda_code(data_info)
            },
            "next_suggestions": [
                "Data appears suitable for modeling",
                "Consider log transformation for skewed variables",
                "Check for multicollinearity before regression"
            ]
        }
    
    async def _create_eda_plan(self, data_info: Dict[str, Any]) -> List[str]:
        """Create EDA plan based on data characteristics"""
        
        prompt = f"""Create an EDA plan for this dataset:

Rows: {data_info.get('n_rows', 'unknown')}
Columns: {data_info.get('n_cols', 'unknown')}
Column Types: {data_info.get('column_types', {})}

List the specific analyses to perform:
1. Summary statistics needed
2. Distributions to examine
3. Relationships to explore
4. Visualizations to create
5. Data quality checks

Be specific and prioritized."""

        result = await self.agent.run(prompt)
        
        # Parse into list (simplified)
        return [
            "summary_statistics",
            "distribution_analysis",
            "correlation_analysis",
            "outlier_detection",
            "missing_data_analysis"
        ]
    
    async def _perform_eda(
        self,
        data_info: Dict[str, Any],
        eda_plan: List[str]
    ) -> EDAReport:
        """Perform the EDA analysis"""
        
        # Simulate analysis results
        await asyncio.sleep(0.3)
        
        return EDAReport(
            summary_statistics={
                "mean_x": 50.5,
                "std_x": 15.2,
                "mean_y": 100.3,
                "std_y": 25.7
            },
            distributions={
                "x_distribution": "approximately normal",
                "y_distribution": "right-skewed",
                "skewness": {"x": 0.1, "y": 1.5}
            },
            correlations={
                "x_y": 0.75,
                "x_z": 0.3
            },
            outliers_detected=["Row 150: y value extreme", "Row 890: potential data entry error"],
            missing_data_summary={
                "x": 0,
                "y": 5,
                "z": 12
            },
            recommended_transformations=[
                "Log transform y (right-skewed)",
                "Standardize all features for modeling"
            ],
            insights=[
                "Strong positive correlation between x and y",
                "Seasonality detected in time series",
                "Some outliers may be legitimate extreme values",
                "Missing data appears to be missing completely at random (MCAR)"
            ],
            plots_generated=[
                "histogram_x.png",
                "scatter_x_vs_y.png",
                "correlation_heatmap.png",
                "boxplot_outliers.png"
            ]
        )
    
    async def _generate_visualizations(
        self,
        data_info: Dict[str, Any],
        eda_report: EDAReport
    ) -> List[Dict[str, Any]]:
        """Generate visualization specifications and code"""
        
        plots = []
        
        # Distribution plots
        plots.append({
            "type": "histogram",
            "title": "Distribution of X",
            "code": self._generate_plot_code("histogram", "x"),
            "interpretation": "X follows approximately normal distribution with slight right skew"
        })
        
        # Scatter plot
        plots.append({
            "type": "scatter",
            "title": "X vs Y Relationship",
            "code": self._generate_plot_code("scatter", "x", "y"),
            "interpretation": "Strong positive linear relationship (r=0.75)"
        })
        
        # Correlation heatmap
        plots.append({
            "type": "heatmap",
            "title": "Correlation Matrix",
            "code": self._generate_plot_code("heatmap"),
            "interpretation": "High correlation between x and y, potential multicollinearity to monitor"
        })
        
        return plots
    
    def _generate_plot_code(
        self,
        plot_type: str,
        x: Optional[str] = None,
        y: Optional[str] = None
    ) -> str:
        """Generate Python plotting code"""
        
        if plot_type == "histogram":
            return f"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data=df['{x}'], kde=True, bins=30)
plt.title('Distribution of {x}')
plt.xlabel('{x}')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram_{x}.png')
plt.close()
"""
        
        elif plot_type == "scatter":
            return f"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='{x}', y='{y}', alpha=0.6)
plt.title('{x} vs {y}')
plt.xlabel('{x}')
plt.ylabel('{y}')
plt.tight_layout()
plt.savefig('scatter_{x}_vs_{y}.png')
plt.close()
"""
        
        elif plot_type == "heatmap":
            return """
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()
"""
        
        return "# Plot code would be generated here"
    
    def _generate_eda_code(self, data_info: Dict[str, Any]) -> str:
        """Generate complete EDA script"""
        
        return """
# Exploratory Data Analysis Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv('data.csv')

# Summary statistics
print(df.describe())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Distribution analysis
for col in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    
    plt.subplot(1, 2, 2)
    stats.probplot(df[col], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {col}')
    
    plt.tight_layout()
    plt.savefig(f'distribution_{col}.png')
    plt.close()

# Correlation analysis
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig('correlation_heatmap.png')
plt.close()

# Outlier detection using IQR
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f'{col}: {len(outliers)} outliers detected')

print("EDA complete!")
"""


# ============================================================================
# MODELING AGENT
# ============================================================================

class ModelSpec(BaseModel):
    """Model specification"""
    method: str  # linear_regression, logistic_regression, bayesian_regression, etc.
    formula: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    library: str = "statsmodels"  # or pymc


class FittedModel(BaseModel):
    """Results from a fitted model"""
    model_id: str
    method: str
    library: str
    summary: str
    coefficients: Dict[str, float] = Field(default_factory=dict)
    statistics: Dict[str, float] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    code: str


class ModelingAgent:
    """Statistical Modeling Agent"""
    
    def __init__(self, model: str = "openai:gpt-4"):
        self.name = "modeling_agent"
        self.model = model
        
        self.system_prompt = """You are an expert statistical modeler with deep knowledge of:
- Classical statistics (statsmodels)
- Bayesian inference (PyMC)
- Regression analysis (linear, logistic, multilevel)
- Time series modeling
- Hypothesis testing
- Model diagnostics

When building models:
1. Choose appropriate method for the problem
2. Write clean, well-documented code
3. Include comprehensive diagnostics
4. Validate assumptions
5. Assess model fit
6. Check for influential points
7. Consider alternative specifications

Always prioritize interpretability and statistical validity."""

        self.agent = Agent(model, system_prompt=self.system_prompt)
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute modeling task"""
        
        input_data = task.get("input_data", {})
        problem_type = input_data.get("problem_type", "regression")
        
        print(f"  🔬 Building {problem_type} model")
        
        # Create model specification
        model_spec = await self._create_model_spec(input_data)
        
        # Generate modeling code
        modeling_code = await self._generate_modeling_code(model_spec)
        
        # Simulate model fitting
        fitted_model = await self._fit_model(model_spec, modeling_code)
        
        return {
            "agent": self.name,
            "status": "success",
            "result": {
                "model": fitted_model.model_dump(),
                "code": modeling_code
            },
            "artifacts": ["model_summary.txt", "diagnostic_plots.png"],
            "next_suggestions": [
                "Model diagnostics look good",
                "Proceed to interpretation",
                "Consider adding interaction terms"
            ]
        }
    
    async def _create_model_spec(self, input_data: Dict[str, Any]) -> ModelSpec:
        """Create model specification"""
        
        problem_type = input_data.get("problem_type", "regression")
        target = input_data.get("target_variable", "y")
        features = input_data.get("features", ["x1", "x2"])
        
        prompt = f"""Specify a statistical model for this problem:

Problem Type: {problem_type}
Target Variable: {target}
Features: {', '.join(features)}

Provide:
1. Recommended method (e.g., OLS, Logistic, Mixed Effects)
2. Model formula (R-style)
3. Key parameters to set
4. Diagnostic tests to run

Be specific about the statsmodels or PyMC implementation."""

        result = await self.agent.run(prompt)
        
        # Parse response (simplified)
        return ModelSpec(
            method="OLS",
            formula=f"{target} ~ {' + '.join(features)}",
            parameters={"cov_type": "HC3"},
            library="statsmodels"
        )
    
    async def _generate_modeling_code(self, spec: ModelSpec) -> str:
        """Generate Python modeling code"""
        
        if spec.library == "statsmodels":
            return self._generate_statsmodels_code(spec)
        elif spec.library == "pymc":
            return self._generate_pymc_code(spec)
        else:
            return "# Unknown library"
    
    def _generate_statsmodels_code(self, spec: ModelSpec) -> str:
        """Generate statsmodels code"""
        
        return f"""
# Statistical Modeling with statsmodels
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Fit model using formula
model = smf.ols(formula='{spec.formula}', data=df)
results = model.fit(cov_type='{spec.parameters.get("cov_type", "nonrobust")}')

# Print summary
print(results.summary())

# Model Diagnostics
print("\\n=== DIAGNOSTICS ===")

# 1. Residual analysis
residuals = results.resid
fitted = results.fittedvalues

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Residuals vs Fitted
axes[0, 0].scatter(fitted, residuals, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
sm.qqplot(residuals, line='s', ax=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')

# Scale-Location plot
axes[1, 0].scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.5)
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('√|Residuals|')
axes[1, 0].set_title('Scale-Location Plot')

# Residuals histogram
axes[1, 1].hist(residuals, bins=30, edgecolor='black')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residual Distribution')

plt.tight_layout()
plt.savefig('diagnostic_plots.png')
plt.close()

# 2. Heteroscedasticity test (Breusch-Pagan)
lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, results.model.exog)
print(f"\\nBreusch-Pagan Test for Heteroscedasticity:")
print(f"  LM Statistic: {{lm:.4f}}, p-value: {{lm_pvalue:.4f}}")
if lm_pvalue < 0.05:
    print("  ⚠️  Evidence of heteroscedasticity")
else:
    print("  ✓ Homoscedasticity assumption satisfied")

# 3. Multicollinearity (VIF)
print("\\nVariance Inflation Factors:")
vif_data = pd.DataFrame()
vif_data["Variable"] = results.model.exog_names[1:]  # Exclude intercept
vif_data["VIF"] = [variance_inflation_factor(results.model.exog, i) 
                   for i in range(1, results.model.exog.shape[1])]
print(vif_data)
if (vif_data["VIF"] > 10).any():
    print("  ⚠️  High VIF detected, potential multicollinearity")

# 4. Influential observations
influence = results.get_influence()
cooks_d = influence.cooks_distance[0]
print(f"\\nInfluential Observations (Cook's D > 1):")
influential = np.where(cooks_d > 1)[0]
if len(influential) > 0:
    print(f"  Observations: {{influential}}")
else:
    print("  None detected")

# 5. Model fit statistics
print(f"\\n=== MODEL FIT ===")
print(f"R-squared: {{results.rsquared:.4f}}")
print(f"Adjusted R-squared: {{results.rsquared_adj:.4f}}")
print(f"AIC: {{results.aic:.2f}}")
print(f"BIC: {{results.bic:.2f}}")

# Save results
with open('model_summary.txt', 'w') as f:
    f.write(str(results.summary()))

print("\\nModeling complete!")
"""
    
    def _generate_pymc_code(self, spec: ModelSpec) -> str:
        """Generate PyMC Bayesian modeling code"""
        
        return f"""
# Bayesian Modeling with PyMC
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# Prepare data
# Parse formula: {spec.formula}
target_var = df['y'].values
features = df[['x1', 'x2']].values

# Build Bayesian model
with pm.Model() as model:
    # Priors
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    slopes = pm.Normal('slopes', mu=0, sigma=10, shape=features.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Linear model
    mu = intercept + pm.math.dot(features, slopes)
    
    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=target_var)
    
    # Sample from posterior
    trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

# Print summary
print(az.summary(trace))

# Diagnostic plots
az.plot_trace(trace)
plt.tight_layout()
plt.savefig('trace_plots.png')
plt.close()

az.plot_posterior(trace)
plt.tight_layout()
plt.savefig('posterior_plots.png')
plt.close()

# Check convergence
print("\\n=== CONVERGENCE DIAGNOSTICS ===")
print(az.rhat(trace))

# Posterior predictive checks
with model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

az.plot_ppc(ppc)
plt.savefig('posterior_predictive_check.png')
plt.close()

print("\\nBayesian modeling complete!")
"""
    
    async def _fit_model(self, spec: ModelSpec, code: str) -> FittedModel:
        """Simulate model fitting"""
        
        await asyncio.sleep(0.5)  # Simulate computation
        
        return FittedModel(
            model_id="model_001",
            method=spec.method,
            library=spec.library,
            summary="OLS Regression Results with robust standard errors",
            coefficients={
                "intercept": 10.5,
                "x1": 2.3,
                "x2": -1.7
            },
            statistics={
                "r_squared": 0.75,
                "adj_r_squared": 0.74,
                "f_statistic": 150.5,
                "f_pvalue": 0.0001,
                "aic": 500.2,
                "bic": 510.5
            },
            diagnostics={
                "heteroscedasticity_test": {"statistic": 2.5, "pvalue": 0.28, "ok": True},
                "normality_test": {"statistic": 1.2, "pvalue": 0.55, "ok": True},
                "autocorrelation_test": {"statistic": 0.8, "pvalue": 0.67, "ok": True},
                "vif_max": 2.3,
                "influential_points": 2
            },
            code=code
        )


# ============================================================================
# INTERPRETER AGENT
# ============================================================================

class Interpretation(BaseModel):
    """Model interpretation results"""
    executive_summary: str
    key_findings: List[str]
    coefficient_interpretations: Dict[str, str]
    statistical_significance: Dict[str, Dict[str, Any]]
    practical_significance: str
    model_fit_assessment: str
    assumptions_check: Dict[str, bool]
    limitations: List[str]
    recommendations: List[str]
    confidence_rating: float = Field(ge=0, le=1)


class InterpreterAgent:
    """Model Interpretation and Reporting Agent"""
    
    def __init__(self, model: str = "openai:gpt-4"):
        self.name = "interpreter_agent"
        self.model = model
        
        self.system_prompt = """You are an expert at interpreting statistical results and communicating them clearly.

Your expertise includes:
- Translating statistical output into plain language
- Assessing practical vs statistical significance
- Explaining coefficients and their meaning
- Evaluating model fit and diagnostics
- Identifying limitations and caveats
- Making data-driven recommendations

When interpreting results:
1. Start with the big picture (executive summary)
2. Explain coefficients in context
3. Distinguish statistical from practical significance
4. Assess model assumptions and fit
5. Be transparent about limitations
6. Provide actionable recommendations
7. Use appropriate confidence qualifiers

Write for a business audience while maintaining statistical accuracy."""

        self.agent = Agent(model, system_prompt=self.system_prompt)
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute interpretation task"""
        
        input_data = task.get("input_data", {})
        model_results = input_data.get("model", {})
        
        print(f"  📝 Interpreting model results")
        
        # Generate interpretation
        interpretation = await self._interpret_model(model_results)
        
        # Generate final report
        report = await self._generate_report(interpretation)
        
        return {
            "agent": self.name,
            "status": "success",
            "result": {
                "interpretation": interpretation.model_dump(),
                "report": report
            },
            "artifacts": ["final_report.pdf"],
            "next_suggestions": [
                "Share results with stakeholders",
                "Consider follow-up analyses",
                "Monitor model performance over time"
            ]
        }
    
    async def _interpret_model(self, model_results: Dict[str, Any]) -> Interpretation:
        """Interpret model results"""
        
        coefficients = model_results.get("coefficients", {})
        statistics = model_results.get("statistics", {})
        diagnostics = model_results.get("diagnostics", {})
        
        prompt = f"""Interpret these statistical model results:

COEFFICIENTS:
{coefficients}

MODEL STATISTICS:
{statistics}

DIAGNOSTICS:
{diagnostics}

Provide:
1. Executive summary (2-3 sentences for business audience)
2. Key findings (3-5 main insights)
3. Coefficient interpretations (what each means practically)
4. Statistical vs practical significance
5. Model fit assessment
6. Assumption violations or concerns
7. Limitations and caveats
8. Recommendations for action or further analysis

Be clear, accurate, and actionable."""

        result = await self.agent.run(prompt)
        response = str(result.data)
        
        # Parse response into structured interpretation
        return Interpretation(
            executive_summary="The model shows strong predictive power (R²=0.75) with all key predictors being statistically significant. Marketing spend has a positive effect on sales, while competitor activity shows negative impact.",
            key_findings=[
                "Marketing spend has strong positive effect on sales (β=2.3, p<0.001)",
                "Competitor activity negatively impacts sales (β=-1.7, p<0.01)",
                "Model explains 75% of variance in sales",
                "All diagnostic tests indicate model assumptions are met",
                "No concerning influential observations detected"
            ],
            coefficient_interpretations={
                "intercept": "Baseline sales level with no marketing spend: $10,500",
                "marketing_spend": "Each $1000 increase in marketing spend associated with $2,300 increase in sales",
                "competitor_activity": "Each unit increase in competitor activity associated with $1,700 decrease in sales"
            },
            statistical_significance={
                "marketing_spend": {"p_value": 0.0001, "significant": True, "effect_size": "large"},
                "competitor_activity": {"p_value": 0.005, "significant": True, "effect_size": "medium"}
            },
            practical_significance="The effect sizes are large enough to inform business decisions. A typical marketing campaign ($50K) would generate approximately $115K in additional sales.",
            model_fit_assessment="Excellent model fit with R²=0.75 and all diagnostics passing. The model is well-specified and reliable for predictions within the data range.",
            assumptions_check={
                "linearity": True,
                "independence": True,
                "homoscedasticity": True,
                "normality": True
            },
            limitations=[
                "Results based on historical data - may not generalize to future",
                "Cannot establish causality without experimental design",
                "Model doesn't account for time-varying effects",
                "External validity limited to similar market conditions"
            ],
            recommendations=[
                "Increase marketing spend - ROI is strongly positive",
                "Monitor competitor activity closely and adjust strategies",
                "Consider A/B testing to establish causal effects",
                "Re-fit model quarterly to capture changing dynamics",
                "Collect data on additional potential predictors (seasonality, economy)"
            ],
            confidence_rating=0.85
        )
    
    async def _generate_report(self, interpretation: Interpretation) -> str:
        """Generate final report"""
        
        report = f"""
# STATISTICAL ANALYSIS REPORT

## Executive Summary

{interpretation.executive_summary}

**Confidence Level**: {interpretation.confidence_rating:.0%}

---

## Key Findings

{self._format_list(interpretation.key_findings)}

---

## Detailed Results

### Model Coefficients

{self._format_dict(interpretation.coefficient_interpretations)}

### Statistical Significance

{self._format_significance(interpretation.statistical_significance)}

### Practical Significance

{interpretation.practical_significance}

---

## Model Quality Assessment

{interpretation.model_fit_assessment}

### Assumptions Check

{self._format_dict({k: "✓ Satisfied" if v else "✗ Violated" for k, v in interpretation.assumptions_check.items()})}

---

## Limitations

{self._format_list(interpretation.limitations)}

---

## Recommendations

{self._format_list(interpretation.recommendations)}

---

*Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return report
    
    def _format_list(self, items: List[str]) -> str:
        """Format list for report"""
        return "\n".join(f"- {item}" for item in items)
    
    def _format_dict(self, items: Dict[str, str]) -> str:
        """Format dictionary for report"""
        return "\n".join(f"- **{k}**: {v}" for k, v in items.items())
    
    def _format_significance(self, sig_dict: Dict[str, Dict[str, Any]]) -> str:
        """Format significance results"""
        lines = []
        for var, stats in sig_dict.items():
            sig_marker = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*"
            lines.append(f"- **{var}**: p={stats['p_value']:.4f} {sig_marker} (Effect size: {stats['effect_size']})")
        return "\n".join(lines)


# ============================================================================
# DEMO
# ============================================================================

async def demo_specialized_agents():
    """Demonstrate the specialized agents"""
    
    print("\n" + "="*70)
    print("🤖 SPECIALIZED AGENTS DEMONSTRATION")
    print("="*70)
    
    # Create agents
    eda_agent = EDAAgent()
    modeling_agent = ModelingAgent()
    interpreter_agent = InterpreterAgent()
    
    # Simulate workflow data
    data_info = {
        "n_rows": 1000,
        "n_cols": 10,
        "column_types": {"x1": "float", "x2": "float", "y": "float"},
        "target_variable": "y",
        "features": ["x1", "x2"]
    }
    
    # 1. EDA
    print("\n📊 PHASE 1: Exploratory Data Analysis")
    print("-" * 70)
    eda_task = {
        "type": "eda",
        "input_data": data_info
    }
    eda_result = await eda_agent.execute_task(eda_task)
    print(f"✓ Generated {len(eda_result['result']['plots'])} visualizations")
    print(f"✓ Key insights: {len(eda_result['result']['eda_report']['insights'])}")
    
    # 2. Modeling
    print("\n🔬 PHASE 2: Statistical Modeling")
    print("-" * 70)
    modeling_task = {
        "type": "modeling",
        "input_data": {
            **data_info,
            "problem_type": "regression"
        }
    }
    modeling_result = await modeling_agent.execute_task(modeling_task)
    model = modeling_result['result']['model']
    print(f"✓ Fitted {model['method']} model")
    print(f"✓ R² = {model['statistics']['r_squared']:.3f}")
    print(f"✓ All diagnostics passed")
    
    # 3. Interpretation
    print("\n📝 PHASE 3: Result Interpretation")
    print("-" * 70)
    interpretation_task = {
        "type": "interpretation",
        "input_data": {
            "model": model
        }
    }
    interp_result = await interpreter_agent.execute_task(interpretation_task)
    interpretation = interp_result['result']['interpretation']
    
    print("\n" + interp_result['result']['report'])
    
    print("\n✅ All specialized agents demonstrated successfully!")


if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(demo_specialized_agents())
