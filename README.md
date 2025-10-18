# Data Science Multi-Agent System - Quick Start Guide

## Overview

This is a sophisticated multi-agent system for automated data science workflows. The system uses specialized AI agents coordinated by an orchestrator to perform end-to-end data analysis, from data loading through statistical modeling to interpretation.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Orchestrator Agent                     │
│         (Plans & Coordinates Workflow)              │
└──────────┬──────────────────────────────────────────┘
           │
    ┌──────┴──────────────────────────────────┐
    │                                         │
┌───▼────────┐  ┌─────────────┐  ┌────────────▼────┐
│Statistician│  │Data Engineer│  │EDA Agent        │
│Agent       │  │Agent        │  │                 │
└────────────┘  └─────────────┘  └─────────────────┘
    │                                          │
┌───▼────────┐                    ┌────────────▼────┐
│Modeling    │                    │Interpreter Agent│
│Agent       │                    │                 │
└────────────┘                    └─────────────────┘
```

## Components

### 1. **Orchestrator Agent**
- Parses user requests
- Creates execution plans
- Coordinates specialized agents
- Aggregates results

### 2. **Statistician Agent**
- Recommends statistical methods
- Researches best practices
- Validates analysis plans
- Has access to web research

### 3. **Data Engineer Agent**
- Loads data from various sources
- Writes custom loading scripts
- Performs data quality checks
- Handles transformations

### 4. **EDA Agent**
- Creates visualizations
- Writes plotting code
- Interprets distributions
- Suggests feature engineering

### 5. **Modeling Agent**
- Implements statsmodels & PyMC models
- Writes analysis scripts
- Performs diagnostics
- Handles model fitting

### 6. **Interpreter Agent**
- Interprets results
- Generates reports
- Provides recommendations
- Communicates findings

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd data-science-agent-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file:

```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: Database
DATABASE_URL=postgresql://user:pass@localhost/dbname

# Optional: Redis
REDIS_URL=redis://localhost:6379
```

---

## Quick Start

### Option 1: Run Demo Script

```bash
python data_science_agent_system.py
```

This will demonstrate the orchestrator coordinating multiple agents.

### Option 2: Run FastAPI Server

```bash
python fastapi_server.py
```

Or with uvicorn:

```bash
uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000
```

Then visit:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Option 3: Use Python API Directly

```python
import asyncio
from data_science_agent_system import OrchestratorAgent, WorkflowEngine
from specialized_agents import (
    StatisticianAgent, DataEngineerAgent, 
    EDAAgent, ModelingAgent, InterpreterAgent
)

async def analyze_data():
    # Create orchestrator
    orchestrator = OrchestratorAgent()
    
    # Register agents
    orchestrator.register_agent(StatisticianAgent())
    orchestrator.register_agent(DataEngineerAgent())
    orchestrator.register_agent(EDAAgent())
    orchestrator.register_agent(ModelingAgent())
    orchestrator.register_agent(InterpreterAgent())
    
    # Create workflow engine
    engine = WorkflowEngine(orchestrator)
    
    # Start analysis
    query = "Analyze the relationship between X and Y"
    workflow_id = await engine.start_analysis(query)
    
    # Execute
    result = await engine.execute_workflow(workflow_id)
    
    # Get report
    report = await orchestrator.synthesize_results(result)
    print(report)

asyncio.run(analyze_data())
```

---

## API Usage

### Start an Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/start" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze sales data and identify key drivers",
    "data_source": {
      "type": "csv",
      "location": "/data/sales.csv"
    },
    "options": {
      "include_diagnostics": true,
      "generate_plots": true
    }
  }'
```

### Check Status

```bash
curl "http://localhost:8000/api/v1/analysis/{analysis_id}"
```

### Get Results

```bash
curl "http://localhost:8000/api/v1/analysis/{analysis_id}/results"
```

### WebSocket for Real-time Updates

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analysis/{analysis_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress + '%');
  console.log('Stage:', data.current_stage);
};
```

---

## Example Workflows

### 1. Simple Regression Analysis

```python
query = """
I have data with variables X and Y.
I want to understand if X predicts Y.
"""

# System will:
# 1. Research appropriate methods (OLS regression)
# 2. Load and validate data
# 3. Create EDA visualizations
# 4. Fit regression model with diagnostics
# 5. Interpret results and provide recommendations
```

### 2. Complex Analysis with Multiple Variables

```python
query = """
Analyze sales data with:
- Marketing spend
- Competitor activity
- Seasonality
- Economic indicators

Determine what drives sales and build a predictive model.
"""

# System will:
# 1. Plan comprehensive analysis
# 2. Load and clean data
# 3. Perform thorough EDA
# 4. Test multiple model specifications
# 5. Validate assumptions
# 6. Compare models
# 7. Generate business recommendations
```

### 3. Bayesian Analysis

```python
query = """
Perform Bayesian regression to estimate the effect of 
treatment on outcome with uncertainty quantification.
"""

# System will:
# 1. Design Bayesian model in PyMC
# 2. Specify priors
# 3. Sample from posterior
# 4. Check convergence
# 5. Interpret posterior distributions
# 6. Generate credible intervals
```

---

## Project Structure

```
data_science_agent_system/
├── data_science_agent_system.py    # Core system & orchestrator
├── specialized_agents.py            # Specialized agent implementations
├── fastapi_server.py                # REST API server
├── requirements.txt                 # Dependencies
├── data_science_agent_implementation.md  # Full documentation
├── README.md                        # This file
├── .env                            # Environment variables (create this)
├── tests/                          # Test suite
│   ├── test_agents.py
│   ├── test_orchestrator.py
│   └── test_api.py
├── outputs/                        # Generated outputs
│   ├── plots/
│   ├── models/
│   └── reports/
└── data/                           # Sample data
    └── examples/
```

---

## Development Roadmap

### Phase 1: ✅ Foundation (Current)
- [x] Base agent architecture
- [x] Orchestrator implementation
- [x] Core specialized agents
- [x] FastAPI server
- [x] Basic documentation

### Phase 2: 🚧 Enhancement (In Progress)
- [ ] Actual data loading implementations
- [ ] Code execution sandbox
- [ ] Real visualization generation
- [ ] PyMC Bayesian modeling
- [ ] Web search integration

### Phase 3: 📋 Advanced Features
- [ ] Persistent storage
- [ ] Result caching
- [ ] Model versioning
- [ ] A/B testing support
- [ ] Causal inference tools

### Phase 4: 🎯 Production
- [ ] Authentication & authorization
- [ ] Rate limiting
- [ ] Monitoring & alerting
- [ ] Docker deployment
- [ ] Kubernetes support

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suite

```bash
pytest tests/test_agents.py -v
pytest tests/test_orchestrator.py -v
pytest tests/test_api.py -v
```

### Test Coverage

```bash
pytest --cov=. tests/
```

---

## Configuration

### Model Selection

By default, agents use `openai:gpt-4`. You can configure this:

```python
orchestrator = OrchestratorAgent(model="anthropic:claude-3-opus-20240229")
```

### Agent Customization

```python
# Custom statistician with specific expertise
class MyStatisticianAgent(StatisticianAgent):
    def default_system_prompt(self):
        return """You are a statistician specializing in 
        time series analysis and causal inference..."""

orchestrator.register_agent(MyStatisticianAgent())
```

---

## Troubleshooting

### Issue: "Module not found"

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "API key not found"

```bash
# Check .env file exists and has correct keys
cat .env

# Export manually for testing
export OPENAI_API_KEY=your_key_here
```

### Issue: "Port already in use"

```bash
# Change port in fastapi_server.py or run:
uvicorn fastapi_server:app --port 8001
```

---

## Best Practices

### 1. Data Preparation

- Ensure data is clean and well-formatted
- Document data sources and transformations
- Check for missing values and outliers
- Validate data types

### 2. Analysis Planning

- Start with clear research questions
- Consider sample size and power
- Plan for assumption validation
- Think about alternative specifications

### 3. Model Building

- Always check diagnostics
- Validate assumptions
- Consider multiple model specifications
- Document modeling choices

### 4. Interpretation

- Focus on practical significance
- Be transparent about limitations
- Provide confidence intervals
- Make actionable recommendations

---

## Contributing

We welcome contributions! Areas for contribution:

1. **New Agents**: Add specialized agents (e.g., time series, ML)
2. **Tools**: Add new data sources, plotting libraries
3. **Methods**: Implement additional statistical methods
4. **Documentation**: Improve docs and examples
5. **Testing**: Add test cases and scenarios

---

## Performance Tips

### 1. Caching

Enable Redis caching for repeated analyses:

```python
engine = WorkflowEngine(orchestrator, use_cache=True)
```

### 2. Parallel Execution

For independent tasks:

```python
# In orchestrator, execute tasks in parallel
results = await asyncio.gather(*tasks)
```

### 3. Resource Limits

Set timeouts and resource limits:

```python
orchestrator = OrchestratorAgent(
    timeout_seconds=300,
    max_memory_mb=4096
)
```

---

## Security Considerations

### 1. Code Execution

The modeling agent executes generated code. In production:

- Use sandboxed execution environments
- Implement code review before execution
- Set resource limits
- Monitor for malicious patterns

### 2. Data Privacy

- Never log sensitive data
- Implement access controls
- Use encryption for data at rest
- Audit data access

### 3. API Security

- Implement authentication
- Use rate limiting
- Validate all inputs
- Enable CORS carefully

---

## Support

- **Documentation**: See `data_science_agent_implementation.md`
- **Issues**: Open GitHub issues
- **Questions**: Use discussions or Q&A

---

## License

[Your License Here]

---

## Acknowledgments

Built with:
- PydanticAI for agent orchestration
- FastAPI for web framework
- PyMC for Bayesian inference
- statsmodels for classical statistics

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Status**: Beta
