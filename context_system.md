# Context Management & RAG System Documentation

## Overview

The Context Management and RAG (Retrieval-Augmented Generation) system provides intelligent memory and learning capabilities for the data science agent system. It solves the problem of growing context by chunking, storing, and intelligently retrieving relevant information.

## Key Features

### 1. **Intelligent Context Chunking**
- Automatically chunks long content into manageable pieces
- Preserves semantic meaning across chunks
- Handles different content types (text, code, structured data)
- Maintains chunk overlap for continuity

### 2. **Vector Storage with ChromaDB**
- Persistent storage of all agent interactions
- Semantic search capabilities
- Metadata-based filtering
- Fast retrieval of relevant context

### 3. **Session Management**
- Track related analyses within sessions
- Isolate unrelated work
- Maintain continuity within a session
- Cross-session learning when appropriate

### 4. **Topic-Based Organization**
- Group related analyses by topic
- Learn from past work on similar topics
- Keep different topics isolated
- Build institutional knowledge over time

### 5. **Context Summarization**
- Automatic summarization of long contexts
- Agent-specific summaries
- Extraction of key insights
- Code pattern identification

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Agent System                        │
│            (Orchestrator + Specialists)              │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│             Context Manager                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   Chunker    │  │ Summarizer   │  │   Store   │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│              ChromaDB Vector Store                   │
│  ┌─────────────────────────────────────────────┐   │
│  │  Chunks with Embeddings + Metadata          │   │
│  │  • session_id    • topic_id                 │   │
│  │  • context_type  • agent_name               │   │
│  │  • timestamp     • custom metadata          │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. ContextStore

Manages ChromaDB storage and retrieval.

**Key Methods:**
```python
# Add single chunk
store.add_chunk(chunk: ContextChunk)

# Add multiple chunks efficiently
store.add_chunks(chunks: List[ContextChunk])

# Query with semantic search
results = store.query(query: RetrievalQuery)

# Get all chunks from a session
chunks = store.get_session_chunks(session_id: str)

# Get historical context for a topic
history = store.get_topic_history(topic_id: str)
```

**Example:**
```python
from context_management_system import ContextStore, ContextChunk, ContextType

store = ContextStore(persist_directory="./chroma_db")

chunk = ContextChunk(
    content="Fitted OLS model with R²=0.75",
    context_type=ContextType.MODEL_RESULT,
    session_id="session_123",
    topic_id="sales_analysis",
    agent_name="modeling_agent",
    metadata={"r_squared": 0.75}
)

store.add_chunk(chunk)
```

### 2. ContextChunker

Intelligently chunks different types of content.

**Key Methods:**
```python
# Chunk text
chunks = chunker.chunk_text(
    text: str,
    context_type: ContextType,
    session_id: str,
    topic_id: Optional[str] = None
)

# Chunk code by logical sections
chunks = chunker.chunk_code(
    code: str,
    language: str,
    session_id: str
)

# Chunk agent output intelligently
chunks = chunker.chunk_agent_output(
    output_dict: Dict[str, Any],
    agent_name: str,
    session_id: str
)
```

**Example:**
```python
from context_management_system import ContextChunker

chunker = ContextChunker(chunk_size=1000, chunk_overlap=200)

code = """
def analyze_data(df):
    model = OLS(df['y'], df[['x1', 'x2']])
    return model.fit()
"""

chunks = chunker.chunk_code(
    code,
    language="python",
    session_id="session_123",
    topic_id="regression_analysis"
)
```

### 3. ContextSummarizer

Creates intelligent summaries for agents.

**Key Methods:**
```python
# Summarize entire session
summary = await summarizer.summarize_session(
    chunks: List[Dict[str, Any]],
    focus: Optional[str] = None
)

# Create agent-specific summary
context_summary = await summarizer.summarize_for_agent(
    agent_name: str,
    current_task: str,
    session_chunks: List[Dict[str, Any]],
    relevant_history: List[Dict[str, Any]]
)
```

**Example:**
```python
from context_management_system import ContextSummarizer

summarizer = ContextSummarizer(model="openai:gpt-4")

summary = await summarizer.summarize_for_agent(
    agent_name="modeling_agent",
    current_task="Build regression model",
    session_chunks=current_session_chunks,
    relevant_history=historical_chunks
)

print(f"Session summary: {summary.session_summary}")
print(f"Code patterns: {summary.code_patterns}")
print(f"Key findings: {summary.key_findings}")
```

### 4. ContextManager

High-level interface coordinating all components.

**Key Methods:**
```python
# Start new session
session_id = context_mgr.start_session(topic="sales_analysis")

# Store different types of content
context_mgr.store_user_query(query: str)
context_mgr.store_agent_thought(agent_name, thought, metadata)
context_mgr.store_agent_output(agent_name, output, metadata)
context_mgr.store_code(code, language, agent_name, purpose)

# Retrieve context
context = await context_mgr.get_context_for_agent(
    agent_name: str,
    current_task: str,
    include_history: bool = True
)

# Get relevant context via search
results = await context_mgr.get_relevant_context(
    query_text: str,
    context_types: Optional[List[ContextType]] = None
)
```

**Complete Example:**
```python
from context_management_system import ContextManager

# Initialize
context_mgr = ContextManager(
    persist_directory="./chroma_db",
    chunk_size=1000
)

# Start session
session_id = context_mgr.start_session(topic="marketing_analysis")

# Store user query
context_mgr.store_user_query(
    "Analyze the effect of marketing spend on sales"
)

# Agent performs work
context_mgr.store_agent_thought(
    "statistician",
    "Recommending OLS regression with heteroscedasticity tests"
)

# Store results
context_mgr.store_agent_output(
    "modeling_agent",
    {
        "result": "Model fitted successfully",
        "r_squared": 0.75,
        "code": "model = OLS(y, X).fit()"
    }
)

# Later, get context for next agent
context = await context_mgr.get_context_for_agent(
    "interpreter_agent",
    "Interpret regression results",
    include_history=True
)

print(f"Context includes {context.total_chunks} relevant chunks")
```

---

## Context Types

The system tracks different types of context:

```python
class ContextType(str, Enum):
    AGENT_THOUGHT = "agent_thought"      # Agent reasoning process
    AGENT_OUTPUT = "agent_output"        # Agent results
    CODE_CHUNK = "code_chunk"            # Generated code
    DATA_SUMMARY = "data_summary"        # Data descriptions
    MODEL_RESULT = "model_result"        # Model outputs
    INTERPRETATION = "interpretation"     # Interpretations
    USER_QUERY = "user_query"            # User requests
    PLAN = "plan"                        # Analysis plans
    ERROR = "error"                      # Errors encountered
    CONVERSATION = "conversation"        # Conversational exchanges
```

---

## Metadata Schema

Each chunk has metadata for filtering and organization:

```python
{
    "session_id": str,          # Required: Session identifier
    "topic_id": str,            # Optional: Topic identifier
    "context_type": str,        # Required: Type of context
    "agent_name": str,          # Optional: Agent that created it
    "timestamp": str,           # Required: ISO format timestamp
    # Custom metadata fields
    "stage": str,               # Analysis stage
    "task": str,                # Task description
    "language": str,            # For code chunks
    "purpose": str,             # Purpose of code
    # ... any other relevant fields
}
```

---

## Usage Patterns

### Pattern 1: Single Session Analysis

```python
# Initialize
context_mgr = ContextManager()
session_id = context_mgr.start_session(topic="customer_churn")

# Store query
context_mgr.store_user_query("Predict customer churn")

# Agents do work, storing context
for agent in agents:
    context = await context_mgr.get_context_for_agent(
        agent.name,
        task="Perform analysis"
    )
    result = await agent.execute_with_context(task, context)
    context_mgr.store_agent_output(agent.name, result)

# Get final summary
summary = await context_mgr.get_session_summary()
```

### Pattern 2: Cross-Session Learning

```python
# Session 1: Initial analysis
session1 = context_mgr.start_session(topic="sales_forecasting")
# ... perform analysis ...

# Session 2: Related analysis (same topic)
session2 = context_mgr.start_session(topic="sales_forecasting")

# This will automatically retrieve relevant context from Session 1
context = await context_mgr.get_context_for_agent(
    "modeling_agent",
    "Build improved forecasting model",
    include_history=True  # Gets context from Session 1
)

# Agent can now learn from past work
```

### Pattern 3: Topic Isolation

```python
# Marketing analysis
session1 = context_mgr.start_session(topic="marketing_roi")
# ... marketing analysis ...

# Customer analysis (different topic)
session2 = context_mgr.start_session(topic="customer_segmentation")

# This will NOT retrieve marketing context
context = await context_mgr.get_context_for_agent(
    "modeling_agent",
    "Perform customer segmentation",
    include_history=True  # Only gets customer_segmentation history
)
```

### Pattern 4: Code Pattern Reuse

```python
# Agent needs to write code
context = await context_mgr.get_context_for_agent(
    "modeling_agent",
    "Build logistic regression model"
)

# Context includes successful code patterns
for pattern in context.code_patterns:
    print(f"Useful pattern: {pattern}")

# Agent can adapt these patterns for current task
```

### Pattern 5: Semantic Search

```python
# Search for specific information across all sessions
results = await context_mgr.get_relevant_context(
    query_text="How to handle missing data in regression?",
    context_types=[ContextType.CODE_CHUNK, ContextType.AGENT_THOUGHT],
    limit=10
)

for result in results:
    print(f"Found: {result['content'][:100]}")
    print(f"From: {result['metadata']['agent_name']}")
```

---

## Integration with Agents

### Context-Aware Base Agent

```python
from context_management_system import ContextAwareAgent

class MyAgent(ContextAwareAgent):
    def get_system_prompt(self) -> str:
        return """You are an expert agent with access to 
        historical context and successful patterns."""
    
    # Automatically gets context and stores results
    async def execute_with_context(self, task: str):
        # Context automatically retrieved
        # Results automatically stored
        return await super().execute_with_context(task)
```

### Manual Context Integration

```python
class CustomAgent:
    def __init__(self, context_manager):
        self.context_mgr = context_manager
    
    async def execute(self, task: str):
        # Get context
        context = await self.context_mgr.get_context_for_agent(
            self.name,
            task,
            include_history=True
        )
        
        # Build enhanced prompt
        prompt = f"""
        Task: {task}
        
        Context: {context.session_summary}
        Past successes: {context.code_patterns}
        Key findings: {context.key_findings}
        """
        
        # Execute
        result = await self.execute_task(prompt)
        
        # Store result
        self.context_mgr.store_agent_output(
            self.name,
            result
        )
        
        return result
```

---

## Performance Considerations

### Chunk Size

- **Small chunks (500-1000)**: Better semantic search, more chunks
- **Large chunks (2000-3000)**: Fewer chunks, more context per chunk
- **Default: 1000**: Good balance for most use cases

### Overlap

- **No overlap**: Faster storage, risk losing context at boundaries
- **Small overlap (100-200)**: Good for most content
- **Large overlap (500+)**: Better continuity, more storage

### Query Limits

- Start with small limits (5-10 chunks)
- Increase if needed, but be mindful of context window
- Use summarization for large result sets

### Storage Management

```python
# Get statistics
stats = context_mgr.get_stats()
print(f"Total chunks: {stats['total_chunks']}")

# Delete old sessions if needed
context_mgr.store.delete_session(old_session_id)

# Clear all data (careful!)
# context_mgr.store.collection.delete()
```

---

## Best Practices

### 1. **Meaningful Topic IDs**

```python
# Good: Descriptive topics
topic_id = "customer_churn_prediction"
topic_id = "sales_forecasting_Q4_2025"

# Avoid: Too generic
topic_id = "analysis"  # Not specific enough
```

### 2. **Rich Metadata**

```python
# Good: Include useful metadata
context_mgr.store_code(
    code,
    language="python",
    agent_name="modeler",
    purpose="logistic_regression_churn",
    metadata={
        "model_type": "logistic_regression",
        "performance": {"auc": 0.85},
        "data_size": 10000
    }
)
```

### 3. **Regular Summarization**

```python
# After major stages, create summaries
summary = await context_mgr.get_session_summary()
context_mgr.store_agent_output(
    "orchestrator",
    {"stage_summary": summary},
    metadata={"stage": "eda_complete"}
)
```

### 4. **Strategic Context Retrieval**

```python
# Be specific about what you need
context = await context_mgr.get_relevant_context(
    query_text="best practices for handling heteroscedasticity",
    context_types=[
        ContextType.AGENT_THOUGHT,
        ContextType.CODE_CHUNK
    ],
    limit=5
)
```

### 5. **Error Context**

```python
# Store errors for learning
try:
    result = agent.execute()
except Exception as e:
    context_mgr.store_agent_output(
        agent.name,
        {"error": str(e), "context": "during model fitting"},
        metadata={"context_type": "error"}
    )
```

---

## Troubleshooting

### Issue: "Collection already exists"

```python
# Solution: Use existing collection or delete
client = chromadb.Client(...)
try:
    client.delete_collection("agent_context")
except:
    pass
collection = client.create_collection("agent_context")
```

### Issue: "Too much context in prompt"

```python
# Solution: Use summarization
context = await context_mgr.get_context_for_agent(...)
# Context automatically summarized for agents
```

### Issue: "Irrelevant context retrieved"

```python
# Solution: Be more specific in queries
results = await context_mgr.get_relevant_context(
    query_text="OLS regression with robust standard errors",  # Specific
    context_types=[ContextType.CODE_CHUNK],  # Filtered
    limit=3  # Limited
)
```

### Issue: "Slow retrieval"

```python
# Solution: Reduce query scope
query = RetrievalQuery(
    query_text=text,
    session_id=current_session,  # Limit to current session
    include_cross_session=False,  # Don't search history
    limit=5  # Smaller limit
)
```

---

## Example: Complete Workflow

```python
import asyncio
from context_management_system import ContextManager, ContextType

async def complete_analysis_workflow():
    # Initialize
    context_mgr = ContextManager(persist_directory="./my_analysis_db")
    
    # Session 1: Initial exploration
    print("Session 1: Exploratory Analysis")
    session1 = context_mgr.start_session(topic="house_price_prediction")
    
    context_mgr.store_user_query(
        "Predict house prices from size, location, and age"
    )
    
    # Statistician work
    context_mgr.store_agent_thought(
        "statistician",
        "This is a regression problem. Will recommend multiple linear regression with potential interactions."
    )
    
    # Get context for modeler
    context = await context_mgr.get_context_for_agent(
        "modeling_agent",
        "Build regression model for house prices"
    )
    
    # Store model results
    context_mgr.store_agent_output(
        "modeling_agent",
        {
            "result": "Model fitted: R²=0.82",
            "code": "model = OLS(y, X).fit()",
            "coefficients": {"size": 150, "age": -5}
        }
    )
    
    context_mgr.store_code(
        "model = sm.OLS(y, sm.add_constant(X)).fit()",
        language="python",
        agent_name="modeling_agent",
        purpose="initial_model"
    )
    
    # Session 2: Improved model (same topic)
    print("\nSession 2: Improved Model")
    session2 = context_mgr.start_session(topic="house_price_prediction")
    
    context_mgr.store_user_query(
        "Improve model by adding neighborhood characteristics"
    )
    
    # Get context - will include Session 1 learnings
    context = await context_mgr.get_context_for_agent(
        "modeling_agent",
        "Extend model with neighborhood features",
        include_history=True
    )
    
    print(f"\nContext retrieved:")
    print(f"  - Total chunks: {context.total_chunks}")
    print(f"  - Code patterns: {len(context.code_patterns)}")
    print(f"  - Past sessions: {len(context.relevant_past_sessions)}")
    
    # Model can now build on past work
    context_mgr.store_agent_output(
        "modeling_agent",
        {
            "result": "Extended model: R²=0.89 (improved from 0.82)",
            "improvements": "Added neighborhood dummies"
        }
    )
    
    # Get final summary
    summary = await context_mgr.get_session_summary()
    print(f"\nFinal summary:\n{summary[:300]}...")
    
    # Statistics
    stats = context_mgr.get_stats()
    print(f"\nStorage stats:")
    print(f"  Total chunks: {stats['total_chunks']}")

asyncio.run(complete_analysis_workflow())
```

---

## Advanced Features

### Custom Chunking Strategies

```python
class CustomChunker(ContextChunker):
    def chunk_by_function(self, code: str):
        # Custom chunking logic
        pass
```

### Custom Summarization

```python
class DomainSummarizer(ContextSummarizer):
    async def summarize_for_finance(self, chunks):
        # Domain-specific summarization
        pass
```

### Multi-Modal Context

```python
# Store visualization context
context_mgr.store_agent_output(
    "eda_agent",
    {
        "plot_type": "scatter",
        "insights": "Strong linear relationship",
        "image_path": "/plots/scatter.png"
    },
    metadata={"visual": True}
)
```

---

## Future Enhancements

- [ ] Automatic topic detection
- [ ] Context pruning strategies
- [ ] Multi-user support
- [ ] Context versioning
- [ ] Export/import functionality
- [ ] Advanced visualization of context relationships
- [ ] Integration with external knowledge bases

---

## References

- ChromaDB Documentation: https://docs.trychroma.com/
- PydanticAI: https://ai.pydantic.dev/
- Vector Embeddings: https://www.pinecone.io/learn/vector-embeddings/

---

**Version**: 1.0  
**Last Updated**: October 2025