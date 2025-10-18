"""
Context Management and RAG System for Data Science Agents
Handles context summarization, chunking, and retrieval using ChromaDB
"""

import asyncio
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai import Agent
import chromadb
from chromadb.config import Settings

from datascienceagent.model_utils import process_model


# ============================================================================
# DATA MODELS
# ============================================================================

class ContextType(str, Enum):
    """Types of context that can be stored"""
    AGENT_THOUGHT = "agent_thought"
    AGENT_OUTPUT = "agent_output"
    CODE_CHUNK = "code_chunk"
    DATA_SUMMARY = "data_summary"
    MODEL_RESULT = "model_result"
    INTERPRETATION = "interpretation"
    USER_QUERY = "user_query"
    PLAN = "plan"
    ERROR = "error"
    CONVERSATION = "conversation"


class ContextChunk(BaseModel):
    """A chunk of context to be stored"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    context_type: ContextType
    session_id: str
    topic_id: Optional[str] = None
    agent_name: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class ContextSummary(BaseModel):
    """Summary of context for an agent"""
    session_summary: str
    relevant_past_sessions: List[Dict[str, str]] = Field(default_factory=list)
    topic_insights: List[str] = Field(default_factory=list)
    code_patterns: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    total_chunks: int
    summary_timestamp: datetime = Field(default_factory=datetime.now)


class RetrievalQuery(BaseModel):
    """Query for retrieving relevant context"""
    query_text: str
    session_id: Optional[str] = None
    topic_id: Optional[str] = None
    context_types: List[ContextType] = Field(default_factory=list)
    agent_name: Optional[str] = None
    limit: int = 10
    include_cross_session: bool = False


# ============================================================================
# CHROMADB CONTEXT STORE
# ============================================================================

class ContextStore:
    """
    Manages context storage and retrieval using ChromaDB
    Provides vector search capabilities for relevant context
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "agent_context"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Agent context and memory"}
        )
        
        print(f"✅ Context store initialized: {persist_directory}")
    
    def add_chunk(self, chunk: ContextChunk):
        """Add a context chunk to the store"""
        
        metadata = {
            "session_id": chunk.session_id,
            "context_type": chunk.context_type.value,
            "timestamp": chunk.timestamp.isoformat(),
            **chunk.metadata
        }
        
        # Add optional fields if present
        if chunk.topic_id:
            metadata["topic_id"] = chunk.topic_id
        if chunk.agent_name:
            metadata["agent_name"] = chunk.agent_name
        
        # Add to ChromaDB
        self.collection.add(
            documents=[chunk.content],
            metadatas=[metadata],
            ids=[chunk.id]
        )
    
    def add_chunks(self, chunks: List[ContextChunk]):
        """Add multiple chunks efficiently"""
        
        if not chunks:
            return
        
        documents = [c.content for c in chunks]
        metadatas = []
        ids = [c.id for c in chunks]
        
        for chunk in chunks:
            metadata = {
                "session_id": chunk.session_id,
                "context_type": chunk.context_type.value,
                "timestamp": chunk.timestamp.isoformat(),
                **chunk.metadata
            }
            if chunk.topic_id:
                metadata["topic_id"] = chunk.topic_id
            if chunk.agent_name:
                metadata["agent_name"] = chunk.agent_name
            
            metadatas.append(metadata)
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(
        self,
        query: RetrievalQuery
    ) -> List[Dict[str, Any]]:
        """Query for relevant context"""
        
        # Build where filter
        where_filter = {}
        
        if query.session_id and not query.include_cross_session:
            where_filter["session_id"] = query.session_id
        
        if query.topic_id:
            where_filter["topic_id"] = query.topic_id
        
        if query.context_types:
            where_filter["context_type"] = {
                "$in": [ct.value for ct in query.context_types]
            }
        
        if query.agent_name:
            where_filter["agent_name"] = query.agent_name
        
        # Perform vector search
        results = self.collection.query(
            query_texts=[query.query_text],
            n_results=query.limit,
            where=where_filter if where_filter else None
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def get_session_chunks(
        self,
        session_id: str,
        context_types: Optional[List[ContextType]] = None
    ) -> List[Dict[str, Any]]:
        """Get all chunks from a session"""
        
        where_filter = {"session_id": session_id}
        
        if context_types:
            where_filter["context_type"] = {
                "$in": [ct.value for ct in context_types]
            }
        
        results = self.collection.get(
            where=where_filter,
            limit=1000  # Adjust as needed
        )
        
        formatted_results = []
        for i, doc in enumerate(results['documents']):
            formatted_results.append({
                "content": doc,
                "metadata": results['metadatas'][i],
                "id": results['ids'][i]
            })
        
        return formatted_results
    
    def get_topic_history(
        self,
        topic_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get historical context for a topic across sessions"""
        
        results = self.collection.get(
            where={"topic_id": topic_id},
            limit=limit
        )
        
        formatted_results = []
        for i, doc in enumerate(results['documents']):
            formatted_results.append({
                "content": doc,
                "metadata": results['metadatas'][i],
                "id": results['ids'][i]
            })
        
        return formatted_results
    
    def delete_session(self, session_id: str):
        """Delete all chunks from a session"""
        
        self.collection.delete(
            where={"session_id": session_id}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        total_count = self.collection.count()
        
        return {
            "total_chunks": total_count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }


# ============================================================================
# CONTEXT CHUNKER
# ============================================================================

class ContextChunker:
    """
    Intelligently chunks context for storage
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(
        self,
        text: str,
        context_type: ContextType,
        session_id: str,
        topic_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContextChunk]:
        """Chunk text into manageable pieces"""
        
        if len(text) <= self.chunk_size:
            return [ContextChunk(
                content=text,
                context_type=context_type,
                session_id=session_id,
                topic_id=topic_id,
                agent_name=agent_name,
                metadata=metadata or {}
            )]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for punct in ['. ', '.\n', '! ', '?\n']:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct > self.chunk_size // 2:
                        end = start + last_punct + len(punct)
                        break
            
            chunk_text = text[start:end]
            
            chunk = ContextChunk(
                content=chunk_text,
                context_type=context_type,
                session_id=session_id,
                topic_id=topic_id,
                agent_name=agent_name,
                metadata={
                    **(metadata or {}),
                    "chunk_index": len(chunks),
                    "is_continuation": start > 0
                }
            )
            
            chunks.append(chunk)
            
            # Move start forward, with overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        return chunks
    
    def chunk_code(
        self,
        code: str,
        language: str,
        session_id: str,
        topic_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ContextChunk]:
        """Chunk code by logical sections"""
        
        # Split by function/class definitions
        chunks = []
        current_chunk = []
        current_size = 0
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line_with_newline = line + '\n'
            current_chunk.append(line_with_newline)
            current_size += len(line_with_newline)
            
            # Check if we should create a chunk
            should_chunk = (
                current_size > self.chunk_size or
                (line.strip().startswith('def ') and current_size > 200) or
                (line.strip().startswith('class ') and current_size > 200) or
                i == len(lines) - 1
            )
            
            if should_chunk and current_chunk:
                chunk = ContextChunk(
                    content=''.join(current_chunk),
                    context_type=ContextType.CODE_CHUNK,
                    session_id=session_id,
                    topic_id=topic_id,
                    agent_name=agent_name,
                    metadata={
                        **(metadata or {}),
                        "language": language,
                        "chunk_index": len(chunks),
                        "line_start": i - len(current_chunk) + 1,
                        "line_end": i
                    }
                )
                chunks.append(chunk)
                
                current_chunk = []
                current_size = 0
        
        return chunks if chunks else [ContextChunk(
            content=code,
            context_type=ContextType.CODE_CHUNK,
            session_id=session_id,
            topic_id=topic_id,
            agent_name=agent_name,
            metadata={**(metadata or {}), "language": language}
        )]
    
    def chunk_agent_output(
        self,
        output_dict: Dict[str, Any],
        agent_name: str,
        session_id: str,
        topic_id: Optional[str] = None
    ) -> List[ContextChunk]:
        """Chunk agent output intelligently"""
        
        chunks = []
        
        # Chunk main output
        if "result" in output_dict:
            result_str = str(output_dict["result"])
            chunks.extend(self.chunk_text(
                result_str,
                ContextType.AGENT_OUTPUT,
                session_id,
                topic_id,
                agent_name,
                {"output_type": "result"}
            ))
        
        # Chunk code separately if present
        if "code" in output_dict:
            code = output_dict["code"]
            if isinstance(code, dict):
                for key, code_str in code.items():
                    chunks.extend(self.chunk_code(
                        code_str,
                        "python",
                        session_id,
                        topic_id,
                        agent_name,
                        {"code_type": key}
                    ))
            elif isinstance(code, str):
                chunks.extend(self.chunk_code(
                    code,
                    "python",
                    session_id,
                    topic_id,
                    agent_name
                ))
        
        return chunks


# ============================================================================
# CONTEXT SUMMARIZER
# ============================================================================

class ContextSummarizer:
    """
    Creates intelligent summaries of context for agents
    """
    
    def __init__(self, model: str = "openai:gpt-4"):
        self.model = process_model(model)
    
        self.agent = Agent(
            self.model,
            system_prompt="""You are an expert at summarizing complex analysis context.

Your role is to:
1. Extract key insights and findings
2. Identify important patterns and relationships
3. Summarize technical details concisely
4. Highlight relevant prior work
5. Note important assumptions and limitations

Create summaries that preserve critical information while reducing verbosity.
Focus on what's actionable and relevant for continuing the analysis."""
        )
    
    async def summarize_session(
        self,
        chunks: List[Dict[str, Any]],
        focus: Optional[str] = None
    ) -> str:
        """Summarize an entire session's context"""
        
        # Group chunks by type
        grouped = {}
        for chunk in chunks:
            ctx_type = chunk['metadata'].get('context_type', 'unknown')
            if ctx_type not in grouped:
                grouped[ctx_type] = []
            grouped[ctx_type].append(chunk['content'])
        
        # Build summary prompt
        prompt = "Summarize this analysis session:\n\n"
        
        for ctx_type, contents in grouped.items():
            prompt += f"\n{ctx_type.upper()}:\n"
            # Limit content per type
            combined = "\n".join(contents[:5])
            if len(combined) > 2000:
                combined = combined[:2000] + "..."
            prompt += combined + "\n"
        
        if focus:
            prompt += f"\nFocus particularly on: {focus}"
        
        prompt += "\n\nProvide a concise summary highlighting key findings, methods used, and important insights."
        
        result = await self.agent.run(prompt)
        return str(result.output)
    
    async def summarize_for_agent(
        self,
        agent_name: str,
        current_task: str,
        session_chunks: List[Dict[str, Any]],
        relevant_history: List[Dict[str, Any]]
    ) -> ContextSummary:
        """Create a targeted summary for a specific agent"""
        
        prompt = f"""Create a summary for the {agent_name} agent.

Current Task: {current_task}

SESSION CONTEXT:
{self._format_chunks(session_chunks[:10])}

RELEVANT HISTORY:
{self._format_chunks(relevant_history[:5])}

Provide:
1. Key context from current session
2. Relevant insights from past sessions
3. Code patterns that worked well
4. Important findings to consider
5. Potential pitfalls to avoid

Be concise but comprehensive."""

        result = await self.agent.run(prompt)
        summary_text = str(result.output)
        
        # Extract structured information
        return ContextSummary(
            session_summary=summary_text,
            relevant_past_sessions=self._extract_past_sessions(relevant_history),
            topic_insights=self._extract_insights(session_chunks),
            code_patterns=self._extract_code_patterns(session_chunks),
            key_findings=self._extract_findings(session_chunks),
            total_chunks=len(session_chunks) + len(relevant_history)
        )
    
    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks for prompt"""
        formatted = []
        for chunk in chunks:
            ctx_type = chunk['metadata'].get('context_type', 'unknown')
            content = chunk['content']
            if len(content) > 500:
                content = content[:500] + "..."
            formatted.append(f"[{ctx_type}] {content}")
        return "\n\n".join(formatted)
    
    def _extract_past_sessions(
        self,
        history: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Extract past session references"""
        sessions = {}
        for chunk in history:
            session_id = chunk['metadata'].get('session_id')
            if session_id and session_id not in sessions:
                sessions[session_id] = {
                    "session_id": session_id,
                    "summary": chunk['content'][:200]
                }
        return list(sessions.values())[:3]
    
    def _extract_insights(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights"""
        insights = []
        for chunk in chunks:
            if chunk['metadata'].get('context_type') == 'interpretation':
                content = chunk['content']
                if len(content) < 200:
                    insights.append(content)
        return insights[:5]
    
    def _extract_code_patterns(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract useful code patterns"""
        patterns = []
        for chunk in chunks:
            if chunk['metadata'].get('context_type') == 'code_chunk':
                code = chunk['content']
                if len(code) < 300 and ('def ' in code or 'class ' in code):
                    patterns.append(code.split('\n')[0])  # Function/class signature
        return patterns[:5]
    
    def _extract_findings(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract key findings"""
        findings = []
        for chunk in chunks:
            metadata = chunk['metadata']
            if metadata.get('context_type') in ['model_result', 'agent_output']:
                # Look for statistical findings
                content = chunk['content'].lower()
                if any(term in content for term in ['significant', 'correlation', 'effect']):
                    findings.append(chunk['content'][:200])
        return findings[:5]


# ============================================================================
# CONTEXT MANAGER
# ============================================================================

class ContextManager:
    """
    High-level context management for the agent system
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        model: str = "openai:gpt-4.1-mini"
    ):
        self.store = ContextStore(persist_directory)
        self.chunker = ContextChunker(chunk_size)
        self.summarizer = ContextSummarizer(model)
        
        # Current session tracking
        self.current_session_id: Optional[str] = None
        self.current_topic_id: Optional[str] = None
    
    def start_session(self, topic: Optional[str] = None) -> str:
        """Start a new analysis session"""
        self.current_session_id = str(uuid.uuid4())
        
        if topic:
            # Generate topic_id from topic name
            topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8]
            self.current_topic_id = f"topic_{topic_hash}"
        else:
            self.current_topic_id = None
        
        print(f"📝 Started session: {self.current_session_id}")
        if self.current_topic_id:
            print(f"   Topic: {self.current_topic_id}")
        
        return self.current_session_id
    
    def store_user_query(self, query: str):
        """Store user's initial query"""
        if not self.current_session_id:
            self.start_session()
        
        chunks = self.chunker.chunk_text(
            query,
            ContextType.USER_QUERY,
            self.current_session_id,
            self.current_topic_id,
            metadata={"is_initial_query": True}
        )
        
        self.store.add_chunks(chunks)
    
    def store_agent_thought(
        self,
        agent_name: str,
        thought: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store an agent's reasoning process"""
        if not self.current_session_id:
            return
        
        chunks = self.chunker.chunk_text(
            thought,
            ContextType.AGENT_THOUGHT,
            self.current_session_id,
            self.current_topic_id,
            agent_name,
            metadata
        )
        
        self.store.add_chunks(chunks)
    
    def store_agent_output(
        self,
        agent_name: str,
        output: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store agent output with intelligent chunking"""
        if not self.current_session_id:
            return
        
        chunks = self.chunker.chunk_agent_output(
            output,
            agent_name,
            self.current_session_id,
            self.current_topic_id
        )
        
        # Add additional metadata
        for chunk in chunks:
            chunk.metadata.update(metadata or {})
        
        self.store.add_chunks(chunks)
    
    def store_code(
        self,
        code: str,
        language: str,
        agent_name: str,
        purpose: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store generated code"""
        if not self.current_session_id:
            return
        
        chunks = self.chunker.chunk_code(
            code,
            language,
            self.current_session_id,
            self.current_topic_id,
            agent_name,
            {**(metadata or {}), "purpose": purpose}
        )
        
        self.store.add_chunks(chunks)
    
    async def get_context_for_agent(
        self,
        agent_name: str,
        current_task: str,
        include_history: bool = True
    ) -> ContextSummary:
        """Get relevant context summary for an agent"""
        
        if not self.current_session_id:
            return ContextSummary(
                session_summary="No active session",
                total_chunks=0
            )
        
        # Get current session context
        session_chunks = self.store.get_session_chunks(self.current_session_id)
        
        # Get relevant historical context
        history_chunks = []
        if include_history and self.current_topic_id:
            query = RetrievalQuery(
                query_text=current_task,
                topic_id=self.current_topic_id,
                include_cross_session=True,
                limit=10
            )
            history_chunks = self.store.query(query)
        
        # Create summary
        summary = await self.summarizer.summarize_for_agent(
            agent_name,
            current_task,
            session_chunks,
            history_chunks
        )
        
        return summary
    
    async def get_relevant_context(
        self,
        query_text: str,
        context_types: Optional[List[ContextType]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get relevant context via semantic search"""
        
        query = RetrievalQuery(
            query_text=query_text,
            session_id=self.current_session_id,
            context_types=context_types or [],
            include_cross_session=True,
            limit=limit
        )
        
        return self.store.query(query)
    
    async def get_session_summary(self) -> str:
        """Get a summary of the current session"""
        
        if not self.current_session_id:
            return "No active session"
        
        chunks = self.store.get_session_chunks(self.current_session_id)
        
        if not chunks:
            return "No context available for current session"
        
        summary = await self.summarizer.summarize_session(chunks)
        return summary
    
    def get_topic_history_summary(self, topic_id: str) -> List[Dict[str, Any]]:
        """Get historical context for a topic"""
        return self.store.get_topic_history(topic_id, limit=20)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        base_stats = self.store.get_stats()
        base_stats.update({
            "current_session_id": self.current_session_id,
            "current_topic_id": self.current_topic_id
        })
        return base_stats


# ============================================================================
# INTEGRATION WITH AGENTS
# ============================================================================

class ContextAwareAgent:
    """
    Base class for agents with context awareness
    Automatically stores thoughts and retrieves relevant context
    """
    
    def __init__(
        self,
        name: str,
        context_manager: ContextManager,
        model: str = "openai:gpt-4.1-mini"
    ):
        self.name = name
        self.context_manager = context_manager
        self.model = process_model(model)
        self.agent = Agent(self.model, system_prompt=self.get_system_prompt())
    
    def get_system_prompt(self) -> str:
        """Override in subclasses"""
        return f"You are {self.name}, a specialized AI agent."
    
    async def execute_with_context(
        self,
        task: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute task with context retrieval"""
        
        # Get relevant context
        context_summary = await self.context_manager.get_context_for_agent(
            self.name,
            task,
            include_history=True
        )
        
        # Build enhanced prompt with context
        enhanced_prompt = self._build_context_prompt(task, context_summary)
        
        # Store the thought process
        self.context_manager.store_agent_thought(
            self.name,
            f"Processing task: {task}\nContext summary: {context_summary.session_summary[:200]}",
            metadata={"task": task}
        )
        
        # Execute task
        result = await self.agent.run(enhanced_prompt)
        output = {"result": str(result.output)}
        
        # Store output
        self.context_manager.store_agent_output(
            self.name,
            output,
            metadata={"task": task}
        )
        
        return output
    
    def _build_context_prompt(
        self,
        task: str,
        context: ContextSummary
    ) -> str:
        """Build prompt with context"""
        
        prompt = f"""Task: {task}

CURRENT SESSION CONTEXT:
{context.session_summary}

"""
        
        if context.relevant_past_sessions:
            prompt += "\nRELEVANT PAST WORK:\n"
            for session in context.relevant_past_sessions[:2]:
                prompt += f"- {session['summary']}\n"
        
        if context.code_patterns:
            prompt += "\nUSEFUL CODE PATTERNS:\n"
            for pattern in context.code_patterns[:3]:
                prompt += f"- {pattern}\n"
        
        if context.key_findings:
            prompt += "\nKEY FINDINGS TO CONSIDER:\n"
            for finding in context.key_findings[:3]:
                prompt += f"- {finding[:150]}...\n"
        
        prompt += "\nNow proceed with the task, leveraging this context appropriately."
        
        return prompt


# ============================================================================
# DEMO
# ============================================================================

async def demo_context_system():
    """Demonstrate the context management system"""
    def print(*msg):
        from rich.console import Console
        console = Console()
        from rich.markdown import Markdown
        try:
            md = Markdown(" ".join(msg))
            console.print(md)
        except Exception:
            console.print(*msg)
    
    print("="*70)
    print("CONTEXT MANAGEMENT & RAG SYSTEM DEMO")
    print("="*70)
    
    # Initialize context manager
    context_mgr = ContextManager(
        persist_directory="./demo_chroma_db",
        chunk_size=500,
        model="ollama:llama3.2"
    )
    
    # Session 1: First analysis
    print("\n📊 SESSION 1: Initial Regression Analysis")
    print("-"*70)
    
    session1_id = context_mgr.start_session(topic="sales_regression_analysis")
    
    # Store user query
    context_mgr.store_user_query(
        "Analyze the relationship between marketing spend and sales"
    )
    
    # Simulate agent work
    context_mgr.store_agent_thought(
        "statistician",
        "This is a simple regression problem. Will recommend OLS regression with diagnostics for heteroscedasticity and normality."
    )
    
    context_mgr.store_agent_output(
        "modeling_agent",
        {
            "result": "Fitted OLS model. R²=0.75, significant positive relationship (p<0.001)",
            "code": """
import statsmodels.api as sm

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
"""
        }
    )
    
    # Get session summary
    summary1 = await context_mgr.get_session_summary()
    print(f"\nSession 1 Summary:\n{summary1}...")
    
    # Session 2: Related analysis (same topic)
    print("\n\n📊 SESSION 2: Extended Analysis with Context")
    print("-"*70)
    
    session2_id = context_mgr.start_session(topic="sales_regression_analysis")
    
    context_mgr.store_user_query(
        "Now add competitor activity and seasonality to the model"
    )
    
    # Get context for agent
    context = await context_mgr.get_context_for_agent(
        "modeling_agent",
        "Add competitor activity and seasonality to regression model",
        include_history=True
    )
    
    print(f"\n📝 Context Summary for Modeling Agent:")
    print(f"   Total chunks: {context.total_chunks}")
    print(f"   Session summary: {context.session_summary}...")
    
    if context.relevant_past_sessions:
        print(f"   Past sessions referenced: {len(context.relevant_past_sessions)}")
    
    if context.code_patterns:
        print(f"   Code patterns found: {len(context.code_patterns)}")
        print(f"   Example: {context.code_patterns[0]}...")
    
    # Store new agent work
    context_mgr.store_code(
        """
import statsmodels.api as sm
import pandas as pd

# Add competitor and seasonality
X_extended = sm.add_constant(df[['marketing_spend', 'competitor_activity', 'season_Q1', 'season_Q2', 'season_Q3']])
model = sm.OLS(df['sales'], X_extended)
results = model.fit()
""",
        language="python",
        agent_name="modeling_agent",
        purpose="Extended regression with competitor and seasonality"
    )
    
    # Session 3: Different topic
    print("\n\n📊 SESSION 3: Different Topic (No Cross-Context)")
    print("-"*70)
    
    session3_id = context_mgr.start_session(topic="customer_churn_analysis")
    
    context_mgr.store_user_query(
        "Predict customer churn using logistic regression"
    )
    
    context3 = await context_mgr.get_context_for_agent(
        "modeling_agent",
        "Build logistic regression for churn prediction",
        include_history=True
    )
    
    print(f"\n📝 Context for Different Topic:")
    print(f"   Total chunks: {context3.total_chunks}")
    print(f"   Should not include sales regression context")
    
    # Semantic search across all sessions
    print("\n\n🔍 SEMANTIC SEARCH: Finding regression examples")
    print("-"*70)
    
    regression_examples = await context_mgr.get_relevant_context(
        "regression model code examples",
        context_types=[ContextType.CODE_CHUNK],
        limit=5
    )
    
    print(f"Found {len(regression_examples)} relevant code examples:")
    for i, example in enumerate(regression_examples[:2], 1):
        print(f"\n{i}. From session: {example['metadata']['session_id'][:8]}...")
        print(f"   {example['content']}...")
    
    # Get statistics
    print("\n\n📈 STORAGE STATISTICS")
    print("-"*70)
    
    stats = context_mgr.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Context management demo complete!")


# ============================================================================
# EXAMPLE INTEGRATION
# ============================================================================

async def example_context_aware_modeling():
    """Example of using context-aware agent"""

    def print(*msg):
        from rich.console import Console
        console = Console()
        from rich.markdown import Markdown
        try:
            md = Markdown(" ".join(msg))
            console.print(md)
        except Exception:
            console.print(*msg)
    
    print("\n" + "="*70)
    print("CONTEXT-AWARE AGENT EXAMPLE")
    print("="*70)
    
    # Initialize
    context_mgr = ContextManager(model="ollama:llama3.2")
    context_mgr.start_session(topic="time_series_forecasting")
    
    # Create context-aware agent
    class ContextAwareModeler(ContextAwareAgent):
        def get_system_prompt(self):
            return """You are a statistical modeling expert.
            You build models, considering past successful approaches."""
    
    modeler = ContextAwareModeler("modeler", context_mgr, model="ollama:llama3.2")
    
    # Execute with automatic context
    result = await modeler.execute_with_context(
        "Build an ARIMA model for sales forecasting"
    )
    
    print(f"\nResult: {result['result']}...")
    print("\n✅ Context automatically retrieved and stored!")


if __name__ == "__main__":
    # asyncio.run(demo_context_system())
    asyncio.run(example_context_aware_modeling())