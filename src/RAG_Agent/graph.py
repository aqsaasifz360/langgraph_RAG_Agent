import os
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import SystemMessage
import numpy as np

load_dotenv()

required_vars = [
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION"
]

for var in required_vars:
    if not os.environ.get(var):
        raise ValueError(f"Required environment variable {var} not found in .env file")

print(f"Using Google Cloud Project: {os.environ['GOOGLE_CLOUD_PROJECT']}")
print(f"Using Location: {os.environ['GOOGLE_CLOUD_LOCATION']}")
print(f"Credentials file: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")

# Set up local FAISS directory
FAISS_INDEX_PATH = "./faiss_index"

def load_or_create_vectorstore():
    """Load existing FAISS index or create a new one with optimized chunking."""
    
    # Initialize embeddings
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-004",
    )
    
    # Check if FAISS index exists locally
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        try:
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("Successfully loaded existing FAISS index")
            return vectorstore, embeddings
        except Exception as e:
            print(f"Error loading existing index: {e}")
            print("Creating new FAISS index...")
    else:
        print("No existing FAISS index found. Creating new one...")
    
    # Create new FAISS index with better chunking strategy
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]
    
    print("Loading documents from URLs...")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    print("Splitting documents with optimized strategy...")
    
    # Use multiple chunking strategies for better coverage
    doc_splits = []
    
    # Strategy 1: Larger chunks for context (512 tokens)
    text_splitter_large = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    large_splits = text_splitter_large.split_documents(docs_list)
    doc_splits.extend(large_splits)
    
    # Strategy 2: Medium chunks for balanced retrieval (256 tokens)
    text_splitter_medium = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    medium_splits = text_splitter_medium.split_documents(docs_list)
    doc_splits.extend(medium_splits)
    
    # Strategy 3: Smaller chunks for precise retrieval (128 tokens)
    text_splitter_small = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=128,
        chunk_overlap=32,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    small_splits = text_splitter_small.split_documents(docs_list)
    doc_splits.extend(small_splits)
    
    print(f"Creating FAISS index with {len(doc_splits)} document chunks...")
    vectorstore = FAISS.from_documents(
        documents=doc_splits,
        embedding=embeddings
    )
    
    # Save the FAISS index locally
    try:
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"Warning: Could not save FAISS index: {e}")
    
    return vectorstore, embeddings

def add_documents_to_vectorstore(vectorstore, new_urls):
    """Add new documents to existing FAISS vectorstore with optimized chunking."""
    print(f"Adding {len(new_urls)} new URLs to vectorstore...")
    
    # Load new documents
    new_docs = [WebBaseLoader(url).load() for url in new_urls]
    new_docs_list = [item for sublist in new_docs for item in sublist]
    
    # Apply the same multi-strategy chunking
    new_doc_splits = []
    
    # Large chunks
    text_splitter_large = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=100
    )
    large_splits = text_splitter_large.split_documents(new_docs_list)
    new_doc_splits.extend(large_splits)
    
    # Medium chunks
    text_splitter_medium = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256, chunk_overlap=64
    )
    medium_splits = text_splitter_medium.split_documents(new_docs_list)
    new_doc_splits.extend(medium_splits)
    
    # Small chunks
    text_splitter_small = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=128, chunk_overlap=32
    )
    small_splits = text_splitter_small.split_documents(new_docs_list)
    new_doc_splits.extend(small_splits)
    
    # Add to existing vectorstore
    vectorstore.add_documents(new_doc_splits)
    
    # Save updated index
    try:
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"Updated FAISS index saved with {len(new_doc_splits)} new chunks")
    except Exception as e:
        print(f"Warning: Could not save updated FAISS index: {e}")
    
    return vectorstore

# Initialize vector store
vectorstore, embeddings = load_or_create_vectorstore()

# Create enhanced retriever with higher k value
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance for diverse results
    search_kwargs={
        "k": 10,  # Retrieve more documents for better coverage
        "lambda_mult": 0.7,  # Balance between relevance and diversity
        "fetch_k": 20  # Fetch more documents before MMR filtering
    }
)

# Create retriever tool with enhanced description
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    """Search and return comprehensive information about Lilian Weng's blog posts covering:
    - Reward hacking in AI systems and alignment
    - Hallucination in large language models
    - Diffusion models for video generation
    - Machine learning concepts, techniques, and research
    - AI safety and alignment topics
    This tool provides detailed technical information from authoritative blog posts.""",
)

# Initialize Vertex AI models with better configuration
response_model = ChatVertexAI(
    model_name="gemini-2.0-flash",
    temperature=0.1,  # Lower temperature for more consistent responses
    max_output_tokens=2048,  # Allow longer responses
)

# Enhanced MessagesState to track additional information
class EnhancedMessagesState(MessagesState):
    """Enhanced state to track retrieved documents and other metadata."""
    retrieved_docs: List[str] = Field(default_factory=list)
    retrieved_sources: List[str] = Field(default_factory=list)
    original_question: str = "" # This will now be set once per top-level query
    rewrite_count: int = 0
    confidence_score: float = 0.0

def generate_query_or_respond(state: EnhancedMessagesState):
    """
    Enhanced query processing with better tool use detection.
    """
    print("🤖 Processing user query...")
    
    # The original_question should be set when the graph is *invoked*
    # For now, we assume it's passed in the initial state or is the first HumanMessage.
    # We will rely on the caller to set original_question for each new query.
    # If this node is reached via a rewrite, original_question should already be set.
    
    # Ensure original_question is set from the initial human message if not already present
    if not state.get("original_question"):
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                state["original_question"] = msg.content
                break
    
    # Enhanced system prompt for better tool usage
    system_prompt = """You are an AI assistant with access to a comprehensive knowledge base about AI and machine learning topics from Lilian Weng's blog posts. 

When a user asks questions about:
- AI safety, alignment, or reward hacking
- Language model hallucinations
- Diffusion models or video generation
- Machine learning techniques and research
- AI safety and alignment topics

You should ALWAYS use the retrieve_blog_posts tool to get accurate, up-to-date information from the knowledge base.

If the question is completely unrelated to AI/ML topics (like cooking recipes, sports scores, etc.), you can respond directly without using tools."""
    
    messages_to_send_to_llm = [SystemMessage(content=system_prompt)] + state["messages"]

    response = (
        response_model
        .bind_tools([retriever_tool])
        .invoke(messages_to_send_to_llm)
    )
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print("🔍 Query requires document retrieval")
    else:
        print("💭 Query answered directly without retrieval")
    
    return {"messages": [response]}

# Enhanced document grading with more sophisticated evaluation
GRADE_PROMPT = """You are an expert document relevance evaluator for AI and machine learning content. 

User Question: {question}

Retrieved Documents (showing first 200 chars of each):
{context_preview}

Full Context Available: {total_docs} documents, {total_chars} total characters

Evaluation Task:
Determine if the retrieved documents contain information that can help answer the user's question about AI/ML topics.

Consider these factors:
1. Direct relevance: Does the content directly address the question?
2. Conceptual relevance: Are related concepts or background information present?
3. Technical depth: Is there sufficient technical detail to provide a comprehensive answer?
4. Coverage: Do the documents cover multiple aspects of the question if it's complex?

Grading Criteria:
- 'yes' if documents contain relevant information that can help answer the question
- 'no' only if documents are completely irrelevant or contain no useful information

Be generous in evaluation - partial relevance or related concepts should score 'yes'.

Your assessment (yes or no):"""

class GradeDocuments(BaseModel):
    """Enhanced document grading with confidence scoring."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )
    confidence: float = Field(
        description="Confidence in the assessment (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of the assessment"
    )

grader_model = ChatVertexAI(
    model="gemini-2.0-flash",
    temperature=0.0,  # Zero temperature for consistent grading
)

def grade_documents(state: EnhancedMessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """
    Enhanced document grading with better evaluation logic.
    """
    print("📊 Grading document relevance...")
    
    # Get the original question
    question = state.get("original_question", "")
    
    # Extract retrieved documents from tool messages
    retrieved_docs = []
    retrieved_sources = [] # Initialize retrieved_sources list
    
    for message in state["messages"]:
        if isinstance(message, ToolMessage):
            # The content of a ToolMessage is the result of the tool execution.
            # For your retriever tool, this content will be a string representation
            # of the retrieved documents.
            retrieved_docs.append(message.content)
            
            # Extract source information if available within the ToolMessage's content
            # or additional_kwargs, depending on how your retriever tool structures it.
            # Langchain's create_retriever_tool typically puts source in additional_kwargs.
            if hasattr(message, 'additional_kwargs') and 'source' in message.additional_kwargs:
                # This part depends on how your retriever actually returns sources.
                # If it's a list of dicts with 'source' key, you'd iterate.
                # For now, assuming it's directly accessible or needs more parsing.
                # If your tool returns a list of Document objects, you'd iterate those.
                # Example: If message.content was a list of Document objects, you'd do:
                # for doc in message.content:
                #     retrieved_sources.append(doc.metadata.get('source'))
                
                # A more robust way to get sources if the ToolNode returns Document objects
                # is to iterate the parsed content, not the raw string content.
                # The current setup `retrieved_docs.append(message.content)` assumes content is a string.
                # If ToolNode directly returned Document objects, message.content would be a list of Document.
                # Let's refine based on the expected output of `ToolNode([retriever_tool])`
                pass # The current `retrieved_docs.append(message.content)` gets the full string output.
                     # Sources are typically part of the Document object's metadata.
                     # If the `ToolNode` wraps individual Document objects in its output,
                     # you might need to parse `message.content` more thoroughly.
                     # However, for simplicity and typical Langchain behavior, the `message.content`
                     # from a retriever tool often combines all document contents into one string.
                     # To get individual sources, we need to access the original Document objects.
                     # The `ToolNode` usually passes the *raw* output of the tool.
                     # Let's assume for a moment that `retriever_tool` returns a list of Document objects,
                     # and the `ToolNode` converts it to a string for `message.content`.
                     # If so, you'd need a more complex parsing or rely on the agent to structure it.

                     # For now, we'll keep `retrieved_sources` as it is, assuming your `retriever_tool`
                     # isn't explicitly putting structured sources into `ToolMessage.additional_kwargs`
                     # in a way that's easily extracted here without more parsing.
                     # If your `retriever_tool` actually returns structured Document objects,
                     # you'd need to modify `retrieve` node to store them in a structured way in the state,
                     # or parse `message.content` here.
    
    if not retrieved_docs:
        print("❌ No retrieved documents found")
        return "rewrite_question"
    
    # Store retrieved docs and sources in state
    state["retrieved_docs"] = retrieved_docs
    # You might want to update state["retrieved_sources"] if you manage to extract them.
    # For now, it will remain an empty list if not explicitly populated.
    
    # Create context preview for evaluation
    context_preview = []
    total_chars = 0
    for i, doc in enumerate(retrieved_docs):
        preview = doc[:200] + "..." if len(doc) > 200 else doc
        context_preview.append(f"Doc {i+1}: {preview}")
        total_chars += len(doc)
    
    print(f"📋 Evaluating {len(retrieved_docs)} documents ({total_chars} total chars)")
    print(f"📋 Question: {question[:150]}...")
    
    prompt = GRADE_PROMPT.format(
        question=question,
        context_preview="\n".join(context_preview),
        total_docs=len(retrieved_docs),
        total_chars=total_chars
    )
    
    try:
        response = (
            grader_model
            .with_structured_output(GradeDocuments)
            .invoke([{"role": "user", "content": prompt}])
        )
        
        score = response.binary_score.lower().strip()
        confidence = response.confidence
        reasoning = response.reasoning
        
        # Store confidence in state
        state["confidence_score"] = confidence
        
        print(f"📋 Relevance: {score} (confidence: {confidence:.2f})")
        print(f"📋 Reasoning: {reasoning}")
        
        if score == "yes":
            print("✅ Documents are relevant - generating answer")
            return "generate_answer"
        else:
            print("❌ Documents not relevant - checking rewrite options")
            rewrite_count = state.get("rewrite_count", 0)
            if rewrite_count >= 2:
                print("⚠️ Max rewrites reached - generating answer with available context")
                return "generate_answer"
            return "rewrite_question"
            
    except Exception as e:
        print(f"❌ Error in document grading: {e}")
        return "generate_answer"

# Enhanced question rewriting with more sophisticated strategies
REWRITE_PROMPT = """You are an expert at optimizing search queries for AI and machine learning content.

Original Question: {question}
Attempt Number: {attempt}
Previous searches didn't return optimal results.

Rewrite this question to improve retrieval from a knowledge base containing:
- AI safety and alignment research
- Language model hallucination studies  
- Diffusion model and video generation techniques
- Machine learning methodologies
- Technical AI research papers and blog posts

Rewriting Strategies for Attempt {attempt}:
{strategy}

Focus on:
- Using precise technical terminology
- Breaking complex questions into key concepts
- Including relevant synonyms and related terms
- Making the query more specific to the domain

Rewritten Question:"""

def rewrite_question(state: EnhancedMessagesState):
    """
    Enhanced question rewriting with different strategies per attempt.
    """
    print("✏️ Rewriting question for better retrieval...")
    
    rewrite_count = state.get("rewrite_count", 0) + 1
    state["rewrite_count"] = rewrite_count
    
    question = state.get("original_question", "")
    
    # Different strategies based on attempt number
    strategies = {
        1: """Strategy 1 - Technical Expansion:
        - Add technical terms and synonyms
        - Include related concepts
        - Make terminology more precise""",
        
        2: """Strategy 2 - Concept Breakdown:
        - Break complex questions into core concepts
        - Focus on fundamental aspects
        - Simplify while maintaining technical accuracy"""
    }
    
    strategy = strategies.get(rewrite_count, strategies[2]) # Fallback to strategy 2 if more attempts
    
    prompt = REWRITE_PROMPT.format(
        question=question,
        attempt=rewrite_count,
        strategy=strategy
    )
    
    response = response_model.invoke([{"role": "user", "content": prompt}])
    
    print(f"📝 Original: {question}")
    print(f"📝 Rewritten (attempt {rewrite_count}): {response.content}")
    
    # Crucially, the rewritten question should be added as a *new HumanMessage*
    # to the state, so the next `query_handler` uses it for tool binding.
    return {"messages": [HumanMessage(content=response.content)]}

# Significantly enhanced answer generation
GENERATE_PROMPT = """You are a knowledgeable AI assistant specializing in machine learning and AI research. You have access to comprehensive information from Lilian Weng's authoritative blog posts.

Original Question: {question}

Retrieved Context ({num_docs} documents, {total_chars} characters):
{context}

Instructions for High-Accuracy Response:
1. COMPREHENSIVE ANALYSIS: Thoroughly analyze all retrieved information
2. DIRECT ANSWERS: Provide specific, direct answers to the question
3. TECHNICAL ACCURACY: Use precise technical terminology and concepts
4. STRUCTURED RESPONSE: Organize information logically with clear sections
5. EVIDENCE-BASED: Support claims with specific information from the context
6. COMPLETENESS: Address all aspects of the question when possible

Response Structure:
- Start with a direct answer to the main question
- Provide detailed explanation with technical details
- Include relevant examples or applications when available
- Mention any limitations or caveats if applicable
- End with a brief summary if the topic is complex

Quality Standards:
- Be thorough but concise (aim for 150-400 words depending on complexity)
- Use the retrieved information extensively
- Maintain technical accuracy
- Provide actionable insights when relevant

Generate your comprehensive response:"""

def generate_answer(state: EnhancedMessagesState):
    """
    Enhanced answer generation with comprehensive context utilization.
    """
    print("📖 Generating comprehensive answer from retrieved documents...")
    
    question = state.get("original_question", "")
    retrieved_docs = state.get("retrieved_docs", [])
    confidence = state.get("confidence_score", 0.0)
    
    if not retrieved_docs:
        print("WARNING: No retrieved documents found during answer generation. Proceeding with limited context.")
        # This case should ideally be caught earlier by grade_documents, but as a fallback.
        # Fallback: extract from tool messages if retrieved_docs wasn't set earlier
        for message in state["messages"]:
            if isinstance(message, ToolMessage):
                retrieved_docs.append(message.content)
    
    # Prepare context with document numbering for better organization
    if retrieved_docs:
        context_parts = []
        total_chars = 0
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc}")
            total_chars += len(doc)
        context = "\n\n" + "="*50 + "\n\n".join(context_parts)
    else:
        context = "No specific documents retrieved."
        total_chars = 0
    
    prompt = GENERATE_PROMPT.format(
        question=question,
        num_docs=len(retrieved_docs),
        total_chars=total_chars,
        context=context
    )
    
    # Use higher temperature for more comprehensive responses
    enhanced_model = ChatVertexAI(
        model_name="gemini-2.0-flash",
        temperature=0.2, # Slightly higher temperature for more comprehensive response
        max_output_tokens=2048,
    )
    
    response = enhanced_model.invoke([{"role": "user", "content": prompt}])
    
    print(f"✅ Comprehensive answer generated (confidence: {confidence:.2f})")
    # Reset rewrite_count for the next independent query
    state["rewrite_count"] = 0
    state["original_question"] = "" # Clear original_question for the next independent query
    
    return {"messages": [response]}

def update_vectorstore_with_new_urls(new_urls):
    """Update the vectorstore with new URLs using enhanced chunking."""
    global vectorstore, retriever, retriever_tool
    
    vectorstore = add_documents_to_vectorstore(vectorstore, new_urls)
    
    # Recreate enhanced retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "lambda_mult": 0.7,
            "fetch_k": 20
        }
    )
    
    # Update retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        """Search and return comprehensive information about Lilian Weng's blog posts covering:
        - Reward hacking in AI systems and alignment
        - Hallucination in large language models  
        - Diffusion models for video generation
        - Machine learning concepts, techniques, and research
        - AI safety and alignment topics
        This tool provides detailed technical information from authoritative blog posts.""",
    )

# Assemble the Enhanced RAG Agent Graph
workflow = StateGraph(EnhancedMessagesState)

# Add nodes
workflow.add_node("query_handler", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

# Define workflow edges
workflow.add_edge(START, "query_handler")

workflow.add_conditional_edges(
    "query_handler",
    tools_condition,
    {
        "tools": "retrieve",
        END: END, # If no tool is called, the LLM directly answered, so end.
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question",
    }
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "query_handler") # Go back to query_handler with the rewritten question

# Compile the enhanced graph
graph = workflow.compile()
