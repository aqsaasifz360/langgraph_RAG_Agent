import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from tools import (
    build_retriever_tool,
    generate_query_or_respond,
    rewrite_question,
    generate_answer,
    grade_documents,
)

# Load environment variables
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

# Build retriever tool
retriever_tool = build_retriever_tool()

# Build graph
workflow = StateGraph(MessagesState)

workflow.add_node("generate_query_or_respond", lambda state: generate_query_or_respond(state, retriever_tool))
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()
