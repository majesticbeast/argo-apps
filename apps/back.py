# main.py
# ---
# This file contains the complete backend for the Bedrock Research Agent.
# It uses FastAPI for the web server, LangGraph to define the agent workflow,
# and Portkey to manage and observe calls to Amazon Bedrock.

# --- 1. Installation ---
# Before running, install the necessary packages:
# pip install "fastapi[all]" uvicorn python-dotenv portkey-ai langgraph langchain-community langchain-anthropic

import os
import json
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.chat_models.bedrock import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel as LangchainBaseModel, Field

from langgraph.graph import StateGraph, END

# --- 2. Environment & Portkey Setup ---
# Load environment variables from a .env file.
# Create a file named ".env" in the same directory with the following content:
#
# PORTKEY_API_KEY="YOUR_PORTKEY_API_KEY"
# # Virtual Keys created in the Portkey Dashboard for your Bedrock provider
# PORTKEY_SONNET_VIRTUAL_KEY="YOUR_SONNET_VIRTUAL_KEY"
# PORTKEY_HAIKU_VIRTUAL_KEY="YOUR_HAIKU_VIRTUAL_KEY"

load_dotenv()

# Portkey will automatically pick up the API key from the environment variables.
# We will initialize the Portkey client when a request comes in, so we can
# dynamically set the virtual key from the frontend.

# --- 3. Mock Search Tool ---
# To make the demo reliable and avoid external dependencies, we'll use a mock
# search tool. In a real application, this could be a call to Tavily, Google Search, etc.
def tavily_search_stub(topic: str) -> str:
    """A mock search tool that returns pre-canned text based on keywords."""
    print(f"--- MOCK SEARCH: Searching for '{topic}' ---")
    topic = topic.lower()
    if "re:invent" in topic:
        return """
        At AWS re:Invent 2024, the major announcements for generative AI revolved around the new Amazon Titan V2 model suite,
        a significant upgrade to Bedrock Agents with a visual graph editor, and the introduction of enhanced Guardrails for Bedrock.
        Titan V2 Pro is positioned as a direct competitor to models like GPT-4 and Gemini Ultra.
        The new agent builder simplifies creating complex, multi-step workflows, and Guardrails provide enterprise-grade safety features.
        """
    elif "google" in topic:
        return """
        Google's latest offerings in the generative AI space are centered on its Gemini family of models.
        Gemini 1.5 Pro, with its industry-leading 1 million token context window, remains a key feature.
        Google also introduced Gemini 1.5 Flash, a lighter, faster, and more cost-effective model for high-throughput tasks.
        Their primary platform for building with these models is the Vertex AI Agent Builder.
        """
    else:
        return f"No specific information found for '{topic}'. The AI landscape is constantly evolving with contributions from many key players."

# --- 4. LangGraph State and Agent Definition ---

# This Pydantic model defines the structure for our planning agent's output.
class Plan(LangchainBaseModel):
    """Plan to follow in order to answer the user query."""
    steps: List[str] = Field(
        description="list of research topics to search for, in order to answer the user's query"
    )

# This TypedDict defines the state of our graph. It's how data is passed between nodes.
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: The initial user query.
        plan: A list of research topics to investigate.
        summaries: A list of summaries for each researched topic.
        report: The final, consolidated report.
    """
    query: str
    plan: List[str]
    summaries: List[str]
    report: str

# Node Functions
def planner_node(state: GraphState, llm: ChatBedrock):
    """
    This node takes the user's query and creates a research plan.
    It uses a structured output call to ensure the plan is a list of strings.
    """
    print("--- AGENT: Planner ---")
    # We wrap the LLM with .with_structured_output to get a clean list of steps
    structured_llm = llm.with_structured_output(Plan)
    
    prompt = f"""You are an expert research planner.
    Your goal is to create a step-by-step research plan to answer the user's query.
    For the query: '{state['query']}', what are the distinct topics you would need to research?
    """
    
    plan_object = structured_llm.invoke(prompt)
    
    print(f"Generated Plan: {plan_object.steps}")
    return {"plan": plan_object.steps, "summaries": []}

def search_node(state: GraphState):
    """
    This node takes the next topic from the plan and "searches" for it using our mock tool.
    """
    print("--- AGENT: Search ---")
    # We'll work on the first topic in the plan list
    current_topic = state["plan"][0]
    
    # Use the mock search tool
    search_result = tavily_search_stub(current_topic)
    
    # The result of the search is passed to the summarizer node in the next step.
    # We don't update the state directly here, as the summarizer will do that.
    return {"search_result": search_result, "current_topic": current_topic}

def summarizer_node(state: GraphState, llm: ChatBedrock):
    """
    This node takes the search result and summarizes it.
    """
    print("--- AGENT: Summarizer ---")
    search_result = state.pop("search_result") # Pop from state as it's temporary
    current_topic = state.pop("current_topic")

    prompt = f"""You are an expert summarizer.
    Based on the following content, please write a concise summary about '{current_topic}'.

    Content:
    {search_result}
    """
    
    summary_message = llm.invoke([HumanMessage(content=prompt)])
    summary = summary_message.content
    print(f"Generated Summary: {summary}")
    
    # Add the new summary to our list of summaries
    updated_summaries = state["summaries"] + [summary]
    
    # Remove the topic we just finished from the plan
    updated_plan = state["plan"][1:]
    
    return {"summaries": updated_summaries, "plan": updated_plan}

def report_node(state: GraphState, llm: ChatBedrock):
    """
    This node takes all the summaries and generates a final, consolidated report.
    """
    print("--- AGENT: Report ---")
    summaries = state["summaries"]
    query = state["query"]
    
    prompt = f"""You are an expert report writer.
    Your task is to synthesize the following summaries into a single, well-structured report that answers the user's original query.
    
    User Query: {query}
    
    Summaries:
    - {"\n- ".join(summaries)}
    
    Please format the final output in Markdown.
    """
    
    report_message = llm.invoke([HumanMessage(content=prompt)])
    report = report_message.content
    print(f"Generated Final Report.")
    
    return {"report": report}

# Conditional Edge
def should_continue(state: GraphState) -> str:
    """
    This function determines the next step in the graph.
    If there are topics left in the plan, it continues to 'search'.
    Otherwise, it proceeds to 'report'.
    """
    if len(state["plan"]) > 0:
        print("--- DECISION: Continue to Search ---")
        return "search"
    else:
        print("--- DECISION: Proceed to Report ---")
        return "report"

# --- 5. FastAPI Server ---

# Pydantic model for the request body
class ReportRequest(BaseModel):
    query: str
    portkeyVirtualKey: str

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def read_root():
    return {"message": "Bedrock Research Agent Backend is running."}

@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    """
    This endpoint receives a query from the frontend, runs the LangGraph agent,
    and returns the final report along with Portkey metadata.
    """
    print(f"Received request with query: '{request.query}' and virtual key: '{request.portkeyVirtualKey}'")

    # --- Portkey & Bedrock LLM Initialization ---
    # We initialize the LLM here to use the specific virtual key from the request.
    # This allows the frontend to control which model configuration Portkey uses.
    # As requested, no explicit AWS credentials are needed. Portkey/Boto3 will
    # find them from the environment (e.g., IAM role on EC2/Lambda).
    portkey_bedrock_llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0", # A default model
        client=None, # Let LangChain's Bedrock client handle this
        model_kwargs={"temperature": 0.1},
        # This is where Portkey magic happens!
        # All requests will be routed through Portkey using this virtual key.
        streaming=False,
        extra_headers={
            "x-portkey-api-key": os.getenv("PORTKEY_API_KEY"),
            "x-portkey-virtual-key": request.portkeyVirtualKey,
            # Add metadata for better observability in Portkey
            "x-portkey-metadata": json.dumps({"app": "research-agent-demo", "_user": "demo-user"})
        }
    )

    # --- Graph Assembly ---
    workflow = StateGraph(GraphState)

    # Add nodes to the graph
    workflow.add_node("planner", lambda state: planner_node(state, portkey_bedrock_llm))
    workflow.add_node("search", search_node)
    workflow.add_node("summarizer", lambda state: summarizer_node(state, portkey_bedrock_llm))
    workflow.add_node("report", lambda state: report_node(state, portkey_bedrock_llm))

    # Define the graph's flow
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "decide_to_continue")
    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "search": "search",
            "report": "report",
        },
    )
    workflow.add_edge("search", "summarizer")
    workflow.add_conditional_edges(
        "summarizer",
        should_continue,
        {
            "search": "search",
            "report": "report",
        },
    )
    workflow.add_edge("report", END)

    # Compile the graph into a runnable app
    graph_app = workflow.compile()

    # --- Run the Graph and Get Response ---
    inputs = {"query": request.query}
    final_state = graph_app.invoke(inputs)

    # The Portkey trace ID is automatically added to the response metadata
    # by the LangChain integration.
    trace_id = final_state.get('__portkey_metadata__', {}).get('trace_id', 'not-found')
    trace_url = f"https://app.portkey.ai/traces/{trace_id}"

    return {
        "report": final_state.get("report", "Error: Could not generate report."),
        "trace_url": trace_url
    }

# --- 6. How to Run ---
#
# 1. Make sure you have a .env file as described in Section 2.
#
# 2. Open your terminal in this directory.
#
# 3. Run the server using uvicorn:
#    uvicorn main:app --reload
#
# 4. The server will be running at http://127.0.0.1:8000
#
# 5. You can now connect your React frontend to this backend.
