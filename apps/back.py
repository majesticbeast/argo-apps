# main.py
# ---
# This file contains the complete backend for the Bedrock Research Agent.
# It uses FastAPI for the web server, LangGraph to define the agent workflow,
# and Portkey to manage and observe calls to Amazon Bedrock.

# --- 1. Installation ---
# Before running, install the necessary packages:
# pip install "fastapi[all]" uvicorn python-dotenv portkey-ai langgraph langchain-community langchain-anthropic boto3

import os
import json
from typing import List, TypedDict
from dotenv import load_dotenv

import boto3
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
# PORTKEY_BEDROCK_VIRTUAL_KEY="YOUR_BEDROCK_VIRTUAL_KEY"

load_dotenv()

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

class Plan(LangchainBaseModel):
    """Plan to follow in order to answer the user query."""
    steps: List[str] = Field(
        description="list of research topics to search for, in order to answer the user's query"
    )

class GraphState(TypedDict):
    """Represents the state of our graph."""
    query: str
    plan: List[str]
    summaries: List[str]
    report: str

def planner_node(state: GraphState, llm: ChatBedrock):
    print("--- AGENT: Planner ---")
    structured_llm = llm.with_structured_output(Plan)
    prompt = f"You are an expert research planner. For the query: '{state['query']}', what are the distinct topics you would need to research?"
    plan_object = structured_llm.invoke(prompt)
    print(f"Generated Plan: {plan_object.steps}")
    return {"plan": plan_object.steps, "summaries": []}

def search_node(state: GraphState):
    print("--- AGENT: Search ---")
    current_topic = state["plan"][0]
    search_result = tavily_search_stub(current_topic)
    return {"search_result": search_result, "current_topic": current_topic}

def summarizer_node(state: GraphState, llm: ChatBedrock):
    print("--- AGENT: Summarizer ---")
    search_result = state.pop("search_result")
    current_topic = state.pop("current_topic")
    prompt = f"You are an expert summarizer. Based on the following content, please write a concise summary about '{current_topic}'.\n\nContent:\n{search_result}"
    summary_message = llm.invoke([HumanMessage(content=prompt)])
    summary = summary_message.content
    print(f"Generated Summary: {summary}")
    updated_summaries = state["summaries"] + [summary]
    updated_plan = state["plan"][1:]
    return {"summaries": updated_summaries, "plan": updated_plan}

def report_node(state: GraphState, llm: ChatBedrock):
    print("--- AGENT: Report ---")
    prompt = f"You are an expert report writer. Synthesize the following summaries into a single, well-structured report that answers the user's original query.\n\nUser Query: {state['query']}\n\nSummaries:\n- {'\n- '.join(state['summaries'])}\n\nPlease format the final output in Markdown."
    report_message = llm.invoke([HumanMessage(content=prompt)])
    report = report_message.content
    print("Generated Final Report.")
    return {"report": report}

def should_continue(state: GraphState) -> str:
    if len(state["plan"]) > 0:
        print("--- DECISION: Continue to Search ---")
        return "search"
    else:
        print("--- DECISION: Proceed to Report ---")
        return "report"

# --- 5. FastAPI Server ---

class ReportRequest(BaseModel):
    query: str
    portkeyVirtualKey: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Bedrock Research Agent Backend is running."}

@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    print(f"Received request with query: '{request.query}' and virtual key: '{request.portkeyVirtualKey}'")

    # --- CORRECTED: Portkey & Bedrock LLM Initialization ---
    # This is the Portkey Gateway URL for Bedrock. All Boto3 requests will be sent here.
    PORTKEY_GATEWAY_URL = "https://api.portkey.ai/v1/proxy/bedrock"
    # The AWS region your Bedrock models are in.
    AWS_REGION = "us-east-1" 

    # 1. Create a custom Boto3 client that points to the Portkey Gateway
    bedrock_runtime_client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        endpoint_url=PORTKEY_GATEWAY_URL
    )

    # 2. Initialize ChatBedrock with the custom client and Portkey headers
    portkey_bedrock_llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock_runtime_client, # Use the client pointing to Portkey
        model_kwargs={"temperature": 0.1},
        streaming=False,
        extra_headers={
            "x-portkey-api-key": os.getenv("PORTKEY_API_KEY"),
            "x-portkey-virtual-key": request.portkeyVirtualKey,
            "x-portkey-metadata": json.dumps({"app": "research-agent-demo", "_user": "demo-user"})
        }
    )

    # --- Graph Assembly ---
    workflow = StateGraph(GraphState)
    workflow.add_node("planner", lambda state: planner_node(state, portkey_bedrock_llm))
    workflow.add_node("search", search_node)
    workflow.add_node("summarizer", lambda state: summarizer_node(state, portkey_bedrock_llm))
    workflow.add_node("report", lambda state: report_node(state, portkey_bedrock_llm))
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner", should_continue, {"search": "search", "report": "report"})
    workflow.add_edge("search", "summarizer")
    workflow.add_conditional_edges("summarizer", should_continue, {"search": "search", "report": "report"})
    workflow.add_edge("report", END)
    graph_app = workflow.compile()

    # --- Run the Graph and Get Response ---
    inputs = {"query": request.query}
    final_state = graph_app.invoke(inputs)
    trace_id = final_state.get('__portkey_metadata__', {}).get('trace_id', 'not-found')
    
    return {
        "report": final_state.get("report", "Error: Could not generate report."),
        "trace_url": f"https://app.portkey.ai/traces/{trace_id}"
    }

# --- 6. How to Run ---
# 1. Make sure you have a .env file as described in Section 2.
# 2. Open your terminal in this directory.
# 3. Run the server using uvicorn: uvicorn main:app --reload
