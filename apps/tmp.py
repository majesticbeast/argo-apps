import os
from typing import Dict, Any, List, Literal
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from portkey_ai import Portkey

# Configuration
class ModelTier(Enum):
    FAST = "fast"      # GPT-3.5, Claude Haiku - for simple tasks
    BALANCED = "balanced"  # Claude Sonnet - for most tasks
    PREMIUM = "premium"    # GPT-4, Claude Opus - for complex/sensitive

class QueryType(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    COMPLAINT = "complaint"
    GENERAL = "general"
    PRODUCT = "product"

class ComplexityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# State definition
class SupportState(TypedDict):
    messages: Annotated[List, add_messages]
    original_query: str
    query_type: QueryType
    complexity_level: ComplexityLevel
    knowledge_base_results: List[str]
    generated_solution: str
    quality_score: float
    escalation_required: bool
    escalation_reason: str
    final_response: str
    model_usage: Dict[str, Any]
    workflow_step: str

# Portkey configuration
class PortkeyModelRouter:
    def __init__(self, api_key: str, config_id: str):
        self.client = Portkey(
            api_key=api_key,
            config=config_id  # Your Portkey config with model fallbacks
        )
        
        # Model routing configuration
        self.model_config = {
            ModelTier.FAST: {
                "model": "gpt-3.5-turbo",
                "max_tokens": 500,
                "temperature": 0.1
            },
            ModelTier.BALANCED: {
                "model": "claude-3-sonnet-20240229", 
                "max_tokens": 1000,
                "temperature": 0.3
            },
            ModelTier.PREMIUM: {
                "model": "gpt-4-turbo-preview",
                "max_tokens": 2000,
                "temperature": 0.2
            }
        }
    
    def generate_response(self, messages: List, tier: ModelTier, metadata: Dict = None):
        """Generate response using specified model tier through Portkey"""
        config = self.model_config[tier]
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                **config,
                metadata=metadata or {}
            )
            return response.choices[0].message.content
        except Exception as e:
            # Portkey will handle fallbacks automatically
            print(f"Model call failed: {e}")
            raise

# Knowledge base (simplified for demo)
KNOWLEDGE_BASE = {
    QueryType.BILLING: [
        "Billing cycles run monthly from the 1st to the last day of each month",
        "Payment methods can be updated in Account Settings > Billing",
        "Refunds are processed within 5-7 business days"
    ],
    QueryType.TECHNICAL: [
        "API rate limits are 1000 requests per minute for paid plans",
        "Authentication uses Bearer tokens in the Authorization header",
        "Webhook endpoints must respond with 200 status within 10 seconds"
    ],
    QueryType.PRODUCT: [
        "New features are released every two weeks",
        "Enterprise features include SSO, custom integrations, and priority support",
        "Free tier includes 100 API calls per day"
    ]
}

class SupportWorkflow:
    def __init__(self, portkey_api_key: str, portkey_config_id: str):
        self.router = PortkeyModelRouter(portkey_api_key, portkey_config_id)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        graph = StateGraph(SupportState)
        
        # Add nodes
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("assess_complexity", self._assess_complexity)
        graph.add_node("search_knowledge", self._search_knowledge)
        graph.add_node("generate_solution", self._generate_solution)
        graph.add_node("validate_quality", self._validate_quality)
        graph.add_node("decide_escalation", self._decide_escalation)
        graph.add_node("format_response", self._format_response)
        
        # Define workflow edges
        graph.add_edge("classify_intent", "assess_complexity")
        graph.add_edge("assess_complexity", "search_knowledge")
        graph.add_edge("search_knowledge", "generate_solution")
        graph.add_edge("generate_solution", "validate_quality")
        
        # Conditional routing after quality validation
        graph.add_conditional_edges(
            "validate_quality",
            self._should_retry_solution,
            {
                "retry": "generate_solution",
                "continue": "decide_escalation"
            }
        )
        
        # Conditional routing after escalation decision
        graph.add_conditional_edges(
            "decide_escalation",
            self._should_escalate,
            {
                "escalate": END,  # End workflow, human takes over
                "respond": "format_response"
            }
        )
        
        graph.add_edge("format_response", END)
        graph.set_entry_point("classify_intent")
        
        return graph.compile()
    
    def _classify_intent(self, state: SupportState) -> SupportState:
        """Classify the customer query intent using fast model"""
        state["workflow_step"] = "intent_classification"
        
        messages = [
            SystemMessage(content="""Classify this customer support query into one of these categories:
            - billing: Payment, invoices, refunds, subscription issues
            - technical: API, integration, bugs, performance issues  
            - complaint: Service dissatisfaction, problems with experience
            - product: Feature requests, product information, capabilities
            - general: Other inquiries
            
            Respond with just the category name."""),
            HumanMessage(content=state["original_query"])
        ]
        
        classification = self.router.generate_response(
            messages, 
            ModelTier.FAST,
            metadata={"step": "intent_classification", "query_id": hash(state["original_query"])}
        )
        
        # Parse classification
        try:
            state["query_type"] = QueryType(classification.strip().lower())
        except ValueError:
            state["query_type"] = QueryType.GENERAL
        
        return state
    
    def _assess_complexity(self, state: SupportState) -> SupportState:
        """Assess query complexity using fast model"""
        state["workflow_step"] = "complexity_assessment"
        
        messages = [
            SystemMessage(content="""Assess the complexity of this customer query:
            - low: Simple questions with straightforward answers
            - medium: Requires some explanation or multiple steps
            - high: Complex issues requiring detailed technical knowledge or sensitive handling
            
            Consider factors like technical depth, emotional tone, and potential business impact.
            Respond with just: low, medium, or high"""),
            HumanMessage(content=f"Query type: {state['query_type'].value}\nQuery: {state['original_query']}")
        ]
        
        complexity = self.router.generate_response(
            messages,
            ModelTier.FAST,
            metadata={"step": "complexity_assessment"}
        )
        
        try:
            state["complexity_level"] = ComplexityLevel(complexity.strip().lower())
        except ValueError:
            state["complexity_level"] = ComplexityLevel.MEDIUM
        
        return state
    
    def _search_knowledge(self, state: SupportState) -> SupportState:
        """Search knowledge base for relevant information"""
        state["workflow_step"] = "knowledge_search"
        
        # Simple knowledge base lookup (in production, use vector search)
        kb_results = KNOWLEDGE_BASE.get(state["query_type"], [])
        
        # Filter results based on query relevance (simplified)
        relevant_results = []
        query_lower = state["original_query"].lower()
        
        for item in kb_results:
            # Simple keyword matching (use semantic search in production)
            if any(word in item.lower() for word in query_lower.split()):
                relevant_results.append(item)
        
        state["knowledge_base_results"] = relevant_results or kb_results[:2]  # Fallback to top 2
        return state
    
    def _generate_solution(self, state: SupportState) -> SupportState:
        """Generate solution using appropriate model tier"""
        state["workflow_step"] = "solution_generation"
        
        # Choose model tier based on complexity and query type
        if state["complexity_level"] == ComplexityLevel.HIGH or state["query_type"] == QueryType.COMPLAINT:
            tier = ModelTier.PREMIUM
        elif state["complexity_level"] == ComplexityLevel.MEDIUM:
            tier = ModelTier.BALANCED
        else:
            tier = ModelTier.FAST
        
        kb_context = "\n".join(state["knowledge_base_results"])
        
        messages = [
            SystemMessage(content=f"""You are a helpful customer support agent. Generate a solution for this {state['query_type'].value} query.
            
            Use this knowledge base information when relevant:
            {kb_context}
            
            Guidelines:
            - Be helpful, professional, and empathetic
            - Provide specific steps when possible
            - If you cannot fully resolve the issue, explain what you can help with
            - Keep responses concise but complete"""),
            HumanMessage(content=state["original_query"])
        ]
        
        solution = self.router.generate_response(
            messages,
            tier,
            metadata={"step": "solution_generation", "tier": tier.value}
        )
        
        state["generated_solution"] = solution
        return state
    
    def _validate_quality(self, state: SupportState) -> SupportState:
        """Validate solution quality using balanced model"""
        state["workflow_step"] = "quality_validation"
        
        messages = [
            SystemMessage(content="""Evaluate this customer support response on a scale of 0-10:
            
            Criteria:
            - Addresses the customer's question directly
            - Provides actionable information
            - Maintains professional tone
            - Is clear and easy to understand
            
            Respond with just a number from 0-10."""),
            HumanMessage(content=f"Original Query: {state['original_query']}\n\nGenerated Response: {state['generated_solution']}")
        ]
        
        score_text = self.router.generate_response(
            messages,
            ModelTier.BALANCED,
            metadata={"step": "quality_validation"}
        )
        
        try:
            state["quality_score"] = float(score_text.strip())
        except ValueError:
            state["quality_score"] = 5.0  # Default to medium score
        
        return state
    
    def _decide_escalation(self, state: SupportState) -> SupportState:
        """Decide if escalation to human is needed"""
        state["workflow_step"] = "escalation_decision"
        
        escalation_reasons = []
        
        # Check various escalation triggers
        if state["quality_score"] < 6.0:
            escalation_reasons.append("Low quality solution")
        
        if state["query_type"] == QueryType.COMPLAINT and state["complexity_level"] == ComplexityLevel.HIGH:
            escalation_reasons.append("High-complexity complaint requires human touch")
        
        # Check for escalation keywords
        escalation_keywords = ["angry", "furious", "lawsuit", "cancel", "unacceptable", "terrible"]
        if any(keyword in state["original_query"].lower() for keyword in escalation_keywords):
            escalation_reasons.append("Escalation keywords detected")
        
        state["escalation_required"] = len(escalation_reasons) > 0
        state["escalation_reason"] = "; ".join(escalation_reasons) if escalation_reasons else ""
        
        return state
    
    def _format_response(self, state: SupportState) -> SupportState:
        """Format final response for customer"""
        state["workflow_step"] = "response_formatting"
        
        messages = [
            SystemMessage(content="""Format this customer support response to be professional, friendly, and helpful.
            Add appropriate greeting and closing. Ensure the tone matches the query type and complexity."""),
            HumanMessage(content=f"Query Type: {state['query_type'].value}\nComplexity: {state['complexity_level'].value}\n\nResponse to format: {state['generated_solution']}")
        ]
        
        formatted_response = self.router.generate_response(
            messages,
            ModelTier.FAST,
            metadata={"step": "response_formatting"}
        )
        
        state["final_response"] = formatted_response
        return state
    
    def _should_retry_solution(self, state: SupportState) -> Literal["retry", "continue"]:
        """Decide if solution needs to be regenerated"""
        # Retry if quality is very low and we haven't tried multiple times
        if state["quality_score"] < 4.0:
            return "retry"
        return "continue"
    
    def _should_escalate(self, state: SupportState) -> Literal["escalate", "respond"]:
        """Decide if we should escalate to human"""
        return "escalate" if state["escalation_required"] else "respond"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a customer support query through the workflow"""
        initial_state = {
            "messages": [],
            "original_query": query,
            "workflow_step": "starting",
            "model_usage": {}
        }
        
        # Run the workflow
        result = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "query_type": result["query_type"].value,
            "complexity": result["complexity_level"].value,
            "quality_score": result["quality_score"],
            "escalation_required": result["escalation_required"],
            "escalation_reason": result.get("escalation_reason", ""),
            "final_response": result.get("final_response", "Escalated to human agent"),
            "workflow_path": result["workflow_step"]
        }

# Demo usage
if __name__ == "__main__":
    # Initialize the workflow
    # You'll need to set your Portkey API key and config ID
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY", "your-portkey-api-key")
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID", "your-config-id")
    
    workflow = SupportWorkflow(PORTKEY_API_KEY, PORTKEY_CONFIG_ID)
    
    # Test queries
    test_queries = [
        "My API calls are returning 429 errors constantly",
        "I want a refund for last month's charge",
        "This service is absolutely terrible and I'm extremely frustrated",
        "How do I update my payment method?",
        "Can you explain your enterprise features?"
    ]
    
    print("Customer Support Escalation System Demo")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = workflow.process_query(query)
        
        print(f"Type: {result['query_type']}")
        print(f"Complexity: {result['complexity']}")
        print(f"Quality Score: {result['quality_score']}")
        print(f"Escalated: {result['escalation_required']}")
        if result['escalation_required']:
            print(f"Escalation Reason: {result['escalation_reason']}")
        else:
            print(f"Response: {result['final_response'][:100]}...")
        print("-" * 30)
