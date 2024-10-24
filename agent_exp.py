from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Define the agents
def create_agent(name):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are {name}. Engage in a conversation with another agent."),
        ("human", "{input}"),
    ])
    return prompt | ChatOpenAI()

agent1 = create_agent("Agent 1")
agent2 = create_agent("Agent 2")

# Define the nodes in the graph
def agent1_node(state):
    if "agent2_response" in state:
        input_message = state["agent2_response"]
    else:
        input_message = state.get("initial_message", "Hello, how are you today?")
    
    response = agent1.invoke({"input": input_message})
    return {"agent1_response": response.content}

def agent2_node(state):
    response = agent2.invoke({"input": state["agent1_response"]})
    return {"agent2_response": response.content}

# Create the graph
workflow = Graph()

# Add nodes to the graph
workflow.add_node("agent1", agent1_node)
workflow.add_node("agent2", agent2_node)

# Connect the nodes
workflow.add_edge("agent1", "agent2")
workflow.add_edge("agent2", "agent1")

# Set the entry point
workflow.set_entry_point("agent1")

# Compile the graph
app = workflow.compile()

app.get_graph().draw_png("agent_exp.png")

# Run the conversation
for step in app.stream({
    "initial_message": "Hello, how are you today?"
}):
    print(step)