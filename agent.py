from utils.node import host_node, player_node
from langgraph.graph import StateGraph, START,END
from utils.state import AgentState
from utils.tools import tool_node
from langchain_core.messages import SystemMessage

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("host", host_node)
workflow.add_node("player", player_node)
workflow.add_node("call_tool", tool_node)


def router(state):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "call_tool"
    
    if state["question_answered"] == 20:
        print ("question answered 20 go to end")
        return "end"
    
    if state["question_asked"] == 20:
        print ("question answered 20 go to end")
        return "end"
    
    if state["guess"] == state["topic"]:
        print ("guess is the topic go to end")
        return "end"
    
    return "continue"



# add conditional edges for host, player and call_tool
workflow.add_conditional_edges(
    "host", 
    router, 
    {"continue": "player", "call_tool": "call_tool", "end": END}
)

workflow.add_conditional_edges(
    "player", 
    router, 
    {"continue": "host", "call_tool": "call_tool", "end": END}
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "host": "host",
        "player": "player",
    },
)

# Set the entry point
workflow.add_edge(START, "host")

# Compile the graph
app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="agent.png")

# Run the graph
events = app.stream(
    {
        "messages": [
            SystemMessage(
                content="Let's play a game of 20 questions"
            )
        ],
        "topic": "",
        "question_asked": 0,
        "question_answered": 0,
        "guess": "",
    },
    {"recursion_limit": 200},
    stream_mode="updates"
)


for s in events:
    for node, values in s.items():
        print (f"node: {node}")

