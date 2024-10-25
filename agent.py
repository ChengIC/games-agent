from utils.node import host_node, player_node
from langgraph.graph import StateGraph, START,END
from utils.state import AgentState
from utils.tools import tool_node
from langchain_core.messages import SystemMessage
from utils.logger import experiment_logger

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("host", host_node)
workflow.add_node("player", player_node)
workflow.add_node("call_tool", tool_node)


def correct_tool_call(state):
    if len(state["messages"][-1].tool_calls) > 1:
        state["messages"][-1].tool_calls = [state["messages"][-1].tool_calls[0]]
        experiment_logger.log(f"multiple tool calls: {state['messages'][-1].tool_calls}")
        
    last_tool_call = state["messages"][-1].tool_calls[0]
    task_for_host = state["task_for_host"]
    # fix the tool call if the host called the wrong tool for "answer_question" and "check_guess"
    if last_tool_call["name"] == "check_guess" and \
            state["task_for_host"] == "answer_question":
        
        state["messages"][-1].tool_calls[0]["name"] = state["task_for_host"]
        state["messages"][-1].tool_calls[0]["args"] = {
            "topic": state["topic"],
            "question": state["most_recent_question"],
            "task_for_host": state["task_for_host"],
        }
        experiment_logger.log(f"fixed tool call: {state['messages'][-1].tool_calls[0]}")
    
    # fix the tool call if the host uses the wrong argument for "check_guess"
    elif last_tool_call["name"] == "check_guess" and \
            state["task_for_host"] == "check_guess":
        if last_tool_call["args"]["topic"] != state["topic"] or \
                last_tool_call["args"]["guess"] != state["guess"]:
            
            state["messages"][-1].tool_calls[0]["args"]["topic"] = state["topic"]
            state["messages"][-1].tool_calls[0]["args"]["guess"] = state["guess"]

            experiment_logger.log(f"fixed tool call args from {last_tool_call['args']} to {state['messages'][-1].tool_calls[0]['args']}")



def router(state):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        experiment_logger.log(f"tool call: {last_message.tool_calls}")
        correct_tool_call(state)
        return "call_tool"
    
    if state["question_asked"] > 20 or state["question_answered"] > 20:
        experiment_logger.log("question asked and answered 20 go to end")
        return "end"
    
    if state["guess"] == state["topic"]:
        experiment_logger.log("guess is the topic go to end")
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
        "task_for_host": "generate_topic",
        "most_recent_question": "",
    },
    {"recursion_limit": 200},
    stream_mode="updates"
)

for s in events:
    experiment_logger.log("********************************************************")
    for node, values in s.items():
        experiment_logger.log(f"update node: {node} and update: {values}")
        print(f"update node: {node} ")