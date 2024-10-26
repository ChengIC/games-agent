from utils.node import agent_node, host_agent, player_agent
from langgraph.graph import StateGraph, START,END
from utils.state import AgentState
from utils.tools import tool_node
from langchain_core.messages import SystemMessage
from utils.logger import ExperimentLogger
import functools
import copy

class Game:
    def __init__(self, game_id):
        self.logger = ExperimentLogger(game_id=game_id)
    
    def _create_app(self):
        # create agent node
        host_node = functools.partial(agent_node, agent=host_agent, name="host", logger=self.logger)
        player_node = functools.partial(agent_node, agent=player_agent, name="player", logger=self.logger)

        # Create the graph  
        workflow = StateGraph(AgentState)

       # Add nodes to the graph
        workflow.add_node("host", host_node)
        workflow.add_node("player", player_node)
        workflow.add_node("call_tool", tool_node)

        # add conditional edges for host, player and call_tool
        workflow.add_conditional_edges(
            "host", 
            self._router, 
            {"continue": "player", "call_tool": "call_tool", "end": END}
        )

        workflow.add_conditional_edges(
            "player", 
            self._router, 
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
        self.app = workflow.compile()
        self.app.get_graph().draw_mermaid_png(output_file_path="agent.png")

    def _correct_tool_call(self, state):
        """
        This function is used to call the correct tool with the correct arguments.

        -- arguments:
            state: the state of the agent

        """

        # if there are multiple tool calls, only keep the first one
        if len(state["messages"][-1].tool_calls) > 1:
            state["messages"][-1].tool_calls = [state["messages"][-1].tool_calls[0]]
            self.logger.log(f"multiple tool calls: {state['messages'][-1].tool_calls}")

        # fix host tool call
        last_tool_call = copy.deepcopy(state["messages"][-1].tool_calls[0])
        task_for_host = state["task_for_host"]

        # fix the host's tool call if the host called the wrong tool for "answer_question" and "check_guess"
        if last_tool_call["name"] == "check_guess" and \
                state["task_for_host"] == "answer_question":
            
            state["messages"][-1].tool_calls[0]["name"] = state["task_for_host"]
            state["messages"][-1].tool_calls[0]["args"] = {
                "topic": state["topic"],
                "question": state["most_recent_question"],
                "task_for_host": state["task_for_host"],
            }
            self.logger.log(f"fixed tool call: {state['messages'][-1].tool_calls[0]}")
        
        # fix the host's tool call if the host uses the wrong argument for "check_guess"
        elif last_tool_call["name"] == "check_guess" and \
                state["task_for_host"] == "check_guess":
            if last_tool_call["args"]["topic"] != state["topic"] or \
                    last_tool_call["args"]["guess"] != state["guess"]:
                
                state["messages"][-1].tool_calls[0]["args"]["topic"] = state["topic"]
                state["messages"][-1].tool_calls[0]["args"]["guess"] = state["guess"]
                self.logger.log(f"fixed tool call args from {last_tool_call['args']} to {state['messages'][-1].tool_calls[0]['args']} for check_guess")
        
        # fix the host's tool call if the host uses the wrong argument for "answer_question"
        elif last_tool_call["name"] == "answer_question":
            if last_tool_call["args"]["topic"] != state["topic"]:
                state["messages"][-1].tool_calls[0]["args"]["topic"] = state["topic"]
                self.logger.log(f"fixed tool call args from {last_tool_call['args']} to {state['messages'][-1].tool_calls[0]['args']} for answer_question")
        else:
            pass

    def _router(self, state):
        """
        The router function is used to route the state to the correct agent node and tool node.

        -- arguments:
            state: the state of the agent
        """

        # if the player has asked 20 questions and the host has answered 20 questions, go to end
        if state["num_questions_asked"] == 20 and state["num_questions_answered"] == 20:
            self.logger.log("question asked and answered 20 go to end")
            return "end"

        # if there is a tool call, correct the tool call with the accurate tool name and arguments before calling the tool
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            self.logger.log(f"tool call: {last_message.tool_calls}")
            self._correct_tool_call(state)
            
            return "call_tool"
        
        # if the player's guess matches the topic, go to end
        if state["guess"].lower() == state["topic"].lower():
            self.logger.log("guess is the topic go to end")
            return "end"
        
        return "continue"

    # Run the graph
    def run(self):
        self._create_app()
        events = self.app.stream(
        {
            "messages": [
                SystemMessage(
                    content="Let's play a game of 20 questions"
                )
            ],
            "topic": "",
            "num_questions_asked": 0,
            "num_questions_answered": 0,
            "guess": "",
            "task_for_host": "generate_topic",
            "most_recent_question": "",
        },
        {"recursion_limit": 200},
            stream_mode="updates"
        )
        for s in events:
            self.logger.log("********************************************************")
            for node, values in s.items():
                self.logger.log(f"update node: {node} and update: {values}")

                # simply print the player's and host's messages for demo
                if node == "player" or node == "host":
                    if len(values["messages"][-1].content) > 0:
                        print (f"{node}: {values['messages'][-1].content}")


if __name__ == "__main__":
    game = Game("test")
    game.run()