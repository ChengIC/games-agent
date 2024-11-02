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
        self.max_questions = 20
        self.dialogs = []
        self.updated_nodes = []

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
            self.logger.log(f"multiple tool calls: {state['messages'][-1].tool_calls}")
            state["messages"][-1].tool_calls = [state["messages"][-1].tool_calls[0]]

        # fix host tool call
        last_tool_call = copy.deepcopy(state["messages"][-1].tool_calls[0])
        task_for_host = state["task_for_host"]

        # fix the host's tool call if the host called the wrong tool for "answer_question" and "check_guess"
        if last_tool_call["name"] == "check_guess" and \
                task_for_host == "answer_question":
            
            state["messages"][-1].tool_calls[0]["name"] = task_for_host
            state["messages"][-1].tool_calls[0]["args"] = {
                "topic": state["topic"],
                "question": state["most_recent_question"],
                "task_for_host": task_for_host,
            }
            self.logger.log(f"fixed tool call: {state['messages'][-1].tool_calls[0]}")
        
        # fix the host's tool call if the host uses the wrong argument for "check_guess"
        elif last_tool_call["name"] == "check_guess" and \
                task_for_host == "check_guess":
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
        if state["num_questions_asked"] >= self.max_questions and state["num_questions_answered"] >= self.max_questions:
            self.logger.log("="*100)
            self.logger.log(f"Questions asked and answered: {self.max_questions} and game ends for topic: {state['topic']}")
            return "end"

        # if there is a tool call, correct the tool call with the accurate tool name and arguments before calling the tool
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            self.logger.log(f"tool call: {last_message.tool_calls}")
            self._correct_tool_call(state)
            return "call_tool"
        
        # if the player's guess matches the topic, go to end
        if state["guess"].lower() == state["topic"].lower():
            self.logger.log("="*100)
            self.logger.log(f"Guess matches topic: {state['topic']} and game ends")
            return "end"
        
        return "continue"

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
        for event in events:
            self.logger.log("*"*100) 
            for node, values in event.items():
                self.logger.log(f"update node: {node} and update: {values}")
                self.updated_nodes.append(node)

                # simply print the player's and host's messages for demo
                if node == "player" or node == "host":
                    if len(values["messages"][-1].content) > 0:
                        print (f"{node}: {values['messages'][-1].content}")
                        self.dialogs.append(f"{node}: {values['messages'][-1].content}")
            
            self.logger.validate_updated_nodes(self.updated_nodes)

        self.logger.log("="*100)
        for dialog in self.dialogs:
            self.logger.log(dialog)


if __name__ == "__main__":
    import uuid
    game_id  = uuid.uuid4()
    game = Game(game_id)
    game.run()