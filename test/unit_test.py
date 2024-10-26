import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.node import host_agent, player_agent, format_chat_history
from utils.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

class Test(unittest.TestCase):

    def test_host_agent_is_generating_topic(self):
        state = AgentState(messages=[HumanMessage(content="Let's play a game of 20 questions")],
                           guess="",
                           task_for_host="generate_topic", 
                           topic="")
        host_state = {"messages": format_chat_history(state["messages"], "host", logging=False),
                      "guess": [state.get("guess")],
                      "task_for_host": [state.get("task_for_host")],
                      "topic": [state.get("topic")]}
        response = host_agent.invoke(host_state)
        self.assertEqual(response.tool_calls[0]['name'], 'generate_topic')

    def test_host_agent_is_answering_question(self):
        state = AgentState(messages=[HumanMessage(content="Let's play a game of 20 questions"), 
                                    AIMessage(content="I have a secret topic for you to guess. Let's start the game."),
                                    HumanMessage(content="Is it a cat?")], 
                           guess="",
                           task_for_host="answer_question", 
                           topic="dog") 
        
        host_state = {"messages": format_chat_history(state["messages"], "host", logging=False),
                      "guess": [state.get("guess")],
                      "task_for_host": [state.get("task_for_host")],
                      "topic": [state.get("topic")]}
        
        response = host_agent.invoke(host_state)
        self.assertEqual(response.tool_calls[0]['name'], 'answer_question')        
        
    def test_player_agent_is_generating_question(self):
        state = AgentState(messages=[HumanMessage(content="I have a secret topic for you to guess. Let's start the game."),
                                     AIMessage(content="Is it a dog?"),
                                     HumanMessage(content="No")])
        
        player_state = {"messages": format_chat_history(state["messages"], "player", logging=False),
                        "guess": [state.get("guess")],
                        "task_for_player": [state.get("task_for_player")]}
        
        response = player_agent.invoke(player_state)
        self.assertEqual(response.tool_calls[0]['name'], 'generate_question')


if __name__ == "__main__":
    unittest.main()