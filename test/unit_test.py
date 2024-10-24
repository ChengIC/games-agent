import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.tools import generate_topic, answer_question, generate_question, make_guess
from utils.node import host_agent, player_agent
from utils.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

class Test(unittest.TestCase):

    def test_host_agent_is_generating_topic(self):
        state = AgentState(messages=[HumanMessage(content="Let's play a game of 20 questions")])
        response = host_agent.invoke(state)
        self.assertEqual(response.tool_calls[0]['name'], 'generate_topic')

    def test_host_agent_is_answering_question(self):
        state = AgentState(messages=[HumanMessage(content="Let's play a game of 20 questions"), 
                                    AIMessage(content="I have a secret topic for you to guess. Let's start the game."),
                                    HumanMessage(content="Is it a cat?")], topic="dog")
        response = host_agent.invoke(state)
        self.assertEqual(response.tool_calls[0]['name'], 'answer_question')
        
        
    def test_player_agent_is_generating_question(self):
        state = AgentState(messages=[HumanMessage(content="I have a secret topic for you to guess. Let's start the game."),
                                     AIMessage(content="I have a secret topic for you to guess. Let's start the game.")])
        response = player_agent.invoke(state)
        self.assertEqual(response.tool_calls[0]['name'], 'generate_question')

    def test_player_agent_is_making_guess(self):
        state = AgentState(messages=[HumanMessage(content="I have a secret topic for you to guess. Let's start the game."),
                                     AIMessage(content="I have a secret topic for you to guess. Let's start the game."),
                                     HumanMessage(content="Is it a cat?"),
                                     AIMessage(content="Yes")])
        response = player_agent.invoke(state)
        self.assertEqual(response.tool_calls[0]['name'], 'make_guess')

if __name__ == "__main__":
    unittest.main()