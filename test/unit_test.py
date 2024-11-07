import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.node import GameAgentNode
from utils.state import AgentState
from utils.llm import host_llm, player_llm
from utils.tools import host_tools, player_tools

from langchain_core.messages import HumanMessage, AIMessage
import yaml

system_prompt = yaml.safe_load(open("system_prompts.yaml"))
host_system_prompt = system_prompt["host"]
player_system_prompt = system_prompt["player"]

host_node = GameAgentNode(llm=host_llm, 
                         tools=host_tools, 
                         role="host", 
                        system_prompt=host_system_prompt).create_node()

player_node = GameAgentNode(llm=player_llm, 
                            tools=player_tools, 
                            role="player", 
                            system_prompt=player_system_prompt).create_node()

class Test(unittest.TestCase):

            
    def test_host_agent_is_generating_topic(self):
        state = AgentState(messages=[HumanMessage(content="Let's play a game of 20 questions")],
                           guess="",
                           task_for_host="generate_topic", 
                           topic="")


        response = host_node(state)
        self.assertEqual(response['messages'][0].tool_calls[0]['name'], 'generate_topic')

    def test_host_agent_is_answering_question(self):

        state = AgentState(messages=[HumanMessage(content="Let's play a game of 20 questions"), 
                                    AIMessage(content="I have a secret topic for you to guess. Let's start the game."),
                                    HumanMessage(content="Is it a cat?")], 
                           guess="",
                           task_for_host="answer_question", 
                           topic="dog") 
        
        response = host_node(state)
        self.assertEqual(response['messages'][0].tool_calls[0]['name'], 'answer_question') 
    
    def test_host_agent_is_checking_guess(self):
        state = AgentState(messages=[HumanMessage(content="Let's play a game of 20 questions"), 
                                    AIMessage(content="I have a secret topic for you to guess. Let's start the game."),
                                    HumanMessage(content="Is it a cat?"),
                                    AIMessage(content="Yes"),
                                    HumanMessage(content="Kitten")],
                           guess="kitten",
                           topic="kitten",
                           task_for_host="check_guess")
        
        response = host_node(state)
        self.assertEqual(response['messages'][0].tool_calls[0]['name'], 'check_guess')
        
    def test_player_agent_is_generating_question(self):
        state = AgentState(messages=[HumanMessage(content="I have a secret topic for you to guess. Let's start the game."),
                                     AIMessage(content="Is it a dog?"),
                                     HumanMessage(content="No")])
        
        response = player_node(state)
        self.assertEqual(response['messages'][0].tool_calls[0]['name'], 'generate_question')


if __name__ == "__main__":
    unittest.main()