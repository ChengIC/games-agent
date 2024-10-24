from langchain_core.messages import ToolMessage, AIMessage
import functools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.llm import host_llm, player_llm
from utils.tools import host_tools, player_tools
import logging
from datetime import datetime

def create_agent(llm, tools, system_prompt, role):
    """ Create an agent with a given llm, tools, and system prompt """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are part of a game of 20 questions. Your role is {role}. "
                "You MUST ALWAYS use the provided tools for EVERY action no matter what."
                "Available tools: {tool_names}.\n"
                "Role-specific instructions: {system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(role=role)
    prompt = prompt.partial(system_message=system_prompt)
    prompt = prompt.partial(tool_names=tools)
    if role == "host":
        prompt = prompt.partial(topic=MessagesPlaceholder(variable_name="topic"))
    return prompt | llm.bind_tools(tools, tool_choice="required")


host_system_prompt = """You are the AI host in a 20 questions game. Your tasks are to generate a topic or answer player questions.

                        The conversation history is formatted as follows:
                        - ('human', 'message 1'), ('ai', 'message 2'), ('human', 'message 3') ...
                        - You are the 'ai', and the player is the 'human'.

                        Tasks:
                        1. Use the 'generate_topic' tool to create a topic at the beginning of the game if none exists.
                        2. Use the 'answer_question' tool to respond 'YES' or 'NO' to the most recent question from the player.

                        Instructions:
                        - Do not generate a topic or answer the question without tools.
                        - Provide responses exactly as the tool outputs.
                        - NO extra explanations beyond tool outputs.

                        Note: If the game starts with "I have a secret topic for you to guess. Let's start the game.", do not generate a new topic.
                        """

player_system_prompt = """You are the AI player in a 20 questions game. 
                        Your task is to guess the secret topic by interacting with the host using the provided tools.
                        IMPORTANT: YOU MUST NEVER OUTPUT A GUESS DIRECTLY - ALWAYS USE THE make_guess TOOL!
                        
                        The conversation history is formatted as follows:
                        - ('human', 'message 1'), ('ai', 'message 2'), ('human', 'message 3') ...
                        - You are the 'ai', and the host is the 'human'.
                        
                        During the game, you have exactly two possible actions:
                        1. Use the 'generate_question' tool to ask a yes-or-no question
                        2. Use the 'make_guess' tool to submit your final guess
                        
                        STRICT RULES:
                        - NEVER write a guess in your response - you must use the make_guess tool
                        - If you think you know the answer, you MUST use the make_guess tool
                        - If you see the word "guess" or "answer" in your thought process, you MUST use the make_guess tool
                        - Any attempt to guess without the make_guess tool will be considered invalid
                        - Do not explain your reasoning - just use the appropriate tool
                        
                        Example correct behavior:
                        - When you want to guess "elephant" or something, use: make_guess tool and return the response from this tool 
                        ignoring your original guess. 
                        
                        Remember: EVERY guess MUST go through the make_guess tool - no exceptions!"""


host_agent = create_agent(host_llm, host_tools, host_system_prompt, "host")
player_agent = create_agent(player_llm, player_tools, player_system_prompt, "player")


def format_chat_history(messages, current_role):
    """ Filter out tool messages from a list of messages """
    logging.info(" get_chat_history messages:")
    filtered_messages = []
    for m in messages:
        if not isinstance(m, ToolMessage) and m.content != "":
            if m.name == current_role:
                filtered_messages.append(("ai", m.content))
            else:
                filtered_messages.append(("human", m.content))

    for i, msg in enumerate(filtered_messages):
        logging.info(f"filtered_message {i}: {msg}")
    return filtered_messages


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    # Set up logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/log_{timestamp}.txt"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')
    logging.info("********************************")
    logging.info(f"call agent name: {name} and state topic: {state['topic']}")
    logging.info(f"question_answered: {state['question_answered']} and question_asked: {state['question_asked']}")

    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        if last_message.name == "generate_topic":
            return {
                "messages": [AIMessage(content="I have a secret topic for you to guess. Let's start the game.", name=name)],
                "sender": "host",
                "topic": last_message.content,
            }
        
        if last_message.name == "answer_question":
            return {
                "messages": [AIMessage(content=last_message.content, name=name)],
                "sender": "host",
                "question_answered": state["question_answered"] + 1,
            }

        if last_message.name == "make_guess":
            return {
                "messages": [AIMessage(content=last_message.content, name=name)],
                "sender": "player",
                "guess": last_message.content,
            }
        
        if last_message.name == "generate_question":
            return {
                "messages": [AIMessage(content=last_message.content, name=name)],
                "sender": "player",
                "question_asked": state["question_asked"] + 1,
            }
    
    else:
        if name == "player":
            result = agent.invoke({"messages": format_chat_history(state["messages"], name)})
        else:
            result = agent.invoke({"messages": format_chat_history(state["messages"], name),
                                   "topic": state.get("topic", "")})

        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)

        return {
            "messages": [result],
            "sender": name,
        }



host_node = functools.partial(agent_node, agent=host_agent, name="host")
player_node = functools.partial(agent_node, agent=player_agent, name="player")