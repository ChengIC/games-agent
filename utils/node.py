from langchain_core.messages import ToolMessage, AIMessage
import functools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.llm import host_llm, player_llm
from utils.tools import host_tools, player_tools
import logging

def create_agent(llm, tools, system_prompt, role):
    """ Create an agent with a given llm, tools, and system prompt """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are part of a game of 20 questions. Your role is {role}. "
                "You MUST ALWAYS use the provided tools for EVERY action. "
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
    return prompt | llm.bind_tools(tools)

from langgraph.prebuilt import create_react_agent

host_system_prompt = """You are the AI host of a game of 20 questions. You will be given the conversation history between you (AI) and the player.
                        You are tasked with either generating a topic for the game (ONLY at the start of the game) or answering questions from the player. 

                        You MUST ALWAYS use the provided tools for every action:
                        1. Use the 'generate_topic' tool to create a new topic ONLY at the start of the game and if you don't have a topic yet.
                        If you have already generated a topic, do not generate a new one.
                        2. Use the 'answer_question' tool to respond to player questions with ONLY 'YES' or 'NO'.
                        
                        When the player try to make a guess he should only return the name of the guessing topic without any additional text or questions.
                        Please don NOT generate a new topic when you already have one.
                        After using these, you should gives the exact same response as the tool output.
                        Do not provide any explanations or additional text beyond the tool output. """

player_system_prompt = """You are the AI player of a game of 20 questions. You will be given the conversation history between you (AI) and the host.
                        You should either ask questions or make guesses.
                        
                        You MUST ALWAYS use the provided tools for every action:
                        1. Use the 'generate_question' tool to ask yes-or-no questions about the topic.
                        2. Use the 'make_guess' tool when you're ready to guess the topic. 
                        You should expect only return the name of the guessing topic without any additional text or questions.

                        Do not ask questions or make guesses directly. Always use the appropriate tool.
                        After using these, you should gives the exact same response as the tool output.
                        Avoid repeating questions you've already asked."""

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
    log_filename = "log.txt"
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