from langchain_core.prompts import PromptTemplate
from utils.llm import player_llm, host_llm
import random
import csv
import os
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def generate_question(messages):
    """For player to generate a YES-or-NO type question to ask the host to help guess the topic. 
    It takes the conversation history to help formulate the question."""
    
    PROMPT = """You are a player of 20 questions game. The host has generated a topic for the game. 
            Your task is to ask a YES-or-NO type question that will help you guess the topic.
            Please only return ONE question, not any additional text. 

            Instructions:
            1. You can only ask a YES-or-NO type question that will help you narrow down the topic.
            2. Only return the question, not any additional text.
            3. You should observe the conversation history as {messages} to formulate your question. Please do not ask the same question twice! 
            For example, if the chat history contains "Is it a living thing?", you should not ask the same question again. """

    query_prompt = PromptTemplate.from_template(PROMPT)
    chain = query_prompt | player_llm
    response = chain.invoke({"messages": messages})
    return response.content

@tool
def make_guess(messages):
    """For the player to make a guess of the topic if the player feels confident about the guessing topic.
    It takes the conversation history to help make the guess."""

    PROMPT = """You are a player of 20 questions game. The host has generated a secret topic for the game. 
            Your task is to make a guess of the topic based on the answer from the host.
            You should observe the conversation history as {messages} and the answer from the host to make a guess.

            Instructions:
            1. Make a guess of the topic based on the conversation history and the answer from the host.
            2. Please only return the name of the topic you guess, not any additional text. For example, you think the topic is "apple", you should return "apple". 
            Don't return "I think the topic is apple" or 'Is the topic apple?' or anything like that.
            3. You should not make the same guess twice! If you found you make a guess in the past, you should not make the same guess again.
            """
    
    query_prompt = PromptTemplate.from_template(PROMPT)
    chain = query_prompt | player_llm
    response = chain.invoke({"messages": messages})
    return response.content


all_reference_topics = []
with open(os.path.join('data', 'reference_things.csv'), 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        all_reference_topics.append(row[1])

@tool
def generate_topic(task_for_host: str):
    """For the host to come up with a topic for the player to guess only when the topic is not generated yet. 
    If the task_for_host is not "generate_topic", you should not use this tool."""

    if task_for_host != "generate_topic":
        raise ValueError("This tool should only be used when the task is to generate a topic.")
    
    PROMPT = """Generate a unique and commonly recognized name for a game of 20 questions. 
            The topic should be a single object or living thing from any of the following categories: 
            animals, plants, places, daily-life items, or famous individuals or characters from movies, TV shows, books, or history. 
            Please provide just one name randomly selected from these categories, without any additional text or explanation.

            You can also reference or be inspired by the following list of topics to help you generate a topic:
            {sample_reference_topics}"""

    query_prompt = PromptTemplate.from_template(PROMPT)
    sample_reference_topics = random.sample(all_reference_topics, 5)
    chain = query_prompt | host_llm
    response = chain.invoke({"sample_reference_topics": sample_reference_topics})
    return response.content

@tool
def answer_question(topic: str, question: str, task_for_host: str):
    """For the host to answer the YES-or-NO type question from the player.
    It takes the topic and the question to answer the question.
    If the task_for_host is not "answer_question", you should not use this tool."""

    if task_for_host != "answer_question":
        raise ValueError("This tool should only be used when the task is to answer a question.")
    
    PROMPT = """You are a host of 20 questions game. You already come up with a secret topic given as {topic}. 
            The player will ask a YES-or-NO type question to guess the topic. 
            The question is given as {question}.

            Your task is to simply answer with "YES" or "NO" regrading to the question in terms of the topic. 
            Please only return "YES" or "NO", not any additional text! 
            """
    
    query_prompt = PromptTemplate.from_template(PROMPT)
    chain = query_prompt | host_llm
    response = chain.invoke({"topic": topic, "question": question})
    return response.content

@tool
def check_guess(topic: str, guess: str, task_for_host: str):
    """For the host to check if the player's guess is correct. 
    Only use this tool if the player has made a guess in a declarative statement.
    If the task_for_host is not "check_guess", you should not use this tool."""

    if task_for_host != "check_guess":
        raise ValueError("This tool should only be used when the task is to check a guess.")

    if guess.lower() == topic.lower() or topic.lower() in guess.lower():
        return "Congratulations, you are right!"
    else:
        return "Sorry, you are wrong. Please ask another question."


host_tools = [
    generate_topic,
    answer_question,
    check_guess
]


player_tools = [
    generate_question,
    make_guess
]

tool_node = ToolNode(host_tools + player_tools)