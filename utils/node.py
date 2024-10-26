from langchain_core.messages import ToolMessage, AIMessage
import functools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.llm import host_llm, player_llm
from utils.tools import host_tools, player_tools


def create_agent(llm, tools, system_prompt, role):
    """ Create an agent with a given llm, tools, and system prompt """
    prompt_settings = [
            (
                "system",
                "You are part of a game of 20 questions. Your role is {role}. "
                "You MUST ALWAYS use the provided tools for EVERY action no matter what."
                "Available tools: {tool_names}.\n"
                "Role-specific instructions: {system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    if role == "host":
        prompt_settings.append(MessagesPlaceholder(variable_name="topic"))
        prompt_settings.append(MessagesPlaceholder(variable_name="task_for_host"))
        prompt_settings.append(MessagesPlaceholder(variable_name="guess"))
        
    prompt = ChatPromptTemplate.from_messages(prompt_settings)
    prompt = prompt.partial(role=role)
    prompt = prompt.partial(system_message=system_prompt)
    prompt = prompt.partial(tool_names=tools)

    return prompt | llm.bind_tools(tools, tool_choice="required")


host_system_prompt = """You are the AI host in a 20 questions game. Your tasks are to generate a topic, answer player questions, 
                        and check if the player's guess is correct. You will be given a {task_for_host}. ALWAYS use the specified tool for the given task.
                        If the {guess} is None, you should not use the check_guess tool.

                        Tasks and corresponding tools:
                        1. "generate_topic": Use the 'generate_topic' tool to create a topic at the beginning of the game if none exists.
                        2. "answer_question": Use the 'answer_question' tool to respond 'YES' or 'NO' to the most recent question from the player.
                        3. "check_guess": Use the 'check_guess' tool to check if the player's guess is correct.

                        IMPORTANT:
                        - ALWAYS use the tool specified by the {task_for_host}.
                        - NEVER change the task or use a different tool than specified.
                        - Provide responses EXACTLY as the tool outputs, with NO additional explanations.

                        Conversation format:
                        - ('human', 'player message'), ('ai', 'your response'), ...
                        - You are 'ai', the player is 'human'.

                        Guidelines:
                        - For "answer_question": The player's question will be a YES-or-NO type.
                        - For "check_guess": The player's guess will be a declarative statement, not a question.
                        - If the game starts with "I have a secret topic for you to guess. Let's start the game.", do not generate a new topic.

                        Remember: Your role is to facilitate the game, not to provide additional information or explanations beyond the tool outputs.
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


def format_chat_history(messages, current_role, logger=False):
    """ Filter out tool messages from a list of messages """
    filtered_messages = []
    for m in messages:
        if not isinstance(m, ToolMessage) and m.content != "":
            if m.name == current_role:
                filtered_messages.append(("ai", m.content))
            else:
                filtered_messages.append(("human", m.content))
    if logger:
        logger.log("get_chat_history messages:")
        for i, msg in enumerate(filtered_messages):
            logger.log(f"filtered_message {i}: {msg}")
    return filtered_messages


# Helper function to create a node for a given agent
def agent_node(state, agent, name, logger=False):
    if logger:
        logger.log(
            f"call agent: {name} with input state topic: {state['topic']}, "
            f"num_questions_answered: {state['num_questions_answered']}, "
            f"num_questions_asked: {state['num_questions_asked']}, "
            f"task_for_host: {state['task_for_host']}, "
            f"guess: {state['guess']}"
    )

    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        if logger:
            logger.log(f"call tools: {last_message.name} with tool content: {last_message.content}")

        if last_message.name == "generate_topic":
            return {
                "messages": [AIMessage(content="I have a secret topic for you to guess. Let's start the game.", name=name)],
                "sender": "host",
                "topic": last_message.content,
                "task_for_host": "answer_question",
            }
        
        elif last_message.name == "answer_question":
            return {
                "messages": [AIMessage(content=last_message.content, name=name)],
                "sender": "host",
                "num_questions_answered": state["num_questions_answered"] + 1,
            }

        elif last_message.name == "make_guess":
            return {
                "messages": [AIMessage(content=last_message.content, name=name)],
                "sender": "player",
                "guess": last_message.content,
                "task_for_host": "check_guess",
            }
        
        elif last_message.name == "generate_question":
            return {
                "messages": [AIMessage(content=last_message.content, name=name)],
                "sender": "player",
                "num_questions_asked": state["num_questions_asked"] + 1,
                "task_for_host": "answer_question",
                "most_recent_question": last_message.content,
            }
        
        elif last_message.name == "check_guess":
            return {
                "messages": [AIMessage(content=last_message.content, name=name)],
                "sender": "host",
            }
        else:
            raise ValueError(f"Unknown tool: {last_message.name}")
        
    else:
        result = ""
        if name == "player":
            result = agent.invoke({"messages": format_chat_history(state["messages"], name, logger)})
        else:
            host_state = {"messages": format_chat_history(state["messages"], name, logger),
                          "topic": [state.get("topic", "")],
                          "task_for_host": [state.get("task_for_host", "")],}
            
            if state.get("guess") is not None:
                host_state["guess"] = [state.get("guess")]

            result = agent.invoke(host_state)

        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        if logger:
            logger.log(f"agent {name} returns: {result}")
        return {
            "messages": [result],
            "sender": name,
        }




