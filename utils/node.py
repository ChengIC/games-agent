from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Literal
import functools


class GameAgentNode:
    def __init__(self, llm, tools, 
                 system_prompt: str, 
                 role: Literal["host", "player"], 
                 logger=False):
        
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools
        self.role = role
        self.logger = logger

    def create_agent(self):
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
        if self.role == "host":
            prompt_settings.append(MessagesPlaceholder(variable_name="topic"))
            prompt_settings.append(MessagesPlaceholder(variable_name="task_for_host"))
            prompt_settings.append(MessagesPlaceholder(variable_name="guess"))
            
        prompt = ChatPromptTemplate.from_messages(prompt_settings)
        prompt = prompt.partial(role=self.role)
        prompt = prompt.partial(system_message=self.system_prompt)
        prompt = prompt.partial(tool_names=self.tools)

        return prompt | self.llm.bind_tools(self.tools, tool_choice="required")

    def handle_tool_message(self, tool_message, state):

        def handle_generate_topic(result, tool_message):
            result["topic"] = tool_message.content
            result["task_for_host"] = "answer_question"

            # Deliberate not including the topic in the messages
            result["messages"] = [AIMessage(content="I have a secret topic for you to guess. Let's start the game.", 
                                            name=self.role)]
            return result
    
        def handle_answer_question(result, state):
            result["num_questions_answered"] = state["num_questions_answered"] + 1
            return result
        
        def handle_make_guess(result, tool_message):
            result["guess"] = tool_message.content
            result["task_for_host"] = "check_guess"
            return result

        def handle_generate_question(result, state, tool_message):
            result["num_questions_asked"] = state["num_questions_asked"] + 1
            result["task_for_host"] = "answer_question"
            result["most_recent_question"] = tool_message.content
            return result

        def handle_check_guess(result):
            # no action needed for check_guess
            return result
        
        if self.logger:
            self.logger.log(f"call tools: {tool_message.name} with tool content: {tool_message.content}")

        result = {
            "messages": [AIMessage(content=tool_message.content, name=self.role)],
            "sender": self.role,
        }

        tool_handlers = {
            "generate_topic": lambda: handle_generate_topic(result, tool_message),
            "answer_question": lambda: handle_answer_question(result, state),
            "make_guess": lambda: handle_make_guess(result, tool_message),
            "generate_question": lambda: handle_generate_question(result, state, tool_message),
            "check_guess": lambda: handle_check_guess(result)
        }

        handler = tool_handlers.get(tool_message.name)
        if not handler:
            raise ValueError(f"Unknown tool: {tool_message.name}")
            
        return handler()

    def format_chat_history(self, messages):
        """ Filter out tool messages from a list of messages """
        filtered_messages = []
        for m in messages:
            if not isinstance(m, ToolMessage) and m.content != "":
                if m.name == self.role:
                    filtered_messages.append(("ai", m.content))
                else:
                    filtered_messages.append(("human", m.content))
        if self.logger:
            self.logger.log("get_chat_history messages:")
            for i, msg in enumerate(filtered_messages):
                self.logger.log(f"filtered_message {i}: {msg}")
        return filtered_messages
    
    def handle_regular_message(self, state):
        
        if self.role == "player":
            player_state = {"messages": self.format_chat_history(state["messages"])}
            result = self.agent.invoke(player_state)

        else:
            host_state = {
                "messages": self.format_chat_history(state["messages"]),
                "topic": [state.get("topic", "")],
                "task_for_host": [state.get("task_for_host", "")],
            }
            if state.get("guess") is not None:
                host_state["guess"] = [state.get("guess")]

            result = self.agent.invoke(host_state)

        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=self.role)

        if self.logger:
            self.logger.log(f"agent {self.role} returns: {result}")
        
        return {
            "messages": [result],
            "sender": self.role,
        }

    def call_agent(self, state):
        if self.logger:
            self.logger.log(
                f"call agent: {self.role} with input state topic: {state['topic']}, "
                f"num_questions_answered: {state['num_questions_answered']}, "
                f"num_questions_asked: {state['num_questions_asked']}, "
                f"task_for_host: {state['task_for_host']}, "
                f"guess: {state['guess']}"
            )
        
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            return self.handle_tool_message(last_message, state)
        else:
            return self.handle_regular_message(state)

    def create_node(self):
        self.agent = self.create_agent()
        return functools.partial(self.call_agent)
        
