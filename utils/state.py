import operator
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    topic: str
    question_asked: int
    question_answered: int
    guess: str
    task_for_host: str
    most_recent_question: str