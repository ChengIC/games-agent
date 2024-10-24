from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

host_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
player_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)