# app/langchain_chat.py
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Use environment variable or load securely
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

llm = ChatOpenAI(temperature=0.3, model_name="gpt-4")

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are an AI expert in Lex Fridman's podcast. Answer with clarity and emotional intelligence.\n\nQ: {question}\nA:"
)

chain = LLMChain(llm=llm, prompt=prompt)

def get_bot_response(question: str) -> str:
    return chain.run(question)
