
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

# === Step 2: Load API key ===
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# === Step 3: Load Tools ===
from .predict_topic import predict_topic
from .wiki_tool import wiki_tool
from .guest_search import list_all_guests

tools = [
    Tool.from_function(
        func=predict_topic,
        name="PredictTopic",
        description="Predict the topic of a podcast quote or segment."
    ),
    Tool.from_function(
        func=wiki_tool,
        name="WikiSearch",
        description="Search Wikipedia and return a brief summary of a topic."
    ),
    Tool.from_function(
        func=list_all_guests,  # ✅ Fixed here
        name="ListAllGuests",
        description="Return known information about a Lex Fridman podcast guest."
    )
]


# === Step 4: Setup LLM ===
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4",
    openai_api_key=openai_key
)

# === Step 5: Setup Memory and Agent ===
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# === Step 6: Wrapper Function ===
def ask_agent(prompt: str) -> str:
    """Send user prompt to LangChain agent and return response."""
    return agent.run(prompt)
