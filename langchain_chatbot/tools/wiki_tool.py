from langchain.tools import tool
import wikipedia

@tool
def wiki_tool(query: str) -> str:
    """Search Wikipedia and return the summary of a topic."""
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Error: {str(e)}"
