# guest_tool.py
from langchain.tools import tool
import pandas as pd

# ✅ Corrected file path and method
df = pd.read_csv(r"C:\Users\archa\Desktop\Lex Project\notebooks\data\processed\lex_fridman_cleaned.csv")

@tool
def list_all_guests() -> str:
    """Returns a list of all guest names from the dataset."""
    guests = df['guest'].dropna().unique()
    return "\n".join(sorted(guests))
