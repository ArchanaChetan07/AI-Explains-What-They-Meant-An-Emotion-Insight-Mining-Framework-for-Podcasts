import json
from datetime import datetime

def log_chat(question: str, response: str, filepath="chat_history.json"):
    """
    Log the conversation to a JSON file with timestamp.

    Parameters:
    - question: User's input string
    - response: Bot's output string
    - filepath: File to append log entries to
    """
    chat_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "response": response
    }

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    data.append(chat_entry)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
