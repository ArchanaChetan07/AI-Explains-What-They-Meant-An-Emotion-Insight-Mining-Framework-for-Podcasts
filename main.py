from flask import Flask, request, jsonify
from langchain_chatbot.tools.langchain_service import ask_agent
from chat_logger import log_chat  # ✅ Add this

app = Flask(__name__)

@app.route('/')
def index():
    return "✅ LangChain Chatbot is running!"

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get("question", "")
    if not user_input:
        return jsonify({"error": "No question provided."}), 400

    response = ask_agent(user_input)

    log_chat(user_input, response)  # ✅ Log conversation here

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
