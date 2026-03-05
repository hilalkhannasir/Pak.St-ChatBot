from flask import Flask, request, jsonify,send_from_directory
import os
from data_retrieval import get_answer_from_query

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory("static", "chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"error": "Empty message"}), 400

    try:
        answer, source = get_answer_from_query(user_query) 
        return jsonify({
            "answer": answer,
            "source": source
        })
    except Exception as e:  
        return jsonify({"error": str(e)}), 500

