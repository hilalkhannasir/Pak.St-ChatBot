from flask import Flask, request, jsonify,send_from_directory
import os
from data_retrieval import get_answer_from_query

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello Render!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # local testing only