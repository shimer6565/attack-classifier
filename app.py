from flask import Flask, request, jsonify, render_template
import json
from gemini import generate_context
from summarize_text import summarize_alert
from RAG_fusion import alert_retriever


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_alert", methods=["POST"])
def process_alert():
    data = request.data.decode("utf-8")
    print(data)

    alert_context = summarize_alert(data)
    similar_alerts = alert_retriever(alert_context)
    result = generate_context(similar_alerts, alert_context)
    print(result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
