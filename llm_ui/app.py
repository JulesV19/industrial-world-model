import json
import os

from flask import Flask, render_template, request, Response, stream_with_context

from agent import run_agent, reset_session

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    message    = data.get("message", "").strip()
    session_id = data.get("session_id")

    if not message:
        return {"error": "message vide"}, 400

    def generate():
        try:
            yield from run_agent(message, session_id)
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/reset", methods=["POST"])
def reset():
    data       = request.get_json()
    session_id = data.get("session_id")
    if session_id:
        reset_session(session_id)
    return {"ok": True}


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
