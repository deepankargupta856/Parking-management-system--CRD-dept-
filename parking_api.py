# parking_api.py

from flask import Flask, jsonify
from flask_cors import CORS
from shared_state import slot_state

app = Flask(__name__)
CORS(app)  # ✅ THIS IS THE FIX

@app.route("/api/slots", methods=["GET"])
def get_slots():
    return jsonify(slot_state)

def start_api():
    print("[INFO] Parking API started on port 8000")
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
