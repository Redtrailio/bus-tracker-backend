from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

# In-memory storage for bus location
bus_location = {
    "lat": None,
    "lng": None,
    "timestamp": None
}


@app.route("/")
def home():
    return render_template("index.html")


# API: Update bus location
@app.route("/update-location", methods=["POST"])
def update_location():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data received"}), 400

    if "lat" not in data or "lng" not in data:
        return jsonify({"error": "Invalid GPS data"}), 400

    try:
        lat = float(data["lat"])
        lng = float(data["lng"])
    except ValueError:
        return jsonify({"error": "Latitude and Longitude must be numbers"}), 400

    bus_location["lat"] = lat
    bus_location["lng"] = lng
    bus_location["timestamp"] = time.time()

    return jsonify({
        "status": "success",
        "message": "Location updated",
        "data": bus_location
    })


# API: Get bus location
@app.route("/bus-location", methods=["GET"])
def get_bus_location():
    if bus_location["lat"] is None:
        return jsonify({
            "status": "error",
            "message": "Bus location not available yet"
        }), 404

    return jsonify({
        "status": "success",
        "data": bus_location
    })


# API: Health check (Render monitoring)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
