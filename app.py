import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# REPLACE THESE with your College's actual GPS coordinates
COLLEGE_LAT = 12.9716 
COLLEGE_LNG = 77.5946
BUS_SPEED_KMH = 20  # Estimated average speed in city traffic

# Global variable to store the latest bus location
current_bus_location = {"lat": COLLEGE_LAT, "lng": COLLEGE_LNG, "eta": 0}

def calculate_eta(bus_lat, bus_lng):
    """Calculates minutes until arrival using Haversine formula."""
    R = 6371  # Earth radius in km
    dlat = math.radians(COLLEGE_LAT - bus_lat)
    dlng = math.radians(COLLEGE_LNG - bus_lng)
    
    a = (math.sin(dlat / 2)**2 + math.cos(math.radians(bus_lat)) * math.cos(math.radians(COLLEGE_LAT)) * math.sin(dlng / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    # Time = Distance / Speed
    hours = distance / BUS_SPEED_KMH
    return round(hours * 60)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_location', methods=['POST'])
def update_location():
    global current_bus_location
    data = request.json
    lat, lng = data['lat'], data['lng']
    
    eta = calculate_eta(lat, lng)
    current_bus_location = {"lat": lat, "lng": lng, "eta": eta}
    
    return jsonify({"status": "success", "eta": eta})

@app.route('/get_location', methods=['GET'])
def get_location():
    return jsonify(current_bus_location)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)