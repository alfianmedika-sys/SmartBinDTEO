import time
import argparse
import requests

# 1-pixel transparent/blank minimal JPEG to test image uploads without cv2/external image dependencies
DUMMY_JPEG = (
    b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06'
    b'\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a'
    b'\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11'
    b'\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02'
    b'\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9'
)

def print_result(endpoint, response):
    status_color = "\033[92m" if response.status_code in [200, 201] else "\033[91m"
    reset_color = "\033[0m"
    print(f"\n--- Testing Endpoint: {endpoint} ---")
    print(f"Status Code: {status_color}{response.status_code}{reset_color}")
    try:
        print("Response JSON:")
        import json
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print("Response Text:")
        print(response.text)

def test_sensor_data_get(base_url, api_key, location):
    url = f"{base_url}/api/sensor-data"
    params = {
        "lokasi": location,
        "tipe": "organik",
        "berat": 1250,
        "volume": 75,
        "device": "ESP32-TEST-GET",
        "ts": time.time()
    }
    if api_key:
        params["api_key"] = api_key
       
    print(f"Sending GET to {url} with params {params}...")
    response = requests.get(url, params=params, timeout=10)
    print_result("GET /api/sensor-data", response)

def test_sensor_data_post(base_url, api_key, location):
    url = f"{base_url}/api/sensor-data"
    payload = {
        "location": location,
        "bin_type": "anorganik",
        "weight": 850,
        "volume": 42.5,
        "device_id": "ESP32-TEST-POST",
        "device_timestamp": time.time()
    }
    headers = {}
    if api_key:
        headers["X-Sensor-Key"] = api_key

    print(f"Sending JSON POST to {url} with payload {payload}...")
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print_result("POST /api/sensor-data", response)

def test_waste_detection(base_url, api_key, location, with_image=True):
    url = f"{base_url}/api/waste-detection"
    payload = {
        "waste_type": "b3",
        "confidence": 0.94,
        "location": location,
        "device_id": "RPI-TEST-CAM",
        "device_timestamp": time.time()
    }
    headers = {}
    if api_key:
        headers["X-Sensor-Key"] = api_key

    if with_image:
        print(f"Sending Multipart POST to {url} with 1-pixel test image...")
        files = {
            "image": ("test_detection.jpg", DUMMY_JPEG, "image/jpeg")
        }
        response = requests.post(url, data=payload, files=files, headers=headers, timeout=10)
    else:
        print(f"Sending JSON POST to {url} without image...")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
       
    print_result(f"POST /api/waste-detection (with_image={with_image})", response)

def test_fusion_log(base_url, api_key, location):
    url = f"{base_url}/api/fusion-log"
    payload = {
        "location": location,
        "bin_type": "Organic",
        "yolo_class": "Organic",
        "yolo_confidence": 0.88,
        "volume_status": "Sedang",
        "weight_status": "Rendah",
        "fuzzy_crisp": 45.2,
        "fuzzy_status": "Normal",
        "servo_action": "close",
        "buzzer_active": False,
        "device_id": "RPI-TEST-FUSION",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    }
    headers = {}
    if api_key:
        headers["X-Sensor-Key"] = api_key

    print(f"Sending JSON POST to {url} with fusion log {payload}...")
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print_result("POST /api/fusion-log", response)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Test Smart Waste Management endpoints.")
    parser.add_argument("--url", default="https://web-production-23834.up.railway.app", help="Base URL of the Laravel app (default: https://web-production-23834.up.railway.app)")
    parser.add_argument("--key", default="smart-waste-test-key", help="API Key for sensor.apikey middleware (default: smart-waste-test-key)")
    parser.add_argument("--location", default="BB102", help="Test location name (default: BB102)")
   
    args = parser.parse_args()
   
    # Strip trailing slash from URL if present
    base_url = args.url.rstrip("/")
   
    print(f"Targeting server: {base_url}")
    print(f"Using API key: {args.key if args.key else 'None (unauthenticated mode)'}")
    print(f"Using location: {args.location}")
   
    try:
        # 1. GET /api/sensor-data
        test_sensor_data_get(base_url, args.key, args.location)
       
        # 2. POST /api/sensor-data
        test_sensor_data_post(base_url, args.key, args.location)
       
        # 3. POST /api/waste-detection (JSON, no image)
        test_waste_detection(base_url, args.key, args.location, with_image=False)
       
        # 4. POST /api/waste-detection (Multipart, with image)
        test_waste_detection(base_url, args.key, args.location, with_image=True)
       
        # 5. POST /api/fusion-log
        test_fusion_log(base_url, args.key, args.location)
       
    except requests.exceptions.ConnectionError:
        print(f"\n\033[91mConnection Error:\033[0m Could not connect to the server at {base_url}.")
        print("Please check that the server is running. You can start it locally with:")
        print("  php artisan serve")
        print("Or target another URL using the --url argument:")
        print("  python test_send.py --url https://your-production-url.com")