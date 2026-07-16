import requests
import json

url = 'http://localhost:8000/api/infer'

try:
    print("[*] Uploading good bolt image...")
    with open(r'd:\ZeroDefect\data\test\test_good_0001.jpg', 'rb') as f:
        res1 = requests.post(url, files={'file': f})
    print("Response:")
    print(json.dumps(res1.json(), indent=2))
except Exception as e:
    print("[ERROR] Good bolt upload failed:", e)

try:
    print("\n[*] Uploading defective crack bolt image...")
    with open(r'd:\ZeroDefect\data\test\test_crack_0001.jpg', 'rb') as f:
        res2 = requests.post(url, files={'file': f})
    print("Response:")
    print(json.dumps(res2.json(), indent=2))
except Exception as e:
    print("[ERROR] Defective bolt upload failed:", e)
