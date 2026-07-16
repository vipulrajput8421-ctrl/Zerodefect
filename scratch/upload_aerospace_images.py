import requests
import json

url = 'http://localhost:8000/api/infer'

try:
    print("[*] Uploading synthetic normal aircraft skin image...")
    with open(r'd:\ZeroDefect\data\test\test_good_0001.jpg', 'rb') as f:
        res1 = requests.post(url, files={'file': f})
    print("Response:")
    print(json.dumps(res1.json(), indent=2))
except Exception as e:
    print("[ERROR] Good image upload failed:", e)

try:
    print("\n[*] Uploading real defective aircraft skin image...")
    # Find the first test defect image
    import os
    from pathlib import Path
    test_dir = Path(r'd:\ZeroDefect\data\test')
    defect_img = sorted(list(test_dir.glob("test_defect_*.jpg")))[0]
    print(f"Uploading file: {defect_img.name}")
    with open(defect_img, 'rb') as f:
        res2 = requests.post(url, files={'file': f})
    print("Response:")
    print(json.dumps(res2.json(), indent=2))
except Exception as e:
    print("[ERROR] Defect image upload failed:", e)
