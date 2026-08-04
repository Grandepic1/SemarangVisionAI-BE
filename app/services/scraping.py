import requests
import json
import re

url = "https://pantausemar.semarangkota.go.id/?cctv_category_id=fc3ed271-787c-4191-a7dd-fc84314a9f71"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

response = requests.get(url, headers=headers)

match = re.search(
    r"var\s+cctvs\s*=\s*(\[.*?\])\s*;",
    response.text,
    re.DOTALL,
)

if not match:
    raise Exception("Could not find CCTV data")


cctvs = json.loads(match.group(1))

results = []

for cctv in cctvs:
    for link in cctv.get("links", []):
        results.append(
            {
                "cctv_id": cctv["cctv_id"],
                "location_name": cctv["owner_name"],
                "latitude": float(cctv["lat"]),
                "longitude": float(cctv["lng"]),
                "camera_id": link["id"],
                "camera_name": link["name"],
                "camera_owner": link["owner_name"],
                "stream_url": link["url"],
                "status_code": link["status"],
                "status": "ACTIVE" if link["status"] == 1 else "OFFLINE",
                "is_preview": bool(link["is_preview"]),
            }
        )

with open("cctvs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} CCTV streams to cctvs.json")