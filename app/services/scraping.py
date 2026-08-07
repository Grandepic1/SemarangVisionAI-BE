"""Scrape the public CCTV list and keep data/cctvs.json up to date.

Can be run standalone (`python -m app.services.scraping`) or called
programmatically (e.g. by the daily APScheduler job in app.core.scheduler).
"""

import json
import re
from pathlib import Path

import requests

URL = "https://pantausemar.semarangkota.go.id/?cctv_category_id=fc3ed271-787c-4191-a7dd-fc84314a9f71"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Runtime data lives under data/ (not the repo root).
DATA_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "cctvs.json"


def fetch_raw_cctvs() -> list[dict]:
    """Fetch and parse the raw CCTV list from the Semarang portal."""
    response = requests.get(URL, headers=HEADERS, timeout=30)
    response.raise_for_status()

    match = re.search(
        r"var\s+cctvs\s*=\s*(\[.*?\])\s*;",
        response.text,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Could not find CCTV data on the page")

    return json.loads(match.group(1))


def normalize_cctvs(raw_cctvs: list[dict]) -> list[dict]:
    """Flatten the raw CCTV groups into a list of camera entries."""
    results = []
    for cctv in raw_cctvs:
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
    return results


def load_existing(path: Path = DATA_FILE) -> list[dict]:
    """Load the previously stored CCTV entries; returns [] if the file is missing."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _key(entry: dict) -> tuple:
    return (entry["cctv_id"], entry["camera_id"])


def diff_cctvs(old: list[dict], new: list[dict]) -> dict:
    """Compare two CCTV lists by (cctv_id, camera_id) key.

    Returns {"added": [...], "removed": [...], "changed": [...]} where
    "changed" holds entries whose other fields (e.g. status, stream_url)
    differ between the two lists.
    """
    old_map = {_key(e): e for e in old}
    new_map = {_key(e): e for e in new}

    return {
        "added": [e for k, e in new_map.items() if k not in old_map],
        "removed": [e for k, e in old_map.items() if k not in new_map],
        "changed": [
            new_map[k] for k in new_map.keys() & old_map.keys() if new_map[k] != old_map[k]
        ],
    }


def save_cctvs(results: list[dict], path: Path = DATA_FILE) -> None:
    """Write the CCTV entries to the JSON file (creating the parent dir)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_scrape(path: Path = DATA_FILE) -> dict:
    """Scrape the CCTV list and rewrite the file only when something changed.

    Returns a summary dict, e.g.:
    {"total": 61, "changes": {"added": 0, "removed": 0, "changed": 2}, "updated": True}
    """
    results = normalize_cctvs(fetch_raw_cctvs())
    existing = load_existing(path)
    changes = diff_cctvs(existing, results)

    summary = {
        "total": len(results),
        "changes": {
            "added": len(changes["added"]),
            "removed": len(changes["removed"]),
            "changed": len(changes["changed"]),
        },
        "updated": any(changes.values()),
    }

    if summary["updated"]:
        save_cctvs(results, path)

    return summary


if __name__ == "__main__":
    print(run_scrape())
