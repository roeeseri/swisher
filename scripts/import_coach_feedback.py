"""Import returned coach labels through the running desktop app, without stale-file writes."""

from __future__ import annotations

import argparse
import http.client
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    payload = args.feedback.read_bytes()
    if len(payload) > 12 * 1024 * 1024:
        raise ValueError("Feedback file exceeds 12MB")
    json.loads(payload)
    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=60)
    connection.request("GET", "/")
    response = connection.getresponse()
    page = response.read().decode()
    match = re.search(r"const TOKEN='([^']+)'", page)
    if not match:
        raise RuntimeError("Open the local Swisher Studio first")
    connection.request(
        "POST",
        "/api/import-feedback",
        body=payload,
        headers={"Content-Type": "application/json", "X-Swisher-Token": match[1]},
    )
    response = connection.getresponse()
    result = json.loads(response.read())
    connection.close()
    if response.status != 200:
        raise ValueError(result.get("error", "Import failed"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
