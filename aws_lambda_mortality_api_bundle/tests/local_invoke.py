from __future__ import annotations

import json
import os
import sys

# Ensure the project root is on sys.path so 'app' and 'common' are importable
# when the script is run as `python tests/local_invoke.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.handler import lambda_handler


def main() -> None:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("MODEL_PATH", os.path.join(_root, "model_artifacts", "fusion_bundle.joblib"))
    os.environ.setdefault("API_KEY", "dev-key")

    with open(os.path.join(_root, "sample_request.json"), "r", encoding="utf-8") as f:
        payload = json.load(f)

    event = {
        "version": "2.0",
        "rawPath": "/predict",
        "headers": {
            "content-type": "application/json",
            "x-api-key": os.environ["API_KEY"],
        },
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps(payload),
        "isBase64Encoded": False,
    }

    response = lambda_handler(event, None)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
