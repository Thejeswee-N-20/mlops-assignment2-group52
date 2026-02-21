"""
Smoke test for deployed API.
Checks if health endpoint is reachable.
"""

import requests

def test_health_endpoint():
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200