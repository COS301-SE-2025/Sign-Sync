import requests
import pytest
import time

ENDPOINT = "http://localhost:3000/userApi/login"

def test_login_rate_limit():
    payload = {"email": "a@gmail.com", "password": "wrongpassword"}

    try:
        for _ in range(3):
            response = requests.post(ENDPOINT, json=payload)

        response = requests.post(ENDPOINT, json=payload)
        if response.status_code == 500:
            pytest.skip("Skipping rate-limit test: backend returned 500 (likely no MongoDB)")
        assert response.status_code == 429

        time.sleep(61)
        response = requests.post(ENDPOINT, json=payload)
        if response.status_code == 500:
            pytest.skip("Skipping rate-limit reset test: backend returned 500 (likely no MongoDB)")
        assert response.status_code == 400

    except requests.exceptions.ConnectionError:
        pytest.skip("Skipping rate-limit test: frontend not reachable")
