import requests
import pytest

ENDPOINT = "http://localhost:3000/userApi/login"


def test_login_rate_limit():
    payload = {"email": "a@gmail.com", "password": "wrongpassword"}

    for i in range(3):
        response = requests.post(ENDPOINT, json=payload)

    response = requests.post(ENDPOINT, json=payload)
    assert response.status_code == 429

