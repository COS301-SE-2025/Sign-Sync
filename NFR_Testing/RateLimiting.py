import asyncio, httpx, pytest

ENDPOINT = "http://localhost:8007/api/word/translate"

@pytest.mark.asyncio
async def test_rate_limit():
    payload = {"text": "STORE I GO"}
    async with httpx.AsyncClient() as client:
        tasks = [client.post(ENDPOINT, json=payload) for _ in range(41)]
        results = await asyncio.gather(*tasks)
        codes = [r.status_code for r in results]
        assert 429 in codes