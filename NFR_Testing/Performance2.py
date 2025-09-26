import asyncio
import httpx
import websockets
import time
import statistics

WS_URL = "ws://localhost:8006"
START = "/v1/session/start"
STOP = "/v1/session/stop"
STREAM = "/v1/stream"

CONCURRENT_USERS = 25
MESSAGES_PER_USER = 5

PAYLOAD = {}

response_times = []

async def simulate_user(user_id: int):
    async with httpx.AsyncClient() as client:
        try:
            start_resp = await client.post(f"{WS_URL + START}")
            start_resp.raise_for_status()
            session_id = start_resp.json().get("session_id")
            if not session_id:
                print(f"[User {user_id}] Failed to get session_id")
                return
            
            print(f"[User {user_id}] Started session {session_id}")

            ws_url = WS_URL + STREAM + "/" + session_id
            async with websockets.connect(ws_url) as ws:
                for i in range(MESSAGES_PER_USER):
                    start = time.perf_counter()
                    await ws.send(PAYLOAD)
                    reply = await ws.recv()
                    elapsed = (time.perf_counter() - start) * 1000
                    response_times.append(elapsed)
            
            stop_resp = await client.post(f"{WS_URL + STOP}")
            stop_resp.raise_for_status()

        except Exception as e:
            print(f"[User {user_id}] ERROR: {e}")

async def main():
    tasks = [simulate_user(i) for i in range(CONCURRENT_USERS)]
    await asyncio.gather(*tasks)

    if response_times:
        print("\n=== RESULTS ===")
        print(f"Total messages: {len(response_times)}")
        print(f"Min: {min(response_times):.2f} ms")
        print(f"Max: {max(response_times):.2f} ms")
        print(f"Average: {statistics.mean(response_times):.2f} ms")
        print(f"Median: {statistics.median(response_times):.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())