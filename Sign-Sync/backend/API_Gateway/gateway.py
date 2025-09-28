import os
from pathlib import Path

import asyncio
import httpx
import websockets
import time

from typing import Optional, Tuple, List
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

CORS_ORIGINS = "http://localhost:3000"
#CORS_ORIGINS = ["https://signsyncportal-a3gyaxb7dhdde4ef.southafricanorth-01.azurewebsites.net"]
WS_ORIGINS = "http://localhost:3000"
#WS_ORIGINS = ["https://signsyncportal-a3gyaxb7dhdde4ef.southafricanorth-01.azurewebsites.net"]

DEFAULT_TIMEOUT = 60.0

SPEECH_URL   = "http://localhost:8003"
ASL_URL      = "http://localhost:8002"
TTS_URL      = "http://localhost:8001"
AUTH_URL     = "http://localhost:8004"
ALPHABET_URL = "http://localhost:8000"
WORD_URL     = "http://localhost:8005"
STT_URL = "http://localhost:8006"
GESTURE_URL = "http://localhost:8008"

#Docker-version:

# SPEECH_URL   = "http://speech-to-text:8003"
# ASL_URL      = "http://text-to-asl-gloss:8002"
# TTS_URL      = "http://localhost:8001"
# AUTH_URL     = "http://localhost:8004"
# ALPHABET_URL = "http://alphabet-translate:8000"
# WORD_URL     = "http://word-prediction:8005"
# STT_URL = "http://sign-to-text:8006"
# GESTURE_URL = "http://gesture-recognition:8008"

#Azure-version:

# SPEECH_URL   = "https://speechtotext-e4fkhwe7hwf6hgh4.southafricanorth-01.azurewebsites.net/"
# ASL_URL      = "https://texttoaslgloss-h3buaxf6g0fqe6hk.southafricanorth-01.azurewebsites.net/"
# TTS_URL      = "http://localhost:8001"
# AUTH_URL     = "http://localhost:8004"
# ALPHABET_URL = "https://alphabettranslate-cdarg5fyhraja3gu.southafricanorth-01.azurewebsites.net/"
# WORD_URL     = "https://wordprediction-d6eke0emc6d5c6eq.southafricanorth-01.azurewebsites.net/"
# STT_URL = "https://signtotext-dnh8gbcqegfve2h9.southafricanorth-01.azurewebsites.net/"
# GESTURE_URL = "https://gestrurerecognition-fmg3grgxawaagzhe.southafricanorth-01.azurewebsites.net/"

class Route(BaseModel):
    prefix: str                 
    backend: str                
    strip_prefix: bool = True   
    upstream_prefix: str = ""   
    timeout: float = DEFAULT_TIMEOUT

routes: List[Route] = []


if AUTH_URL:
    routes.append(Route(
        prefix="/api/auth",
        backend=AUTH_URL,
        strip_prefix=True,
        upstream_prefix=""
    ))

if SPEECH_URL:
    routes.append(Route(
        prefix="/api/speech",
        backend=SPEECH_URL,
        strip_prefix=True,
        upstream_prefix=""
    ))

if ASL_URL:
    routes.append(Route(
        prefix="/api/asl",
        backend=ASL_URL,
        strip_prefix=True,
        upstream_prefix=""
    ))

if ALPHABET_URL:
    routes.append(Route(
        prefix="/api/alphabet",
        backend=ALPHABET_URL,
        strip_prefix=True,
        upstream_prefix=""
    ))

if WORD_URL:
    routes.append(Route(
        prefix="/api/word",
        backend=WORD_URL,
        strip_prefix=True,
        upstream_prefix=""
    ))

if TTS_URL:
    routes.append(Route(
        prefix="/api/sign",
        backend=TTS_URL,
        strip_prefix=True,
        upstream_prefix=""
    ))
if STT_URL:
    routes.append(Route(
        prefix="/api/stt",
        backend=STT_URL,
        strip_prefix=True,
        upstream_prefix=""
    ))

if GESTURE_URL:
    routes.append(Route(
        prefix="/api/gesture",
        backend=GESTURE_URL,
        strip_prefix=True,
        upstream_prefix=""
    ))

print([(r.prefix, r.backend) for r in routes])

app = FastAPI()

# Configure the CORS_ORIGINS at deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rate_limits = {}
MAX_REQUESTS = 40
WINDOW = 60

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    
    path = request.url.path.rstrip("/")
    print(path)
    # this won't allow any other path to bypass the limiter
    if path == "/api/alphabet/predict" or path == "/api/gesture/predict":
        return await call_next(request)
    
    client_ip = request.client.host
    now = time.time()

    if client_ip not in rate_limits:
        rate_limits[client_ip] = []

    requests = [t for t in rate_limits[client_ip] if now - t < WINDOW]
    requests.append(now)
    rate_limits[client_ip] = requests

    print(f"IP={client_ip}, total={len(requests)}, times={requests}")

    if len(requests) > MAX_REQUESTS:
        # return JSONResponse(
        #     {"detail": "Too many requests"},
        #     status_code=429,
        #     headers={"Retry-After": str(WINDOW),
        #              "Access-Control-Allow-Origin": CORS_ORIGINS}
        # )

        # raise HTTPException(
        #     status_code = status.HTTP_429_TOO_MANY_REQUESTS,
        #     detail = "Too many requests"
        # )

        return JSONResponse(
            {"detail": "Too many requests"},
            status_code=429,
            headers={
                "Retry-After": str(WINDOW),
                "Access-Control-Allow-Origin": CORS_ORIGINS[0],
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS"
            }
        )
    
    response = await call_next(request)
    return response

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "routes": [f"{r.prefix} -> {r.backend}{r.upstream_prefix}" for r in routes]
    }

def match_route(path: str) -> Tuple[Optional[Route], Optional[str]]:
    print(path)
    candidates = [r for r in routes if path.startswith(r.prefix)]
    if not candidates:
        return None, None

    
    route = max(candidates, key=lambda r: len(r.prefix))
    
    rest = path[len(route.prefix):] if route.strip_prefix else path

    
    if rest and not rest.startswith("/"):
        rest = "/" + rest

    
    upstream_path = (route.upstream_prefix.rstrip("/") + rest) if route.upstream_prefix else (rest or "/")
    if not upstream_path.startswith("/"):
        upstream_path = "/" + upstream_path

    return route, upstream_path



async def proxy(req: Request, route: Route, upstream_path: str) -> Response:
    url = route.backend.rstrip("/") + upstream_path
    if req.url.query:
        url += f"?{req.url.query}"

    
    drop = {"host", "content-length", "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}
    headers = {k: v for k, v in req.headers.items() if k.lower() not in drop}

    body = await req.body()
    timeout = httpx.Timeout(route.timeout)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            upstream = await client.request(req.method, url, headers=headers, content=body)
        except httpx.ReadTimeout:
            return JSONResponse({"error": "Request timed out"}, status_code=504)
        except httpx.ConnectError:
            return JSONResponse({"error": "Connection error"}, status_code=502)

    resp_drop = {"content-encoding", "transfer-encoding", "connection"}
    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in resp_drop}

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("Content-Type"),
    )


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def gateway(full_path: str, req: Request) -> Response:
    print(full_path)
    route, upstream_path = match_route("/" + full_path)
    print(route)
    if not route:
        raise HTTPException(status_code=404, detail="Not Found")
    return await proxy(req, route, upstream_path)



def _http_to_ws(url: str) -> str:
    if url.startswith(("ws://", "wss://")):
        return url
    if url.startswith("https://"):
        return "wss://" + url[len("https://"):]
    if url.startswith("http://"):
        return "ws://" + url[len("http://"):]
    
    return "ws://" + url.lstrip("/")

@app.websocket("/{full_path:path}")
async def ws_gateway(websocket: WebSocket, full_path: str):
    origin = (websocket.headers.get("origin") or "").strip()
    if WS_ORIGINS and origin and origin not in WS_ORIGINS:
        await websocket.close(code=4403)  
        return

    path = "/" + full_path
    route, upstream_path = match_route(path)
    if not route:
        await websocket.close(code=4404)  
        return

    
    qs = str(websocket.query_params)
    base_ws = _http_to_ws(route.backend.rstrip("/"))
    upstream_url = base_ws + upstream_path + (f"?{qs}" if qs else "")

    
    extra_headers = {}
    auth = websocket.headers.get("authorization")
    if auth:
        extra_headers["Authorization"] = auth

    
    raw_subs = websocket.headers.get("sec-websocket-protocol", "")
    subprotocols = [s.strip() for s in raw_subs.split(",") if s.strip()] or None

    
    try:
        upstream = await websockets.connect(
            upstream_url,
            extra_headers=extra_headers,
            subprotocols=subprotocols,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,  
        )
    except Exception:
        await websocket.close(code=1011)  
        return

    
    try:
        await websocket.accept(subprotocol=upstream.subprotocol)
    except Exception:
        await upstream.close()
        return

    async def client_to_upstream():
        try:
            while True:
                msg = await websocket.receive()
                typ = msg.get("type")
                if typ == "websocket.receive":
                    if msg.get("text") is not None:
                        await upstream.send(msg["text"])
                    elif msg.get("bytes") is not None:
                        await upstream.send(msg["bytes"])
                elif typ == "websocket.disconnect":
                    try:
                        await upstream.close()
                    finally:
                        break
        except Exception:
            try:
                await upstream.close()
            except:
                pass

    async def upstream_to_client():
        try:
            async for message in upstream:
                if isinstance(message, (bytes, bytearray)):
                    await websocket.send_bytes(message)
                else:
                    await websocket.send_text(message)
        except Exception:
            try:
                await websocket.close()
            except:
                pass

    await asyncio.gather(client_to_upstream(), upstream_to_client())
