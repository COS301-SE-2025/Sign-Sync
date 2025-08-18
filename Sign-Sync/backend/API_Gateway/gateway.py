import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).with_name(".env")) 

import httpx
from typing import Optional, Tuple, List
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# Frontend origins for CORS
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
DEFAULT_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15.0"))

# Extraction of URL's from env
#AUTH_URL   = os.getenv("AUTH_URL", "")     
SPEECH_URL = os.getenv("SPEECH_URL", "")  
ASL_URL    = os.getenv("ASL_URL", "")     
TTS_URL   = os.getenv("TTS_URL", "")    
AUTH_URL = os.getenv("AUTH_URL", "")
ALPHABET_URL = os.getenv("ALPHABET_URL", "")
WORD_URL = os.getenv("WORD_URL", "")

# Simple Route class for encapsulation of route information
class Route(BaseModel):
    prefix: str                 
    backend: str                
    strip_prefix: bool = True   
    upstream_prefix: str = ""   
    timeout: float = DEFAULT_TIMEOUT

routes: List[Route] = []
# Append routes if present in env
if AUTH_URL:
    routes.append(Route(
    prefix="/api/auth",
    backend=AUTH_URL,
    strip_prefix=True,
    upstream_prefix=""
    ))

if SPEECH_URL:
    routes.append(Route(
        prefix="/api/stt",
        backend=SPEECH_URL,
        strip_prefix=True,
        upstream_prefix="/api"
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
print([(r.prefix, r.backend) for r in routes])  

# create fastapi app
# It will currently only having routing
# Later security will be added (JWT tokens)
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# for debugging
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "routes": [f"{r.prefix} -> {r.backend}{r.upstream_prefix}" for r in routes]
    }

# Function to match request route to known routes in order to choose the correct microservice to forward the request to
# Returns:
# route (match for backend service)
# upstream_path (request path withing the backend service indicated by route)
def match_route(path: str) -> Tuple[Optional[Route], Optional[str]]:
    print(path)
    candidates = [r for r in routes if path.startswith(r.prefix)]

    if not candidates:
        return None, None
    
    route = max(candidates, key=lambda r: len(r.prefix)) # match longest prefix
    rest = path[len(route.prefix):] if route.strip_prefix else path # remove the prefix, since  paths within backend api's are relative

    if rest and not rest.startswith("/"):
        rest = "/"  + rest  # ensure leading slash
    
    upstream_path = (route.upstream_prefix.rstrip("/") + rest) if route.upstream_prefix else (rest or "/") # if some prefix is needed by the backend api add it here
    # if prefix is added, ensure it starts with a slash
    if not upstream_path.startswith("/"):
        upstream_path = "/" + upstream_path
    return route, upstream_path

# Proxy function to forward requests to the appropriate backend service
async def proxy(req: Request, route: Route, upstream_path: str) -> Response:
    url = route.backend.rstrip("/") + upstream_path

    if req.url.query:
        url += f"?{req.url.query}"

    # avoid forwarding hop-by-hop headers
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


# Gateway function to handle all incoming requests
@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def gateway(full_path: str, req: Request) -> Response:
    print(full_path)
    route, upstream_path = match_route("/" + full_path)
    print(route)
    # If no matched route, return 404
    if not route:
        raise HTTPException(status_code=404, detail="Not Found")
    # Otherwise, forward the request and return awaited response
    return await proxy(req, route, upstream_path)