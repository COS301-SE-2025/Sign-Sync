from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

@app.get("/")
async def root():
    return {"Awaiting landmark data"}

@app.post("/landmarks/")
async def getLandmarks(request: Request):
    landmarks = await request.json()
    return {"JSON recieved : ": landmarks}