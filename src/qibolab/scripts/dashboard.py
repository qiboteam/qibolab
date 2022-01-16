#!/usr/bin/env python
"""
    Start dashboard server.
"""
import uvicorn
import pathlib
import json
import asyncio
from fastapi import FastAPI
from fastapi import Request
from fastapi import WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


app = FastAPI()
path = pathlib.Path(__file__).parent
app.mount("/static", StaticFiles(directory=path / "static"), name="static")
templates = Jinja2Templates(directory=path / "templates")

with open(path / "measurements.json", "r") as file:
    measurements = iter(json.loads(file.read()))


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await asyncio.sleep(0.1)
        payload = next(measurements)
        await websocket.send_json(payload)


def main():
    uvicorn.run(app)


if __name__ == "__main__":
    main()
