from fastapi import FastAPI
from fastapi.params import Form
from pydantic import BaseModel
from main import prediction

import uvicorn


app = FastAPI()


@app.get("/")
async def index(id: int = Form(...)):
    if id is not None:
        return prediction(id)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
