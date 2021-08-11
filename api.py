from fastapi import FastAPI
from pydantic import BaseModel
from main import prediction


class Item(BaseModel):
    id: int


app = FastAPI()


@app.get("/")
async def index(item: Item):
    if item.id is not None:
        return prediction(item.id)
