from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import motor.motor_asyncio
import os
from fastapi import HTTPException
from bson.objectid import ObjectId
import pydantic
pydantic.json.ENCODERS_BY_TYPE[ObjectId]=str

load_dotenv()

# connect to db
app = FastAPI()
client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["DATABASE_URL"])
db = client.dataSet
Movie_anime_collection = db.get_collection("Movie_anime")
TV_anime_collection = db.get_collection("TV_anime")

origins = [
    "http://localhost:3000",
    "localhost:3000"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to your todo list."}


@app.get("/anime/{anime_id}", tags=["anime"])
async def get_anime(anime_id: int):
    anime = await Movie_anime_collection.find_one({"anime_id": anime_id}, {'_id': 0})
    if anime is not None:
        return anime
    anime = await TV_anime_collection.find_one({"anime_id": anime_id})
    if anime:
        return anime
    raise HTTPException(status_code=404, detail="Anime not found")
