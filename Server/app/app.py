from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import motor.motor_asyncio
import os
from fastapi import HTTPException
from bson.objectid import ObjectId
import pydantic
pydantic.json.ENCODERS_BY_TYPE[ObjectId]=str
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# endpoint to retrieve anime info from database
@app.get("/anime/{anime_id}", tags=["anime"])
async def get_anime(anime_id: int):
    anime = await Movie_anime_collection.find_one({"anime_id": anime_id}, {'_id': 0})
    if anime is not None:
        return anime
    anime = await TV_anime_collection.find_one({"anime_id": anime_id})
    if anime:
        return anime
    raise HTTPException(status_code=404, detail="Anime not found")



# define model structure
class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_animes, embedding_size):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.anime_embedding = nn.Embedding(num_animes, embedding_size)
        self.dense = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.activation = nn.Sigmoid()  # or nn.ReLU()

    def forward(self, user_input, anime_input):
        user_embedded = self.user_embedding(user_input)
        anime_embedded = self.anime_embedding(anime_input)
        dot_product = torch.mul(user_embedded, anime_embedded).sum(dim=-1)
        flattened = self.flatten(dot_product.unsqueeze(-1))
        activated = self.activation(flattened)
        dense_output = self.dense(activated)
        output = self.sigmoid(dense_output)
        return output
    
# loaded model for tv anime
model_TV = CollaborativeFilteringModel(num_users=268584, num_animes=12321, embedding_size=128)
model_TV.load_state_dict(torch.load('../models/tv_model_weights.pth', map_location=torch.device('cpu')))
model_TV.eval()
# loaded model for movie anime
model_Movie = CollaborativeFilteringModel(num_users=183380, num_animes=2685, embedding_size=128)
model_Movie.load_state_dict(torch.load('../models/movie_model_weights.pth', map_location=torch.device('cpu')))
model_Movie.eval()

# function to extract weight
def extract_weights(layer):
    # Get the weights from the PyTorch layer
    weights = layer.weight.data
    
    # Convert to numpy for normalization
    weights = weights.numpy()
    
    # Normalize the weights
    weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
    
    return weights

# endpoint to inform model loaded
app.on_event("startup")
async def load_model():
    print("Models loaded and ready to use.")

# load weights from models
tv_anime_weights = extract_weights(model_TV.anime_embedding)
tv_user_weights = extract_weights(model_TV.user_embedding)
movie_anime_weights = extract_weights(model_Movie.anime_embedding)
movie_user_weights = extract_weights(model_Movie.user_embedding)

# load encoder
tv_user_encoder = LabelEncoder()
tv_anime_encoder = LabelEncoder()
movie_user_encoder = LabelEncoder()
movie_anime_encoder = LabelEncoder()
