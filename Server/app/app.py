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

app = FastAPI()



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


# connect to db
client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["DATABASE_URL"])
db = client.dataSet
Movie_anime_collection = db.get_collection("Movie_anime")
TV_anime_collection = db.get_collection("TV_anime")


@app.on_event("startup")
async def startup_event():
    tv_anime_data = await TV_anime_collection.find().to_list(None)  # Adjust query as needed
    all_anime_ids = [anime['anime_id'] for anime in tv_anime_data]

    if 3086 not in all_anime_ids:
        print("Anime ID 3086 is missing from the dataset fetched during startup.")
        
    tv_anime_encoder.fit(all_anime_ids)


# endpoint to retrieve tv_anime info from database via id
@app.get("/anime_id/tv/{anime_id}", tags=["anime"])
async def get_TV_anime_by_id(anime_id: int):
    anime = await TV_anime_collection.find_one({"anime_id": anime_id})
    if anime:
        return anime
    raise HTTPException(status_code=404, detail="Anime TV not found")

# endpoint to retrieve movie_anime info from database via id
@app.get("/anime_id/movie/{anime_id}", tags=["anime"])
async def get_movie_anime_by_id(anime_id: int):
    anime = await Movie_anime_collection.find_one({"anime_id": anime_id})
    if anime is not None:
        return anime
    raise HTTPException(status_code=404, detail="Anime Movie not found")


# endpoint to retrieve tv_anime info from database via name (case insensitive)
@app.get("/anime_name/tv/{anime_name}", tags=["anime"])
async def get_TV_anime_by_name(anime_name: str):
    anime = await TV_anime_collection.find_one({"Name": {"$regex": f"^{anime_name}$", "$options": "i"}})
    if anime:
        return anime
    raise HTTPException(status_code=404, detail="Anime TV not found")

# endpoint to retrieve movie_anime info from database via name (case insensitive)
@app.get("/anime_name/movie/{anime_name}", tags=["anime"])
async def get_movie_anime_by_name(anime_name: str):
    anime = await Movie_anime_collection.find_one({"Name": {"$regex": f"^{anime_name}$", "$options": "i"}})
    if anime is not None:
        return anime
    raise HTTPException(status_code=404, detail="Anime Movie not found")


# endpoint to get similar tv_anime
@app.get("/similar-animes-tv/{anime_name}")
async def find_similar_animes_tv(anime_name: str, n: int = 10, return_dist: bool = False, neg: bool = False):
    try:
        anime_row = await get_TV_anime_by_name(anime_name)
        index = anime_row['anime_id']
        print(f"Encoding ID: {index}, Type: {type(index)}")
        encoded_index = tv_anime_encoder.transform([index])[0]
        weights = tv_anime_weights
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n = n + 1
        closest = sorted_dists[:n] if neg else sorted_dists[-n:]

        similarity_arr = []
        for close in closest:
            decoded_id = tv_anime_encoder.inverse_transform([close])[0]
            anime_frame = await get_TV_anime_by_id(decoded_id)
            if anime_frame.empty:
                continue
            
            anime_name = anime_frame['Name'].values[0]
            english_name = anime_frame['English name'].values[0]
            name = english_name if english_name != "UNKNOWN" else anime_name
            genre = anime_frame['Genres'].values[0]
            synopsis = anime_frame['Synopsis'].values[0]
            similarity = dists[close]
            similarity = "{:.2f}%".format(similarity * 100)

            similarity_arr.append({"Name": name, "Similarity": similarity, "Genres": genre, "Synopsis": synopsis})

        similarity_arr = sorted(similarity_arr, key=lambda x: x['Similarity'], reverse=True)
        return [anime for anime in similarity_arr if anime['Name'].lower() != anime_name.lower()]
    except Exception as e:
        raise HTTPException(status_code=404, detail=f'{anime_name} not found in Anime list. Error: {str(e)}')
    
    