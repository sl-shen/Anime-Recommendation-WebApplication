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
from urllib.parse import unquote
from pydantic import BaseModel
import httpx 

load_dotenv()

app = FastAPI()



origins = [
    "http://localhost:5173",
    "localhost:5173",
    "http://localhost",
    "http://localhost:80",
    "http://localhost:8080",
    "http://192.168.1.240:7979",
    "http://192.168.1.240:8000",
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
    tv_anime_encoder.fit(all_anime_ids)

    movie_anime_data = await Movie_anime_collection.find().to_list(None)  # Adjust query as needed
    all_movie_anime_ids = [anime['anime_id'] for anime in movie_anime_data]
    movie_anime_encoder.fit(all_movie_anime_ids)


# endpoint to retrieve tv_anime info from database via id
@app.get("/anime_id/tv/{anime_id}", tags=["tv_anime"])
async def get_TV_anime_by_id(anime_id: int):
    anime = await TV_anime_collection.find_one({"anime_id": anime_id})
    if anime:
        return anime
    else:
        return None

# endpoint to retrieve movie_anime info from database via id
@app.get("/anime_id/movie/{anime_id}", tags=["movie_anime"])
async def get_movie_anime_by_id(anime_id: int):
    anime = await Movie_anime_collection.find_one({"anime_id": anime_id})
    if anime:
        return anime
    else:
        return None

# endpoint to retrieve tv_anime info from database via name (case insensitive)
@app.get("/anime_name/tv/{anime_name}", tags=["tv_anime"])
async def get_TV_anime_by_name(anime_name: str):
    anime = await TV_anime_collection.find_one({"Name": {"$regex": f"^{anime_name}$", "$options": "i"}})
    if anime:
        anime['anime_id'] = int(anime['anime_id'])
        return anime
    else:
        return None

# endpoint to retrieve tv_anime info from database via English name (case insensitive)
@app.get("/anime_eng_name/tv/{anime_eng_name}", tags=["tv_anime"])
async def get_TV_anime_by_eng_name(anime_eng_name: str):
    anime = await TV_anime_collection.find_one({"English name": {"$regex": f"^{anime_eng_name}$", "$options": "i"}})
    if anime:
        anime['anime_id'] = int(anime['anime_id'])
        return anime
    else:
        return None


# endpoint to retrieve movie_anime info from database via name (case insensitive)
@app.get("/anime_name/movie/{anime_name}", tags=["movie_anime"])
async def get_movie_anime_by_name(anime_name: str):
    anime = await Movie_anime_collection.find_one({"Name": {"$regex": f"^{anime_name}$", "$options": "i"}})
    if anime:
        anime['anime_id'] = int(anime['anime_id'])
        return anime
    else:
        return None

# endpoint to retrieve movie_anime info from database via English name (case insensitive)
@app.get("/anime_eng_name/movie/{anime_eng_name}", tags=["movie_anime"])
async def get_movie_anime_by_eng_name(anime_eng_name: str):
    anime = await Movie_anime_collection.find_one({"English name": {"$regex": f"^{anime_eng_name}$", "$options": "i"}})
    if anime:
        anime['anime_id'] = int(anime['anime_id'])
        return anime
    else:
        return None

# autocomplete endpoint for tv anime
@app.get("/autocomplete_tv")
async def autocomplete_tv(term: str):
    animes = await TV_anime_collection.find(
        {"$or": [
            {"Name": {"$regex": term, "$options": "i"}},
            {"English name": {"$regex": term, "$options": "i"}}
        ]}
    ).to_list(length=10)

    names = [anime["Name"] for anime in animes]
    eng_names = [anime["English name"] for anime in animes if "English name" in anime]

    return list(set(names + eng_names))

# autocomplete endpoint for movie anime
@app.get("/autocomplete_movie")
async def autocomplete_movie(term: str):
    animes = await Movie_anime_collection.find(
        {"$or": [
            {"Name": {"$regex": term, "$options": "i"}},
            {"English name": {"$regex": term, "$options": "i"}}
        ]}
    ).to_list(length=10)

    names = [anime["Name"] for anime in animes]
    eng_names = [anime["English name"] for anime in animes if "English name" in anime]

    return list(set(names + eng_names))


# endpoint to get similar tv_anime
@app.get("/similar-animes-tv/{anime_name}")
async def find_similar_animes_tv(anime_name: str, n: int = 10, return_dist: bool = False, neg: bool = False):
    try:
        anime_row = await get_TV_anime_by_name(anime_name)
        if anime_row is None:
            anime_row = await get_TV_anime_by_eng_name(anime_name)

        index = int(anime_row['anime_id'])
        print(f"Encoding ID: {index}, Type: {type(index)}")
        encoded_index = tv_anime_encoder.transform([index])[0]
        weights = tv_anime_weights
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n = n + 1
        closest = sorted_dists[:n] if neg else sorted_dists[-n:]

        similarity_arr = []
        for close in closest:
            decoded_id = int(tv_anime_encoder.inverse_transform([close])[0])
            anime_frame = await get_TV_anime_by_id(decoded_id)
            if anime_frame is None:
                continue
            
            anime_name = anime_frame['Name']
            english_name = anime_frame.get('English name', 'UNKNOWN')
            name = english_name if english_name != "UNKNOWN" else anime_name
            genre = anime_frame.get('Genres', '')
            synopsis = anime_frame.get('Synopsis', '')
            similarity = dists[close]
            similarity = "{:.2f}%".format(similarity * 100)

            similarity_arr.append({"Name": name, "Similarity": similarity, "Genres": genre, "Synopsis": synopsis})

        similarity_arr = sorted(similarity_arr, key=lambda x: x['Similarity'], reverse=True)
        
        return [anime for anime in similarity_arr if anime['Name'].lower() != anime_name.lower()]
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f'{anime_name} not found in Anime list. Error: {str(e)}')

# endpoint to get similar movie_anime
@app.get("/similar-animes-movie/{anime_name}")
async def find_similar_animes_movie(anime_name: str, n: int = 10, return_dist: bool = False, neg: bool = False):
    try:
        anime_row = await get_movie_anime_by_name(anime_name)
        if anime_row is None:
            anime_row = await get_movie_anime_by_eng_name(anime_name)

        index = int(anime_row['anime_id'])
        print(f"Encoding ID: {index}, Type: {type(index)}")
        encoded_index = movie_anime_encoder.transform([index])[0]
        weights = movie_anime_weights
        dists = np.dot(weights, weights[encoded_index])
        sorted_dists = np.argsort(dists)
        n = n + 1
        closest = sorted_dists[:n] if neg else sorted_dists[-n:]

        similarity_arr = []
        for close in closest:
            decoded_id = int(movie_anime_encoder.inverse_transform([close])[0])
            anime_frame = await get_movie_anime_by_id(decoded_id)
            if anime_frame is None:
                continue
            
            anime_name = anime_frame['Name']
            english_name = anime_frame.get('English name', 'UNKNOWN')
            name = english_name if english_name != "UNKNOWN" else anime_name
            genre = anime_frame.get('Genres', '')
            synopsis = anime_frame.get('Synopsis', '')
            similarity = dists[close]
            similarity = "{:.2f}%".format(similarity * 100)

            similarity_arr.append({"Name": name, "Similarity": similarity, "Genres": genre, "Synopsis": synopsis})

        similarity_arr = sorted(similarity_arr, key=lambda x: x['Similarity'], reverse=True)
        return [anime for anime in similarity_arr if anime['Name'].lower() != anime_name.lower()]
    except Exception as e:
        raise HTTPException(status_code=404, detail=f'{anime_name} not found in Anime list. Error: {str(e)}')
    


from pydantic import BaseModel
from typing import Optional, List, Dict

class User(BaseModel):
    #hash: str
    #similarity: int
    username: str

class SimilarUsersResponse(BaseModel):
    data: list[User]

CLIENT_ID = os.getenv("MAL_CLIENT_ID")

# get_similar_users help func via api
async def similar_users(username: str, page: int = 1):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.reko.moe/{username}/similar", params={"page": page})
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Error fetching similar users")

# help func to fetch data
async def fetch_anime_page(url: str, headers: Dict[str, str], params: Dict[str, str]) -> Dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
    
# get user's anime_list via api
async def user_anime_list(username: str, status: Optional[str] = None) -> List[Dict]:
    url = f"https://api.myanimelist.net/v2/users/{username}/animelist"
    params = {
        "fields": "list_status",
        "limit": 1000 
    }
    if status:
        params["status"] = status
    
    headers = {
        "X-MAL-CLIENT-ID": CLIENT_ID
    }

    all_data = []
    # use loop since mal uses paging for data
    while url:
        page_data = await fetch_anime_page(url, headers, params)
        all_data.extend(page_data['data'])
        url = page_data['paging'].get('next')
        params = {}

    return all_data


# endpoint to return similar user
@app.get("/similar-users/{username}")
async def get_similar_users(username: str, page: int = 1):
    try:
        result = await similar_users(username, page)
        similar_users_response = SimilarUsersResponse(**result)
        return similar_users_response.data[0] if similar_users_response.data else None
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# endpoint to return user's anime list
@app.get("/anime-list")
async def anime_list(username: str, status: Optional[str] = None):
    try:
        result = await user_anime_list(username, status)
        return {"data": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



from statistics import mean

# 
async def get_anime_recommendations(username: str) -> Dict:
    try:
        # 获取输入用户的已完成动漫列表
        user_anime = await user_anime_list(username, status="completed")
        
        # 计算用户的动漫平均评分
        user_scores = [anime['list_status']['score'] for anime in user_anime if anime['list_status']['score'] > 0]
        if not user_scores:
            return {"message": f"用户 {username} 的动漫列表没有评分。"}
        threshold = mean(user_scores)
        
        # 获取用户的已放弃动漫列表
        user_dropped = await user_anime_list(username, status="dropped")
        user_dropped_ids = set(anime['node']['id'] for anime in user_dropped)
        
         # 获取最相似用户
        similar_user = await get_similar_users(username)
        if not similar_user:
            return {"message": "没有找到相似用户。"}
        
        # 检查 similar_user 的类型并相应地获取用户名
        if isinstance(similar_user, dict):
            similar_username = similar_user.get('username')
        elif hasattr(similar_user, 'username'):
            similar_username = similar_user.username
        else:
            similar_username = str(similar_user)

        if not similar_username:
            return {"message": "无法获取相似用户的用户名。"}

        
        # 获取相似用户的已完成动漫列表
        similar_user_anime = await user_anime_list(similar_username, status="completed")
        if not similar_user_anime:
            return {"message": f"相似用户 {similar_username} 的动漫列表为空。"}
        
        # 过滤掉共同观看的和用户已放弃的动漫
        user_watched_ids = set(anime['node']['id'] for anime in user_anime)
        filtered_anime = [
            anime for anime in similar_user_anime 
            if anime['node']['id'] not in user_watched_ids 
            and anime['node']['id'] not in user_dropped_ids 
            and anime['list_status']['score'] > threshold
        ]
        
        # 按评分排序并选择前5个
        sorted_anime = sorted(filtered_anime, key=lambda x: x['list_status']['score'], reverse=True)
        #top_5_anime = sorted_anime[:5]
        
        #if not top_5_anime:
        # return all possible anime recommendations
        if not sorted_anime:
            return {"message": "没有可推荐的动漫。"}
        
        recommendations = [
            {
                "title": anime['node']['title'],
                "score": anime['list_status']['score']
            } for anime in sorted_anime
        ]
        
        return {
            "similar_user": similar_username,
            "recommendations": recommendations
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# endpoint to return recommendations
@app.get("/anime-recommendations/{username}")
async def anime_recommendations(username: str):
    return await get_anime_recommendations(username)