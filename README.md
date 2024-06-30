# Anime Recommender KKSK

This project is an Anime Recommendation System that utilizes a React frontend and a FastAPI backend to provide personalized anime recommendations based on user preferences and MyAnimeList data.

Use the following link to visit the site:
- [Anime Recommender](kksk.yukinolov.com)

## Tech Stack

- Frontend: React with TypeScript, built using Vite
- Backend: Python with FastAPI
- Database: MongoDB

## Features

- TV Anime Recommender: Get suggestions for similar TV anime based on your input.
- Movie Anime Recommender: Discover movie anime recommendations similar to your favorites.
- MAL-based Recommender: Receive personalized recommendations based on your MyAnimeList profile.

The TV and movie anime recommenders are based on machine learning cf model, the model training code is in the following repository:
- [Anime-Recommendation-Model](github.com/sl-shen/Anime-Recommendation-Model)

The MAL-based recommender retrieves the most similar mal user and recommend anime by comparing the anime lists.


