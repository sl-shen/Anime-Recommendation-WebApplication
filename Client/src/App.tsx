import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
  const [animeName, setAnimeName] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (event) => {
    setAnimeName(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await axios.get(`http://127.0.0.1:8000/similar-animes-tv/${animeName}`);
      setRecommendations(response.data);
    } catch (error) {
      setError('Anime not found. Please try again.');
    }

    setIsLoading(false);
  };

  return (
    <div>
      <h1>Anime Recommender</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={animeName}
          onChange={handleInputChange}
          placeholder="Enter anime name"
        />
        <button type="submit">Get Recommendations</button>
      </form>

      {isLoading && <p>Loading...</p>}
      {error && <p>{error}</p>}

      {recommendations.length > 0 && (
        <div>
          <h2>Recommended Animes:</h2>
          <ul>
            {recommendations.map((anime, index) => (
              <li key={index}>
                <h3>{anime.Name}</h3>
                <p>Similarity: {anime.Similarity}</p>
                <p>Genres: {anime.Genres}</p>
                <p>Synopsis: {anime.Synopsis}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default App;