import React, { useState } from 'react';
import axios from 'axios';

const AnimeRecommendations = () => {
  const [username, setUsername] = useState('');
  const [similarUser, setSimilarUser] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [totalRecommendations, setTotalRecommendations] = useState(0);
  const [recommendationCount, setRecommendationCount] = useState(0);
  const [displayedRecommendations, setDisplayedRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSimilarUser('');
    setRecommendations([]);
    setTotalRecommendations(0);
    setDisplayedRecommendations([]);

    try {
      const response = await axios.get(`http://localhost:8000/anime-recommendations/${username}`);
      setSimilarUser(response.data.similar_user);
      setRecommendations(response.data.recommendations);
      setTotalRecommendations(response.data.recommendations.length);
    } catch (error) {
      setError('Error fetching recommendations. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleShowRecommendations = () => {
    setDisplayedRecommendations(recommendations.slice(0, recommendationCount));
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Enter your MAL username"
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Loading...' : 'Get Recommendations'}
        </button>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {similarUser && (
        <div>
          <h3>Similar User: {similarUser}</h3>
          <a 
            href={`https://myanimelist.net/profile/${similarUser}`} 
            target="_blank" 
            rel="noopener noreferrer"
          >
            View Profile
          </a>
        </div>
      )}

      {totalRecommendations > 0 && (
        <div>
          <h3>Total recommendations: {totalRecommendations}</h3>
          <input
            type="number"
            value={recommendationCount}
            onChange={(e) => setRecommendationCount(Math.min(parseInt(e.target.value), totalRecommendations))}
            placeholder="How many recommendations?"
            min="1"
            max={totalRecommendations}
          />
          <button onClick={handleShowRecommendations}>Show Recommendations</button>
        </div>
      )}

    {displayedRecommendations.length > 0 && (
      <div>
        <h3>Recommended Anime:</h3>
        <ul>
          {displayedRecommendations.map((anime: any, index) => (
            <li key={index}>
              {anime.title}
            </li>
          ))}
        </ul>
      </div>
    )}
    </div>
  );
};

export default AnimeRecommendations;