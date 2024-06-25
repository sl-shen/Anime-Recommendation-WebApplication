import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { Hint } from 'react-autocomplete-hint';

const Movie = () => {
  const [animeName, setAnimeName] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  const handleInputChange = async (event) => {
    const value = event.target.value;
    setAnimeName(value);

    if (value.length >= 2) {
      try {
        const response = await axios.get(`http://127.0.0.1:8000/autocomplete_movie?term=${encodeURIComponent(value)}`);
        setSuggestions(response.data);
        setShowSuggestions(true);
      } catch (error) {
        console.error('Error fetching suggestions:', error);
      }
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setAnimeName(suggestion);
    setShowSuggestions(false);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await axios.get(`http://127.0.0.1:8000/similar-animes-movie/${animeName}`);
      setRecommendations(response.data);
    } catch (error) {
      setError('Anime not found. Please try again.');
    }

    setIsLoading(false);
  };

  return (
    <div>
      <h2>Movie Anime Recommender</h2>
      <form onSubmit={handleSubmit}>
        <div style={{ position: 'relative' }}>
          <Hint options={suggestions} allowTabFill onHint={() => {}}>
            <input
              type="text"
              value={animeName}
              onChange={handleInputChange}
              placeholder="Enter TV anime name"
            />
          </Hint>
          {showSuggestions && (
            <ul style={{ position: 'absolute', top: '100%', left: 0, right: 0, background: 'white', listStyle: 'none', padding: 0, margin: 0 }}>
              {suggestions.map((suggestion, index) => (
                <li
                  key={index}
                  onClick={() => handleSuggestionClick(suggestion)}
                  style={{ padding: '5px', cursor: 'pointer', color: 'black' }}
                >
                  {suggestion}
                </li>
              ))}
            </ul>
          )}
        </div>
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
      
      <Link to="/">Go back to Home</Link>
    </div>
  );
};

export default Movie;