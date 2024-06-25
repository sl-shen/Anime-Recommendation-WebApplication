import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import TV from "./pages/TV_page";
import Movie from "./pages/Movie_page";
import Mal from "./pages/Mal_page";
import './shared.css';
import Layout from './component/Layout';
import { BackgroundProvider } from './component/BackgroundContext';
import BackgroundManager from './component/BackgroundManager';

// Home component
const Home: React.FC = () => {
  return (
    <Layout>
      <div>
        <h1>Welcome to Anime Recommendation Platform <br /> ------------------------kksk!------------------------- </h1>
        
        <h2>We offer three unique features to help you discover new anime:<br /></h2>

        <h2>1. TV Anime Recommender:</h2>
          <h4>
             Simply enter the name of a TV anime you enjoy, and we'll suggest 10 similar shows. This feature supports English input only and includes anime up to and including 2023.<br />
          </h4>
         
        <h2>2. Movie Anime Recommender:</h2>
          <h4>
            Similar to our TV anime feature, this tool focuses on anime movies. Enter a movie title you like, and we'll recommend 10 similar anime films. Again, this feature supports English input and covers movies up to 2023.<br />
          </h4>

        <h2>3. MyAnimeList (MAL) Personalized Recommendations:</h2>
          <h4>
            For a more tailored experience, enter your MyAnimeList username. Our system will:<br /><br />
          - Find the user most similar to you based on your anime preferences.<br />
          - Analyze their anime list.<br />
          - Provide personalized recommendations based on what this similar user enjoys.<br />
          </h4>
         
        <h2>
          Whether you're looking for your next binge-worthy series, a great anime movie for movie night, or personalized recommendations based on your viewing history, we've got you covered. <br /><br />Start exploring and find your next favorite anime today!
        </h2>
      </div>
    </Layout>
   
  );
};

// App component
const App: React.FC = () => {
  return (
    <BackgroundProvider>
      <Router>
        <BackgroundManager>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/tv" element={<TV />} />
            <Route path="/movie" element={<Movie />} />
            <Route path="/mal" element={<Mal/>} />
          </Routes>
        </BackgroundManager>
      </Router>
    </BackgroundProvider>
  );
};

export default App;