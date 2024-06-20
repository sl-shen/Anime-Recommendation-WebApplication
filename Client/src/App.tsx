import React from 'react';
import { BrowserRouter as Router, Route, Link, Routes } from 'react-router-dom';
import TV from "./pages/TV_page";
import Movie from "./pages/Movie_page";
import Mal_TV from "./pages/Mal_TV_page";
import Mal_Movie from "./pages/Mal_Movie_page";

// Home component
const Home: React.FC = () => {
  return (
    <div>
      <h1>Anime Recommendation</h1>
      <nav>
        <ul>
          <li>
            <Link to="/tv">TV Anime</Link>
          </li>
          <li>
            <Link to="/movie">Movie Anime</Link>
          </li>
          <li>
            <Link to="/mal_tv">MAL TV Anime</Link>
          </li>
          <li>
            <Link to="/mal_movie">MAL Movie Anime</Link>
          </li>
        </ul>
      </nav>
    </div>
  );
};

// App component
const App: React.FC = () => {
  return (
    <Router>
      <div>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/tv" element={<TV />} />
          <Route path="/movie" element={<Movie />} />
          <Route path="/mal_tv" element={<Mal_TV />} />
          <Route path="/mal_movie" element={<Mal_Movie />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;