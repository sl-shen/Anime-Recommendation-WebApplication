import React from 'react';
import { BrowserRouter as Router, Route, Link, Routes } from 'react-router-dom';
import TV from "./pages/TV_page";
import Movie from "./pages/Movie_page";
import Mal from "./pages/Mal_page";
import './shared.css';
import Layout from './component/Layout';

// Home component
const Home: React.FC = () => {
  return (
    <Layout>
      <div>
        <h1>Anime Recommendation</h1>
      </div>
    </Layout>
   
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
          <Route path="/mal" element={<Mal/>} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;