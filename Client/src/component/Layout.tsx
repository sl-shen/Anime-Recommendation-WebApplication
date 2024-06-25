import React from 'react';
import { Link } from 'react-router-dom';
import '../shared.css';

const Layout = ({ children }) => {
  return (
    <div>
      <nav>
        <ul>
          <li><Link to="/">Home</Link></li>
          <li><Link to="/tv">TV Anime</Link></li>
          <li><Link to="/movie">Movie Anime</Link></li>
          <li><Link to="/mal">MAL Recommendation</Link></li>
        </ul>
      </nav>
      <div className="page-content">
        {children}
      </div>
    </div>
  );
};

export default Layout;