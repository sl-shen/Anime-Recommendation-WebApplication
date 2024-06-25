import React, { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import BackgroundManager from './BackgroundManager';
import '../shared.css';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <BackgroundManager>
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
    </BackgroundManager>
  );
};

export default Layout;