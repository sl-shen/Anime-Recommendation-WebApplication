import React from 'react';
import { useBackground } from './BackgroundContext';
import { backgroundImages } from '../backgroundImages';

interface BackgroundManagerProps {
  children: React.ReactNode;
}

const BackgroundManager: React.FC<BackgroundManagerProps> = ({ children }) => {
  const { currentImageIndex } = useBackground();

  return (
    <div
      style={{
        backgroundImage: `url(${backgroundImages[currentImageIndex]})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed',
        minHeight: '100vh',
        transition: 'background-image 1s ease-in-out',
      }}
    >
      {children}
    </div>
  );
};

export default BackgroundManager;