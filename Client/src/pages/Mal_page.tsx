import React from 'react';
import { Link } from 'react-router-dom';
import AnimeRecommendations from '../component/Mal_compotent'
const mal = () => {
    return (
        <div>
            <h2>Mal-based Recommendation</h2>
            <AnimeRecommendations/>
            <Link to="/">Go back to Home</Link>
        </div>
    )
}

export default mal;