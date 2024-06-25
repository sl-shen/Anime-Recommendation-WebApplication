import React from 'react';
import { Link } from 'react-router-dom';
import AnimeRecommendations from '../component/Mal_compotent'
import '../shared.css';
import Layout from '../component/Layout';

const mal = () => {
    return (
        <Layout>
            <div>
                <h1>Mal-Based Recommender</h1>
                <AnimeRecommendations/>
                <Link to="/">Go back to Home</Link>
            </div>
        </Layout>
        
    )
}

export default mal;