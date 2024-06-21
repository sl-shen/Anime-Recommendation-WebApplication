import React from 'react';
import MyAnimeListAuth from '../component/MyAnimeListAuth';
import { Link } from 'react-router-dom';

const mal = () => {
    return (
        <div>

            <MyAnimeListAuth />

            <Link to="/">Go back to Home</Link>
        </div>
    )
}

export default mal;