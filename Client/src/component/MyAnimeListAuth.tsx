import React, { useEffect, useState } from 'react';
import * as crypto from 'crypto-js';

// 配置信息
const CLIENT_ID = 'c8645b434e72effbc3206f6757c2ec34';
const REDIRECT_URI = 'http://localhost:5173/mal'; 
const BACKEND_URL = 'http://127.0.0.1:8000';

interface AuthState {
  isAuthenticated: boolean;
  accessToken: string | null;
  userName: string | null;
}

const MyAnimeListAuth: React.FC = () => {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    accessToken: null,
    userName: null,
  });

  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');
    
    if (code) {
      handleCallback(code);
    }
  }, []);

  // 1. Generate a new Code Verifier / Code Challenge
  const getNewCodeVerifier = (): string => {
    const token = crypto.lib.WordArray.random(100);
    return crypto.enc.Base64url.stringify(token).slice(0, 128);
  };

  // 2. Print the URL needed to authorise your application
  const getNewAuthorisationUrl = (codeChallenge: string): string => {
    return `https://myanimelist.net/v1/oauth2/authorize?response_type=code&client_id=${CLIENT_ID}&code_challenge=${codeChallenge}&redirect_uri=${encodeURIComponent(REDIRECT_URI)}`;
  };

  // 3. Generate new token (now using backend)
  const generateNewToken = async (authorisationCode: string, codeVerifier: string): Promise<any> => {
    const url = `${BACKEND_URL}/api/oauth/token`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        code: authorisationCode,
        codeVerifier: codeVerifier,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to exchange token');
    }

    return await response.json();
  };

  const handleLogin = () => {
    const codeVerifier = getNewCodeVerifier();
    const authUrl = getNewAuthorisationUrl(codeVerifier);

    localStorage.setItem('codeVerifier', codeVerifier);
    window.location.href = authUrl;
  };

  const handleCallback = async (code: string) => {
    const codeVerifier = localStorage.getItem('codeVerifier');
    if (!codeVerifier) {
      console.error('No code verifier found');
      return;
    }

    try {
      const data = await generateNewToken(code, codeVerifier);

      setAuthState({
        isAuthenticated: true,
        accessToken: data.access_token,
        userName: data.user.name,
      });

      localStorage.removeItem('codeVerifier');
    } catch (error) {
      console.error('Error during authentication:', error);
    }
  };

  return (
    <div>
      <h1>MyAnimeList OAuth Login</h1>
      {!authState.isAuthenticated ? (
        <button onClick={handleLogin}>Login with MyAnimeList</button>
      ) : (
        <div>
          <p>Authenticated!</p>
          <p>Welcome, {authState.userName}!</p>
          <p>Access Token: {authState.accessToken}</p>
        </div>
      )}
    </div>
  );
};

export default MyAnimeListAuth;