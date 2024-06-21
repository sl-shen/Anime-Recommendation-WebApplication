import React, { useEffect, useState } from 'react';
import * as crypto from 'crypto-js';

// 配置信息
const CLIENT_ID = 'c8645b434e72effbc3206f6757c2ec34';
const CLIENT_SECRET = '9454a54aae85aacaca134687576ebea38f045121787bb2b6afed61a01791d500';
const REDIRECT_URI = 'http://localhost:5173/mal_tv'; // 更改为你的实际回调 URL

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

  // 3. Generate new token
  const generateNewToken = async (authorisationCode: string, codeVerifier: string): Promise<any> => {
    const url = 'https://myanimelist.net/v1/oauth2/token';
    const data = {
      client_id: CLIENT_ID,
      client_secret: CLIENT_SECRET,
      code: authorisationCode,
      code_verifier: codeVerifier,
      grant_type: 'authorization_code',
      redirect_uri: REDIRECT_URI,
    };

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams(data),
    });

    if (!response.ok) {
      throw new Error('Failed to exchange token');
    }

    return await response.json();
  };

  // 4. Test the API by requesting user profile information
  const getUserInfo = async (accessToken: string): Promise<any> => {
    const url = 'https://api.myanimelist.net/v2/users/@me';
    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${accessToken}`
      }
    });

    if (!response.ok) {
      throw new Error('Failed to fetch user info');
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
      const token = await generateNewToken(code, codeVerifier);
      const userInfo = await getUserInfo(token.access_token);

      setAuthState({
        isAuthenticated: true,
        accessToken: token.access_token,
        userName: userInfo.name,
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