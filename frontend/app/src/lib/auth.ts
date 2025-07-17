import crypto from 'crypto'

export function generateState(): string {
  return crypto.randomBytes(32).toString('hex')
}

export function validateState(storedState: string, receivedState: string): boolean {
  return storedState === receivedState
}

export function createSpotifyAuthUrl(clientId: string, redirectUri: string, state: string): string {
  const scope = 'user-read-private user-read-email user-read-recently-played user-top-read playlist-read-private'
  
  const params = new URLSearchParams({
    client_id: clientId,
    response_type: 'code',
    redirect_uri: redirectUri,
    state: state,
    scope: scope
  })

  return `https://accounts.spotify.com/authorize?${params.toString()}`
}

export function createLastfmAuthUrl(apiKey: string, redirectUri: string): string {
  const params = new URLSearchParams({
    api_key: apiKey,
    cb: redirectUri
  })

  return `https://www.last.fm/api/auth?${params.toString()}`
}

export async function exchangeSpotifyCode(
  code: string,
  clientId: string,
  clientSecret: string,
  redirectUri: string
): Promise<{
  access_token: string
  refresh_token: string
  expires_in: number
}> {
  if (!clientId || !clientSecret || !redirectUri) {
    throw new Error('Missing Spotify configuration')
  }

  const response = await fetch('https://accounts.spotify.com/api/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Authorization': `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString('base64')}`
    },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: redirectUri
    })
  })

  if (!response.ok) {
    throw new Error(`Spotify token exchange failed: ${response.statusText}`)
  }

  return response.json()
}

export async function getSpotifyProfile(accessToken: string): Promise<{
  id: string
  display_name: string
  email: string
  images: Array<{ url: string }>
}> {
  const response = await fetch('https://api.spotify.com/v1/me', {
    headers: {
      'Authorization': `Bearer ${accessToken}`
    }
  })

  if (!response.ok) {
    throw new Error(`Failed to get Spotify profile: ${response.statusText}`)
  }

  return response.json()
}

export async function getLastfmSession(
  token: string,
  apiKey: string
): Promise<{
  session: {
    key: string
    name: string
  }
}> {
  if (!token || !apiKey) {
    throw new Error('Missing Last.fm configuration')
  }

  const response = await fetch('https://ws.audioscrobbler.com/2.0/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      method: 'auth.getSession',
      api_key: apiKey,
      token: token,
      format: 'json'
    })
  })

  if (!response.ok) {
    throw new Error(`Failed to get Last.fm session: ${response.statusText}`)
  }

  const data = await response.json()
  
  if (data.error) {
    throw new Error(`Last.fm error: ${data.message}`)
  }

  return data
} 