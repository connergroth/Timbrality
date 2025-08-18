import { NextRequest, NextResponse } from 'next/server';

// Force this route to be dynamic
export const dynamic = 'force-dynamic';

interface SpotifyTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

interface SpotifyTrackSearchResponse {
  tracks: {
    items: Array<{
      id: string;
      name: string;
      artists: Array<{
        id: string;
        name: string;
        images?: Array<{
          url: string;
          height: number;
          width: number;
        }>;
      }>;
      album: {
        id: string;
        name: string;
        images: Array<{
          url: string;
          height: number;
          width: number;
        }>;
        release_date: string;
      };
      duration_ms: number;
      popularity: number;
      preview_url: string | null;
      external_urls: {
        spotify: string;
      };
    }>;
  };
}

// Get Spotify client credentials token
async function getSpotifyToken(): Promise<string | null> {
  const clientId = process.env.SPOTIFY_CLIENT_ID;
  const clientSecret = process.env.SPOTIFY_CLIENT_SECRET;

  if (!clientId || !clientSecret) {
    console.error('Missing Spotify client credentials');
    return null;
  }

  try {
    const response = await fetch('https://accounts.spotify.com/api/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString('base64')}`,
      },
      body: 'grant_type=client_credentials',
    });

    if (!response.ok) {
      throw new Error(`Failed to get Spotify token: ${response.statusText}`);
    }

    const data: SpotifyTokenResponse = await response.json();
    return data.access_token;
  } catch (error) {
    console.error('Error getting Spotify token:', error);
    return null;
  }
}

export async function GET(request: NextRequest) {
  try {
    const query = request.nextUrl.searchParams.get('q');

    if (!query) {
      return NextResponse.json(
        { error: 'Query parameter "q" is required' },
        { status: 400 }
      );
    }

    // Get access token
    const accessToken = await getSpotifyToken();
    if (!accessToken) {
      return NextResponse.json(
        { error: 'Failed to authenticate with Spotify' },
        { status: 500 }
      );
    }

    // Search for track
    const searchResponse = await fetch(
      `https://api.spotify.com/v1/search?q=${encodeURIComponent(query)}&type=track&limit=1`,
      {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      }
    );

    if (!searchResponse.ok) {
      throw new Error(`Spotify API error: ${searchResponse.statusText}`);
    }

    const searchData: SpotifyTrackSearchResponse = await searchResponse.json();
    const tracks = searchData.tracks.items;

    if (tracks.length === 0) {
      return NextResponse.json(
        { error: 'Track not found' },
        { status: 404 }
      );
    }

    const track = tracks[0];

    // Format response to match backend structure
    const response = {
      id: track.id,
      name: track.name,
      artist: track.artists[0]?.name,
      artists: track.artists.map(artist => artist.name),
      album: track.album.name,
      album_id: track.album.id,
      artwork_url: track.album.images[0]?.url || null,
      preview_url: track.preview_url,
      external_urls: track.external_urls,
      duration_ms: track.duration_ms,
      popularity: track.popularity,
      release_date: track.album.release_date,
    };

    return NextResponse.json(response);

  } catch (error) {
    console.error('Error searching track:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
