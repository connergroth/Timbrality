import { NextRequest, NextResponse } from 'next/server'
import { generateState, createSpotifyAuthUrl } from '@/lib/auth'

export async function GET(request: NextRequest) {
  const clientId = process.env.SPOTIFY_CLIENT_ID
  const redirectUri = process.env.SPOTIFY_REDIRECT_URI
  
  if (!clientId || !redirectUri) {
    console.error('Missing Spotify environment variables:', { 
      hasClientId: !!clientId, 
      hasRedirectUri: !!redirectUri 
    })
    return NextResponse.json(
      { error: 'Spotify configuration missing. Please check your environment variables.' },
      { status: 500 }
    )
  }

  const state = generateState()
  const authUrl = createSpotifyAuthUrl(clientId, redirectUri, state)

  // Store state in a cookie for verification
  const response = NextResponse.redirect(authUrl)
  response.cookies.set('spotify_state', state, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 10 // 10 minutes
  })

  return response
} 