import { NextRequest, NextResponse } from 'next/server'
import { generateState, createLastfmAuthUrl } from '@/lib/auth'

export async function GET(request: NextRequest) {
  const apiKey = process.env.LASTFM_API_KEY
  const redirectUri = process.env.LASTFM_REDIRECT_URI
  
  if (!apiKey || !redirectUri) {
    console.error('Missing Last.fm environment variables:', { 
      hasApiKey: !!apiKey, 
      hasRedirectUri: !!redirectUri 
    })
    return NextResponse.json(
      { error: 'Last.fm configuration missing. Please check your environment variables.' },
      { status: 500 }
    )
  }

  const state = generateState()
  const authUrl = createLastfmAuthUrl(apiKey, redirectUri)

  // Store state in a cookie for verification
  const response = NextResponse.redirect(authUrl)
  response.cookies.set('lastfm_state', state, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 10 // 10 minutes
  })

  return response
} 