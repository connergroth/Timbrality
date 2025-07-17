import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { validateState, exchangeSpotifyCode, getSpotifyProfile } from '@/lib/auth'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const code = searchParams.get('code')
  const state = searchParams.get('state')
  const error = searchParams.get('error')

  // Check for errors
  if (error) {
    return NextResponse.redirect(new URL('/auth?error=spotify_denied', request.url))
  }

  if (!code || !state) {
    return NextResponse.redirect(new URL('/auth?error=invalid_request', request.url))
  }

  // Verify state parameter
  const storedState = request.cookies.get('spotify_state')?.value
  if (!storedState || !validateState(storedState, state)) {
    return NextResponse.redirect(new URL('/auth?error=invalid_state', request.url))
  }

  try {
    const clientId = process.env.SPOTIFY_CLIENT_ID
    const clientSecret = process.env.SPOTIFY_CLIENT_SECRET
    const redirectUri = process.env.SPOTIFY_REDIRECT_URI

    if (!clientId || !clientSecret || !redirectUri) {
      console.error('Missing Spotify environment variables in callback')
      return NextResponse.redirect(new URL('/auth?error=configuration_error', request.url))
    }

    // Exchange code for access token
    const tokenData = await exchangeSpotifyCode(code, clientId, clientSecret, redirectUri)
    const { access_token, refresh_token, expires_in } = tokenData

    // Get user profile from Spotify
    const profile = await getSpotifyProfile(access_token)

    // Get current user from Supabase
    const { data: { user }, error: authError } = await supabase.auth.getUser()
    
    if (authError || !user) {
      return NextResponse.redirect(new URL('/auth?error=not_authenticated', request.url))
    }

    // Update or create user record
    const { error: upsertError } = await supabase
      .from('users')
      .upsert({
        id: user.id,
        email: user.email!,
        username: user.email!.split('@')[0], // Use email prefix as username
        spotify_id: profile.id,
        spotify_access_token: access_token,
        spotify_refresh_token: refresh_token,
        spotify_token_expires_at: new Date(Date.now() + expires_in * 1000).toISOString(),
        display_name: profile.display_name,
        avatar_url: profile.images?.[0]?.url,
        updated_at: new Date().toISOString()
      }, {
        onConflict: 'id'
      })

    if (upsertError) {
      console.error('Error upserting user:', upsertError)
      return NextResponse.redirect(new URL('/auth?error=database_error', request.url))
    }

    // Clear the state cookie
    const response = NextResponse.redirect(new URL('/recommend', request.url))
    response.cookies.delete('spotify_state')
    
    return response

  } catch (error) {
    console.error('Spotify callback error:', error)
    return NextResponse.redirect(new URL('/auth?error=spotify_error', request.url))
  }
} 