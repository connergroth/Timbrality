import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { getLastfmSession } from '@/lib/auth'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const token = searchParams.get('token')
  const error = searchParams.get('error')

  // Check for errors
  if (error) {
    return NextResponse.redirect(new URL('/auth?error=lastfm_denied', request.url))
  }

  if (!token) {
    return NextResponse.redirect(new URL('/auth?error=invalid_request', request.url))
  }

  try {
    const apiKey = process.env.LASTFM_API_KEY

    if (!apiKey) {
      console.error('Missing Last.fm environment variables in callback')
      return NextResponse.redirect(new URL('/auth?error=configuration_error', request.url))
    }

    // Get session key from Last.fm
    const sessionData = await getLastfmSession(token, apiKey)
    const { session: { key: sessionKey, name: username } } = sessionData

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
        lastfm_username: username,
        lastfm_session_key: sessionKey,
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
    response.cookies.delete('lastfm_state')
    
    return response

  } catch (error) {
    console.error('Last.fm callback error:', error)
    return NextResponse.redirect(new URL('/auth?error=lastfm_error', request.url))
  }
} 