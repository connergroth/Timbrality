import { supabase } from './supabase'

export async function clearSpotifyAuth() {
  try {
    // Clear Spotify tokens from user metadata
    const { data: { user } } = await supabase.auth.getUser()
    if (user) {
      const { error } = await supabase.auth.updateUser({
        data: {
          spotify_access_token: null,
          spotify_refresh_token: null,
          spotify_expires_at: null,
          spotify_user_id: null
        }
      })
      
      if (error) {
        console.error('Error clearing Spotify tokens:', error)
        throw error
      }
    }
    
    // Clear any local storage items related to Spotify
    localStorage.removeItem('spotify_access_token')
    localStorage.removeItem('spotify_refresh_token')
    localStorage.removeItem('spotify_expires_at')
    localStorage.removeItem('spotify_user_id')
    
    return true
  } catch (error) {
    console.error('Error clearing Spotify authentication:', error)
    throw error
  }
}

export async function forceSpotifyReauth() {
  try {
    // Clear current Spotify auth
    await clearSpotifyAuth()
    
    // Sign out completely to force re-authentication
    const { error } = await supabase.auth.signOut()
    if (error) {
      console.error('Error signing out:', error)
      throw error
    }
    
    // Redirect to auth page for re-authentication
    window.location.href = '/auth'
    
    return true
  } catch (error) {
    console.error('Error forcing Spotify re-authentication:', error)
    throw error
  }
}

export function hasRequiredSpotifyScopes(user: any): boolean {
  if (!user?.user_metadata?.spotify_access_token) {
    return false
  }
  
  // Check if user has the required scopes by looking at the token
  // This is a simplified check - in a real app you'd decode the JWT token
  // For now, we'll assume if they have a token, they need to re-authenticate
  // with the new scopes
  return false
} 