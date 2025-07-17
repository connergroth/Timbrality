import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables. Please check your .env.local file.')
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

export type Database = {
  public: {
    Tables: {
      users: {
        Row: {
          id: string
          username: string
          email: string
          password_hash: string | null
          provider: string
          created_at: string
          updated_at: string
          display_name: string | null
          avatar_url: string | null
          bio: string | null
          location: string | null
          spotify_id: string | null
          lastfm_username: string | null
          aoty_username: string | null
          spotify_access_token: string | null
          spotify_refresh_token: string | null
          spotify_token_expires_at: string | null
          lastfm_session_key: string | null
          public_profile: boolean
          share_listening_history: boolean
          email_notifications: boolean
          is_active: boolean
          email_verified: boolean
          last_login: string | null
        }
        Insert: {
          id?: string
          username: string
          email: string
          password_hash?: string | null
          provider?: string
          created_at?: string
          updated_at?: string
          display_name?: string | null
          avatar_url?: string | null
          bio?: string | null
          location?: string | null
          spotify_id?: string | null
          lastfm_username?: string | null
          aoty_username?: string | null
          spotify_access_token?: string | null
          spotify_refresh_token?: string | null
          spotify_token_expires_at?: string | null
          lastfm_session_key?: string | null
          public_profile?: boolean
          share_listening_history?: boolean
          email_notifications?: boolean
          is_active?: boolean
          email_verified?: boolean
          last_login?: string | null
        }
        Update: {
          id?: string
          username?: string
          email?: string
          password_hash?: string | null
          provider?: string
          created_at?: string
          updated_at?: string
          display_name?: string | null
          avatar_url?: string | null
          bio?: string | null
          location?: string | null
          spotify_id?: string | null
          lastfm_username?: string | null
          aoty_username?: string | null
          spotify_access_token?: string | null
          spotify_refresh_token?: string | null
          spotify_token_expires_at?: string | null
          lastfm_session_key?: string | null
          public_profile?: boolean
          share_listening_history?: boolean
          email_notifications?: boolean
          is_active?: boolean
          email_verified?: boolean
          last_login?: string | null
        }
      }
    }
  }
} 