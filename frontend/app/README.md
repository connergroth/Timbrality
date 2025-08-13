# Timbrality App - Next.js OAuth Authentication

This is the Next.js application for Timbrality's authentication and recommendation interface. It handles OAuth flows for Spotify and Last.fm, and provides a user dashboard for managing connected services.

## Features

- **Supabase Authentication**: Email/password sign-in and sign-up
- **Spotify OAuth**: Connect Spotify account and store access tokens
- **Last.fm OAuth**: Connect Last.fm account and store session keys
- **User Dashboard**: View connected services and manage profile
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Environment Variables

Create a `.env.local` file in the root directory with the following variables:

```env
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url_here
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Spotify OAuth Configuration
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:3000/api/auth/spotify/callback

# Last.fm API Configuration
LASTFM_API_KEY=your_lastfm_api_key_here
LASTFM_SHARED_SECRET=your_lastfm_shared_secret_here
LASTFM_REDIRECT_URI=http://localhost:3000/api/auth/lastfm/callback

# App Configuration
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your_nextauth_secret_here
```

### 3. OAuth Setup

#### Spotify

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Add `http://localhost:3000/api/auth/spotify/callback` to Redirect URIs
4. Copy Client ID and Client Secret to environment variables

#### Last.fm

1. Go to [Last.fm API](https://www.last.fm/api/account/create)
2. Create a new API account
3. Copy API Key and Shared Secret to environment variables

### 4. Database Setup

Ensure your Supabase database has the `users` table with the following fields:

- `id` (uuid, primary key)
- `email` (varchar, unique)
- `username` (varchar, unique)
- `spotify_id` (varchar, nullable)
- `lastfm_username` (varchar, nullable)
- `spotify_access_token` (text, nullable)
- `spotify_refresh_token` (text, nullable)
- `spotify_token_expires_at` (timestamp, nullable)
- `lastfm_session_key` (text, nullable)
- And other profile fields as needed

### 5. Run Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

## API Routes

### Authentication Routes

- `GET /api/auth/spotify` - Initiates Spotify OAuth flow
- `GET /api/auth/spotify/callback` - Handles Spotify OAuth callback
- `GET /api/auth/lastfm` - Initiates Last.fm OAuth flow
- `GET /api/auth/lastfm/callback` - Handles Last.fm OAuth callback

### Pages

- `/` - Redirects to `/auth` or `/recommend` based on authentication status
- `/auth` - Authentication page with email/password and OAuth options
- `/recommend` - User dashboard showing connected services

## Architecture

- **Frontend**: Next.js 14 with App Router
- **Styling**: Tailwind CSS
- **Authentication**: Supabase Auth
- **Database**: Supabase PostgreSQL
- **OAuth**: Manual implementation using fetch API

## Security Features

- State parameter validation for OAuth flows
- Secure cookie storage for OAuth state
- Environment variable protection
- CSRF protection through state verification
- Secure token storage in database

## Deployment

This app can be deployed to Vercel, Netlify, or any other Next.js-compatible platform. Make sure to:

1. Set all environment variables in your deployment platform
2. Update OAuth redirect URIs to your production domain
3. Configure Supabase for production use
4. Set up proper CORS and security headers

## Development

### File Structure

```
src/
├── app/                    # Next.js App Router
│   ├── api/               # API routes
│   │   └── auth/          # OAuth endpoints
│   ├── auth/              # Authentication page
│   ├── recommend/         # User dashboard
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
├── components/            # React components
│   ├── SupabaseProvider.tsx
│   ├── ConnectSpotifyButton.tsx
│   └── ConnectLastfmButton.tsx
└── lib/                   # Utility libraries
    └── supabase.ts        # Supabase client
```

### Adding New OAuth Providers

To add a new OAuth provider:

1. Create API routes in `src/app/api/auth/[provider]/`
2. Add environment variables for the provider
3. Create a connection button component
4. Update the user table schema if needed
5. Add the provider to the dashboard

## Troubleshooting

### Common Issues

1. **OAuth redirect errors**: Check that redirect URIs match exactly
2. **Database errors**: Ensure Supabase is properly configured
3. **Environment variables**: Verify all required variables are set
4. **CORS issues**: Check Supabase CORS settings for your domain

### Debug Mode

Enable debug logging by setting `NODE_ENV=development` and checking browser console and server logs.
