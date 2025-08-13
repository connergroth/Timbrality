-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.album_compatibilities (
  id character varying NOT NULL,
  album_id character varying NOT NULL,
  compatibility_score integer NOT NULL,
  user_id_1 uuid NOT NULL,
  user_id_2 uuid NOT NULL,
  CONSTRAINT album_compatibilities_pkey PRIMARY KEY (id),
  CONSTRAINT album_compatibilities_album_id_fkey FOREIGN KEY (album_id) REFERENCES public.albums(id),
  CONSTRAINT album_compatibilities_user_id_2_fkey FOREIGN KEY (user_id_2) REFERENCES public.users(id),
  CONSTRAINT album_compatibilities_user_id_1_fkey FOREIGN KEY (user_id_1) REFERENCES public.users(id)
);
CREATE TABLE public.albums (
  id character varying NOT NULL,
  title character varying NOT NULL,
  artist character varying NOT NULL,
  release_date timestamp without time zone,
  genre character varying,
  aoty_score real,
  cover_url character varying,
  genres ARRAY DEFAULT '{}'::text[],
  duration_ms integer,
  total_tracks integer,
  spotify_url text,
  explicit boolean DEFAULT false,
  created_at timestamp without time zone DEFAULT now(),
  updated_at timestamp without time zone DEFAULT now(),
  aoty_num_ratings integer DEFAULT 0,
  CONSTRAINT albums_pkey PRIMARY KEY (id)
);
CREATE TABLE public.alembic_version (
  version_num character varying NOT NULL,
  CONSTRAINT alembic_version_pkey PRIMARY KEY (version_num)
);
CREATE TABLE public.aoty_attrs (
  song_id character varying NOT NULL,
  user_score numeric,
  rating_count integer,
  tags jsonb,
  genres jsonb,
  album_url text,
  album_title text,
  pulled_at timestamp without time zone DEFAULT now(),
  CONSTRAINT aoty_attrs_pkey PRIMARY KEY (song_id),
  CONSTRAINT aoty_attrs_song_id_fkey FOREIGN KEY (song_id) REFERENCES public.tracks(id)
);
CREATE TABLE public.aoty_user_reviews (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  album_id character varying,
  username character varying NOT NULL,
  rating integer,
  review_text text NOT NULL,
  likes_count integer DEFAULT 0,
  created_at timestamp without time zone DEFAULT now(),
  updated_at timestamp without time zone DEFAULT now(),
  CONSTRAINT aoty_user_reviews_pkey PRIMARY KEY (id),
  CONSTRAINT aoty_user_reviews_album_id_fkey FOREIGN KEY (album_id) REFERENCES public.albums(id)
);
CREATE TABLE public.artist_compatibilities (
  id character varying NOT NULL,
  artist_id character varying NOT NULL,
  compatibility_score integer NOT NULL,
  user_id_1 uuid NOT NULL,
  user_id_2 uuid NOT NULL,
  CONSTRAINT artist_compatibilities_pkey PRIMARY KEY (id),
  CONSTRAINT artist_compatibilities_artist_id_fkey FOREIGN KEY (artist_id) REFERENCES public.artists(id),
  CONSTRAINT artist_compatibilities_user_id_2_fkey FOREIGN KEY (user_id_2) REFERENCES public.users(id),
  CONSTRAINT artist_compatibilities_user_id_1_fkey FOREIGN KEY (user_id_1) REFERENCES public.users(id)
);
CREATE TABLE public.artists (
  id character varying NOT NULL,
  name character varying NOT NULL,
  genre character varying,
  popularity integer,
  aoty_score integer,
  aoty_num_ratings integer DEFAULT 0,
  CONSTRAINT artists_pkey PRIMARY KEY (id)
);
CREATE TABLE public.collaborative_recommendations (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  target_user_id uuid NOT NULL,
  track_id character varying NOT NULL,
  recommendation_score real NOT NULL CHECK (recommendation_score >= 0::double precision AND recommendation_score <= 1::double precision),
  algorithm_type character varying NOT NULL DEFAULT 'user_based'::character varying,
  confidence_score real DEFAULT 0.0 CHECK (confidence_score >= 0::double precision AND confidence_score <= 1::double precision),
  reason text,
  created_at timestamp without time zone DEFAULT now(),
  CONSTRAINT collaborative_recommendations_pkey PRIMARY KEY (id),
  CONSTRAINT collaborative_recommendations_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id),
  CONSTRAINT collaborative_recommendations_target_user_id_fkey FOREIGN KEY (target_user_id) REFERENCES public.lastfm_users(id)
);
CREATE TABLE public.compatibilities (
  id character varying NOT NULL,
  compatibility_score integer NOT NULL,
  user_id_1 uuid NOT NULL,
  user_id_2 uuid NOT NULL,
  CONSTRAINT compatibilities_pkey PRIMARY KEY (id),
  CONSTRAINT compatibilities_user_id_2_fkey FOREIGN KEY (user_id_2) REFERENCES public.users(id),
  CONSTRAINT compatibilities_user_id_1_fkey FOREIGN KEY (user_id_1) REFERENCES public.users(id)
);
CREATE TABLE public.data_fetch_logs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  lastfm_username character varying NOT NULL,
  fetch_type character varying NOT NULL,
  status character varying NOT NULL DEFAULT 'pending'::character varying,
  tracks_fetched integer DEFAULT 0,
  albums_fetched integer DEFAULT 0,
  artists_fetched integer DEFAULT 0,
  error_message text,
  started_at timestamp without time zone DEFAULT now(),
  completed_at timestamp without time zone,
  duration_ms integer,
  CONSTRAINT data_fetch_logs_pkey PRIMARY KEY (id)
);
CREATE TABLE public.lastfm_stats (
  song_id character varying NOT NULL,
  playcount integer NOT NULL,
  user_loved boolean,
  tags jsonb,
  pulled_at timestamp without time zone DEFAULT now(),
  mb_recording_id uuid,
  pulled_from text,
  CONSTRAINT lastfm_stats_pkey PRIMARY KEY (song_id),
  CONSTRAINT lastfm_stats_song_id_fkey FOREIGN KEY (song_id) REFERENCES public.tracks(id)
);
CREATE TABLE public.lastfm_users (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  lastfm_username character varying NOT NULL UNIQUE,
  display_name character varying,
  real_name character varying,
  country character varying,
  age integer,
  gender character varying,
  subscriber boolean DEFAULT false,
  playcount_total bigint DEFAULT 0,
  playlists_count integer DEFAULT 0,
  registered_date timestamp without time zone,
  last_updated timestamp without time zone DEFAULT now(),
  is_active boolean DEFAULT true,
  data_fetch_enabled boolean DEFAULT true,
  created_at timestamp without time zone DEFAULT now(),
  CONSTRAINT lastfm_users_pkey PRIMARY KEY (id)
);
CREATE TABLE public.playlists (
  id integer NOT NULL DEFAULT nextval('playlists_id_seq'::regclass),
  name character varying NOT NULL,
  track_ids json NOT NULL,
  cover_url character varying,
  CONSTRAINT playlists_pkey PRIMARY KEY (id)
);
CREATE TABLE public.recommendations (
  id character varying NOT NULL,
  track_id character varying NOT NULL,
  album character varying NOT NULL,
  recommendation_score integer NOT NULL,
  user_id uuid NOT NULL,
  CONSTRAINT recommendations_pkey PRIMARY KEY (id),
  CONSTRAINT recommendations_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id),
  CONSTRAINT recommendations_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id),
  CONSTRAINT recommendations_album_fkey FOREIGN KEY (album) REFERENCES public.albums(id)
);
CREATE TABLE public.spotify_attrs (
  song_id character varying NOT NULL,
  duration_ms integer,
  popularity integer,
  album_id text,
  artist_id text,
  album_name text,
  release_date text,
  explicit boolean,
  track_number integer,
  created_at timestamp without time zone DEFAULT now(),
  CONSTRAINT spotify_attrs_pkey PRIMARY KEY (song_id),
  CONSTRAINT spotify_attrs_song_id_fkey FOREIGN KEY (song_id) REFERENCES public.tracks(id)
);
CREATE TABLE public.track_compatibilities (
  id character varying NOT NULL,
  track_id character varying NOT NULL,
  compatibility_score integer NOT NULL,
  user_id_1 uuid NOT NULL,
  user_id_2 uuid NOT NULL,
  CONSTRAINT track_compatibilities_pkey PRIMARY KEY (id),
  CONSTRAINT track_compatibilities_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id),
  CONSTRAINT track_compatibilities_user_id_2_fkey FOREIGN KEY (user_id_2) REFERENCES public.users(id),
  CONSTRAINT track_compatibilities_user_id_1_fkey FOREIGN KEY (user_id_1) REFERENCES public.users(id)
);
CREATE TABLE public.track_listening_histories (
  id character varying NOT NULL,
  track_id character varying NOT NULL,
  play_count integer NOT NULL,
  user_id uuid NOT NULL,
  CONSTRAINT track_listening_histories_pkey PRIMARY KEY (id),
  CONSTRAINT track_listening_histories_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id),
  CONSTRAINT track_listening_histories_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
CREATE TABLE public.track_ml_features (
  track_id character varying NOT NULL,
  track_vector USER-DEFINED,
  pred_energy real,
  pred_valence real,
  mood_conf real,
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT track_ml_features_pkey PRIMARY KEY (track_id),
  CONSTRAINT track_ml_features_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id)
);
CREATE TABLE public.track_tags (
  track_id character varying NOT NULL,
  tag text NOT NULL,
  weight real NOT NULL DEFAULT 1,
  CONSTRAINT track_tags_pkey PRIMARY KEY (track_id, tag),
  CONSTRAINT track_tags_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id)
);
CREATE TABLE public.tracks (
  id character varying NOT NULL DEFAULT (gen_random_uuid())::text,
  title character varying NOT NULL,
  artist character varying NOT NULL,
  album character varying,
  popularity integer,
  aoty_score real,
  cover_url character varying,
  release_date date,
  duration_ms integer,
  genres ARRAY DEFAULT '{}'::text[],
  moods ARRAY DEFAULT '{}'::text[],
  spotify_url text,
  explicit boolean,
  track_number integer,
  album_total_tracks integer,
  created_at timestamp without time zone DEFAULT now(),
  updated_at timestamp without time zone DEFAULT now(),
  spotify_id text,
  pred_energy real,
  pred_valence real,
  mood_confidence real CHECK (mood_confidence >= 0::double precision AND mood_confidence <= 1::double precision),
  track_vector USER-DEFINED,
  data_source_mask smallint DEFAULT 0,
  aoty_num_ratings integer DEFAULT 0,
  CONSTRAINT tracks_pkey PRIMARY KEY (id)
);
CREATE TABLE public.tracks_backup (
  id character varying,
  title character varying,
  artist character varying,
  album character varying,
  genre character varying,
  popularity integer,
  aoty_score integer,
  audio_features json,
  cover_url character varying
);
CREATE TABLE public.user_album_interactions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  lastfm_user_id uuid NOT NULL,
  album_title character varying NOT NULL,
  album_artist character varying NOT NULL,
  play_count integer DEFAULT 0,
  user_loved boolean DEFAULT false,
  last_played timestamp without time zone,
  tags jsonb DEFAULT '{}'::jsonb,
  created_at timestamp without time zone DEFAULT now(),
  updated_at timestamp without time zone DEFAULT now(),
  CONSTRAINT user_album_interactions_pkey PRIMARY KEY (id),
  CONSTRAINT user_album_interactions_lastfm_user_id_fkey FOREIGN KEY (lastfm_user_id) REFERENCES public.lastfm_users(id)
);
CREATE TABLE public.user_artist_interactions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  lastfm_user_id uuid NOT NULL,
  artist_name character varying NOT NULL,
  play_count integer DEFAULT 0,
  user_loved boolean DEFAULT false,
  last_played timestamp without time zone,
  tags jsonb DEFAULT '{}'::jsonb,
  created_at timestamp without time zone DEFAULT now(),
  updated_at timestamp without time zone DEFAULT now(),
  CONSTRAINT user_artist_interactions_pkey PRIMARY KEY (id),
  CONSTRAINT user_artist_interactions_lastfm_user_id_fkey FOREIGN KEY (lastfm_user_id) REFERENCES public.lastfm_users(id)
);
CREATE TABLE public.user_id_mapping (
  old_id integer NOT NULL,
  new_id uuid NOT NULL UNIQUE,
  CONSTRAINT user_id_mapping_pkey PRIMARY KEY (old_id)
);
CREATE TABLE public.user_similarities (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id_1 uuid NOT NULL,
  user_id_2 uuid NOT NULL,
  similarity_score real NOT NULL CHECK (similarity_score >= '-1'::integer::double precision AND similarity_score <= 1::double precision),
  similarity_type character varying NOT NULL DEFAULT 'cosine'::character varying,
  shared_tracks_count integer DEFAULT 0,
  shared_albums_count integer DEFAULT 0,
  shared_artists_count integer DEFAULT 0,
  calculated_at timestamp without time zone DEFAULT now(),
  CONSTRAINT user_similarities_pkey PRIMARY KEY (id),
  CONSTRAINT user_similarities_user_id_2_fkey FOREIGN KEY (user_id_2) REFERENCES public.lastfm_users(id),
  CONSTRAINT user_similarities_user_id_1_fkey FOREIGN KEY (user_id_1) REFERENCES public.lastfm_users(id)
);
CREATE TABLE public.user_track_interactions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  lastfm_user_id uuid NOT NULL,
  track_id character varying NOT NULL,
  interaction_type character varying NOT NULL DEFAULT 'play'::character varying,
  play_count integer DEFAULT 0,
  user_loved boolean DEFAULT false,
  last_played timestamp without time zone,
  tags jsonb DEFAULT '{}'::jsonb,
  created_at timestamp without time zone DEFAULT now(),
  updated_at timestamp without time zone DEFAULT now(),
  CONSTRAINT user_track_interactions_pkey PRIMARY KEY (id),
  CONSTRAINT user_track_interactions_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id),
  CONSTRAINT user_track_interactions_lastfm_user_id_fkey FOREIGN KEY (lastfm_user_id) REFERENCES public.lastfm_users(id)
);
CREATE TABLE public.users (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  username character varying NOT NULL UNIQUE,
  email character varying NOT NULL UNIQUE,
  password_hash character varying,
  provider character varying DEFAULT 'email'::character varying,
  created_at timestamp without time zone DEFAULT now(),
  updated_at timestamp without time zone DEFAULT now(),
  display_name character varying,
  avatar_url character varying,
  bio text,
  location character varying,
  spotify_id character varying UNIQUE,
  lastfm_username character varying UNIQUE,
  aoty_username character varying UNIQUE,
  spotify_access_token text,
  spotify_refresh_token text,
  spotify_token_expires_at timestamp without time zone,
  lastfm_session_key text,
  public_profile boolean DEFAULT true,
  share_listening_history boolean DEFAULT false,
  email_notifications boolean DEFAULT true,
  is_active boolean DEFAULT true,
  email_verified boolean DEFAULT false,
  last_login timestamp without time zone,
  CONSTRAINT users_pkey PRIMARY KEY (id)
);