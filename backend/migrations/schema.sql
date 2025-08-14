-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.agent_memory (
  id bigint NOT NULL DEFAULT nextval('agent_memory_id_seq'::regclass),
  user_id uuid NOT NULL,
  kind text NOT NULL CHECK (kind = ANY (ARRAY['preference'::text, 'session_note'::text, 'feedback'::text, 'goal'::text, 'summary'::text, 'fact'::text])),
  content text NOT NULL,
  embedding USER-DEFINED,
  weight real DEFAULT 1.0,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT agent_memory_pkey PRIMARY KEY (id),
  CONSTRAINT agent_memory_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
CREATE TABLE public.album_compatibilities (
  id text NOT NULL,
  album_id text NOT NULL,
  compatibility_score integer NOT NULL,
  user_id_1 uuid NOT NULL,
  user_id_2 uuid NOT NULL,
  CONSTRAINT album_compatibilities_pkey PRIMARY KEY (id),
  CONSTRAINT album_compatibilities_user_id_2_fkey FOREIGN KEY (user_id_2) REFERENCES public.users(id),
  CONSTRAINT album_compatibilities_album_id_fkey FOREIGN KEY (album_id) REFERENCES public.albums(id),
  CONSTRAINT album_compatibilities_user_id_1_fkey FOREIGN KEY (user_id_1) REFERENCES public.users(id)
);
CREATE TABLE public.albums (
  id text NOT NULL,
  title text NOT NULL,
  artist text NOT NULL,
  release_date timestamp without time zone,
  genre text,
  aoty_score real,
  cover_url text,
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
  version_num text NOT NULL,
  CONSTRAINT alembic_version_pkey PRIMARY KEY (version_num)
);
CREATE TABLE public.aoty_attrs (
  song_id text NOT NULL,
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
  album_id text,
  username text NOT NULL,
  rating integer,
  review_text text NOT NULL,
  likes_count integer DEFAULT 0,
  created_at timestamp without time zone DEFAULT now(),
  updated_at timestamp without time zone DEFAULT now(),
  CONSTRAINT aoty_user_reviews_pkey PRIMARY KEY (id),
  CONSTRAINT aoty_user_reviews_album_id_fkey FOREIGN KEY (album_id) REFERENCES public.albums(id)
);
CREATE TABLE public.artist_compatibilities (
  id text NOT NULL,
  artist_id text NOT NULL,
  compatibility_score integer NOT NULL,
  user_id_1 uuid NOT NULL,
  user_id_2 uuid NOT NULL,
  CONSTRAINT artist_compatibilities_pkey PRIMARY KEY (id),
  CONSTRAINT artist_compatibilities_user_id_2_fkey FOREIGN KEY (user_id_2) REFERENCES public.users(id),
  CONSTRAINT artist_compatibilities_user_id_1_fkey FOREIGN KEY (user_id_1) REFERENCES public.users(id),
  CONSTRAINT artist_compatibilities_artist_id_fkey FOREIGN KEY (artist_id) REFERENCES public.artists(id)
);
CREATE TABLE public.artists (
  id text NOT NULL,
  name text NOT NULL,
  genre text,
  popularity integer,
  aoty_score integer,
  aoty_num_ratings integer DEFAULT 0,
  CONSTRAINT artists_pkey PRIMARY KEY (id)
);
CREATE TABLE public.chat_messages (
  id bigint NOT NULL DEFAULT nextval('chat_messages_id_seq'::regclass),
  user_id uuid NOT NULL,
  chat_id uuid NOT NULL,
  session_id text,
  message_type text NOT NULL CHECK (message_type = ANY (ARRAY['user'::text, 'agent'::text, 'system'::text, 'tool'::text])),
  content text NOT NULL,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT chat_messages_pkey PRIMARY KEY (id)
);
CREATE TABLE public.collaborative_recommendations (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  target_user_id uuid NOT NULL,
  track_id text NOT NULL,
  recommendation_score real NOT NULL CHECK (recommendation_score >= 0::double precision AND recommendation_score <= 1::double precision),
  algorithm_type text NOT NULL DEFAULT 'user_based'::character varying,
  confidence_score real DEFAULT 0.0 CHECK (confidence_score >= 0::double precision AND confidence_score <= 1::double precision),
  reason text,
  created_at timestamp without time zone DEFAULT now(),
  CONSTRAINT collaborative_recommendations_pkey PRIMARY KEY (id),
  CONSTRAINT collaborative_recommendations_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id),
  CONSTRAINT collaborative_recommendations_target_user_id_fkey FOREIGN KEY (target_user_id) REFERENCES public.lastfm_users(id)
);
CREATE TABLE public.compatibilities (
  id text NOT NULL,
  compatibility_score integer NOT NULL,
  user_id_1 uuid NOT NULL,
  user_id_2 uuid NOT NULL,
  CONSTRAINT compatibilities_pkey PRIMARY KEY (id),
  CONSTRAINT compatibilities_user_id_2_fkey FOREIGN KEY (user_id_2) REFERENCES public.users(id),
  CONSTRAINT compatibilities_user_id_1_fkey FOREIGN KEY (user_id_1) REFERENCES public.users(id)
);
CREATE TABLE public.data_fetch_logs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  lastfm_username text NOT NULL,
  fetch_type text NOT NULL,
  status text NOT NULL DEFAULT 'pending'::character varying,
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
  song_id text NOT NULL,
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
  lastfm_username text NOT NULL UNIQUE,
  display_name text,
  real_name text,
  country text,
  age integer,
  gender text,
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
  name text NOT NULL,
  track_ids json NOT NULL,
  cover_url text,
  CONSTRAINT playlists_pkey PRIMARY KEY (id)
);
CREATE TABLE public.rec_events (
  user_id uuid NOT NULL,
  item_id text NOT NULL,
  item_type text NOT NULL CHECK (item_type = ANY (ARRAY['track'::text, 'album'::text])),
  reason text,
  score real,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT rec_events_pkey PRIMARY KEY (user_id, item_id),
  CONSTRAINT rec_events_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
CREATE TABLE public.recommendations (
  id text NOT NULL,
  track_id text NOT NULL,
  album text NOT NULL,
  recommendation_score integer NOT NULL,
  user_id uuid NOT NULL,
  CONSTRAINT recommendations_pkey PRIMARY KEY (id),
  CONSTRAINT recommendations_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id),
  CONSTRAINT recommendations_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id),
  CONSTRAINT recommendations_album_fkey FOREIGN KEY (album) REFERENCES public.albums(id)
);
CREATE TABLE public.spotify_attrs (
  song_id text NOT NULL,
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
CREATE TABLE public.taste_snapshot (
  id bigint NOT NULL DEFAULT nextval('taste_snapshot_id_seq'::regclass),
  user_id uuid NOT NULL,
  summary text,
  embedding USER-DEFINED,
  source text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT taste_snapshot_pkey PRIMARY KEY (id),
  CONSTRAINT taste_snapshot_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
CREATE TABLE public.track_compatibilities (
  id text NOT NULL,
  track_id text NOT NULL,
  compatibility_score integer NOT NULL,
  user_id_1 uuid NOT NULL,
  user_id_2 uuid NOT NULL,
  CONSTRAINT track_compatibilities_pkey PRIMARY KEY (id),
  CONSTRAINT track_compatibilities_user_id_2_fkey FOREIGN KEY (user_id_2) REFERENCES public.users(id),
  CONSTRAINT track_compatibilities_user_id_1_fkey FOREIGN KEY (user_id_1) REFERENCES public.users(id),
  CONSTRAINT track_compatibilities_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id)
);
CREATE TABLE public.track_listening_histories (
  id text NOT NULL,
  track_id text NOT NULL,
  play_count integer NOT NULL,
  user_id uuid NOT NULL,
  CONSTRAINT track_listening_histories_pkey PRIMARY KEY (id),
  CONSTRAINT track_listening_histories_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id),
  CONSTRAINT track_listening_histories_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
CREATE TABLE public.track_ml_features (
  track_id text NOT NULL,
  track_vector USER-DEFINED,
  pred_energy real,
  pred_valence real,
  mood_conf real,
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT track_ml_features_pkey PRIMARY KEY (track_id),
  CONSTRAINT track_ml_features_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id)
);
CREATE TABLE public.track_tags (
  track_id text NOT NULL,
  tag text NOT NULL,
  weight real NOT NULL DEFAULT 1,
  CONSTRAINT track_tags_pkey PRIMARY KEY (track_id, tag),
  CONSTRAINT track_tags_track_id_fkey FOREIGN KEY (track_id) REFERENCES public.tracks(id)
);
CREATE TABLE public.tracks (
  id text NOT NULL DEFAULT (gen_random_uuid())::text,
  title text NOT NULL,
  artist text NOT NULL,
  album text,
  popularity integer,
  aoty_score real,
  cover_url text,
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
  id text,
  title text,
  artist text,
  album text,
  genre text,
  popularity integer,
  aoty_score integer,
  audio_features json,
  cover_url text
);
CREATE TABLE public.user_album_interactions (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  lastfm_user_id uuid NOT NULL,
  album_title text NOT NULL,
  album_artist text NOT NULL,
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
  artist_name text NOT NULL,
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
CREATE TABLE public.user_memories (
  id bigint NOT NULL DEFAULT nextval('user_memories_id_seq'::regclass),
  user_id uuid NOT NULL,
  chat_id uuid,
  kind text NOT NULL CHECK (kind = ANY (ARRAY['summary'::text, 'fact'::text, 'preference'::text, 'tool_output'::text, 'conversation'::text])),
  content text NOT NULL,
  embedding USER-DEFINED,
  importance smallint DEFAULT 1 CHECK (importance >= 1 AND importance <= 5),
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT user_memories_pkey PRIMARY KEY (id)
);
CREATE TABLE public.user_prefs (
  user_id uuid NOT NULL,
  top_genres ARRAY DEFAULT '{}'::text[],
  top_moods ARRAY DEFAULT '{}'::text[],
  artist_affinities jsonb DEFAULT '{}'::jsonb,
  depth_weight real DEFAULT 0.5,
  novelty_weight real DEFAULT 0.5,
  created_at timestamp with time zone DEFAULT now(),
  updated_at timestamp with time zone DEFAULT now(),
  CONSTRAINT user_prefs_pkey PRIMARY KEY (user_id),
  CONSTRAINT user_prefs_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
CREATE TABLE public.user_similarities (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id_1 uuid NOT NULL,
  user_id_2 uuid NOT NULL,
  similarity_score real NOT NULL CHECK (similarity_score >= '-1'::integer::double precision AND similarity_score <= 1::double precision),
  similarity_type text NOT NULL DEFAULT 'cosine'::character varying,
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
  track_id text NOT NULL,
  interaction_type text NOT NULL DEFAULT 'play'::character varying,
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
  username text NOT NULL UNIQUE,
  email text NOT NULL UNIQUE,
  password_hash text,
  provider text DEFAULT 'email'::character varying,
  created_at timestamp without time zone DEFAULT now(),
  updated_at timestamp without time zone DEFAULT now(),
  display_name text,
  avatar_url text,
  bio text,
  location text,
  spotify_id text UNIQUE,
  lastfm_username text UNIQUE,
  aoty_username text UNIQUE,
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