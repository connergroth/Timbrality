create table public.tracks (
  id character varying not null default (gen_random_uuid ())::text,
  title character varying not null,
  artist character varying not null,
  album character varying null,
  popularity integer null,
  aoty_score real null,
  cover_url character varying null,
  release_date date null,
  duration_ms integer null,
  genres text[] null default '{}'::text[],
  moods text[] null default '{}'::text[],
  spotify_url text null,
  explicit boolean null,
  track_number integer null,
  album_total_tracks integer null,
  created_at timestamp without time zone null default now(),
  updated_at timestamp without time zone null default now(),
  spotify_id text null,
  pred_energy real null,
  pred_valence real null,
  mood_confidence real null,
  track_vector public.vector null,
  data_source_mask smallint null default 0,
  aoty_num_ratings integer null default 0,
  constraint tracks_pkey primary key (id),
  constraint tracks_mood_confidence_check check (
    (
      (mood_confidence >= (0)::double precision)
      and (mood_confidence <= (1)::double precision)
    )
  )
) TABLESPACE pg_default;

create index IF not exists idx_tracks_artist_title on public.tracks using btree (artist, title) TABLESPACE pg_default;

create index IF not exists idx_tracks_spotify_id on public.tracks using btree (spotify_id) TABLESPACE pg_default;

create index IF not exists idx_tracks_vector_cosine on public.tracks using ivfflat (track_vector vector_cosine_ops)
with
  (lists = '100') TABLESPACE pg_default;

create index IF not exists idx_tracks_moods_gin on public.tracks using gin (moods) TABLESPACE pg_default;

create index IF not exists idx_tracks_artist on public.tracks using btree (artist) TABLESPACE pg_default;

create index IF not exists idx_tracks_album on public.tracks using btree (album) TABLESPACE pg_default;

create index IF not exists idx_tracks_genres_gin on public.tracks using gin (genres) TABLESPACE pg_default;

create index IF not exists idx_tracks_title_artist on public.tracks using btree (title, artist) TABLESPACE pg_default;

create index IF not exists idx_tracks_aoty_num_ratings on public.tracks using btree (aoty_num_ratings) TABLESPACE pg_default;

create trigger update_tracks_updated_at BEFORE
update on tracks for EACH row
execute FUNCTION update_updated_at_column ();