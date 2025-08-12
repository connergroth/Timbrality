create table public.tracks (
  id character varying not null default (gen_random_uuid ())::text,
  title character varying not null,
  artist character varying not null,
  album character varying null,
  genre character varying null,
  popularity integer null,
  aoty_score real null,
  cover_url character varying null,
  release_date date null,
  duration_ms integer null,
  genres text[] null default '{}'::text[],
  moods text[] null default '{}'::text[],
  spotify_url text null,
  explicit boolean null default false,
  track_number integer null,
  album_total_tracks integer null,
  created_at timestamp without time zone null default now(),
  updated_at timestamp without time zone null default now(),
  canonical_id text null,
  isrc text null,
  spotify_id text null,
  mb_recording_id uuid null,
  mb_release_id uuid null,
  pred_energy real null,
  pred_valence real null,
  mood_confidence real null,
  track_vector public.vector null,
  data_source_mask smallint null default 0,
  constraint tracks_pkey primary key (id),
  constraint tracks_canonical_id_key unique (canonical_id),
  constraint tracks_mb_recording_id_key unique (mb_recording_id),
  constraint tracks_mood_confidence_check check (
    (
      (mood_confidence >= (0)::double precision)
      and (mood_confidence <= (1)::double precision)
    )
  )
) TABLESPACE pg_default;

create index IF not exists idx_tracks_canonical_id on public.tracks using btree (canonical_id) TABLESPACE pg_default;

create index IF not exists idx_tracks_artist_title on public.tracks using btree (artist, title) TABLESPACE pg_default;

create index IF not exists idx_tracks_spotify_id on public.tracks using btree (spotify_id) TABLESPACE pg_default;

create index IF not exists idx_tracks_isrc on public.tracks using btree (isrc) TABLESPACE pg_default;

create index IF not exists idx_tracks_vector_cosine on public.tracks using ivfflat (track_vector vector_cosine_ops)
with
  (lists = '100') TABLESPACE pg_default;

create index IF not exists idx_tracks_moods_gin on public.tracks using gin (moods) TABLESPACE pg_default;

create index IF not exists idx_tracks_artist on public.tracks using btree (artist) TABLESPACE pg_default;

create index IF not exists idx_tracks_album on public.tracks using btree (album) TABLESPACE pg_default;

create trigger update_tracks_updated_at BEFORE
update on tracks for EACH row
execute FUNCTION update_updated_at_column ();