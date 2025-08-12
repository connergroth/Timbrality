-- NUCLEAR OPTION: Clear all collaborative filtering data for fresh start
-- ⚠️  WARNING: This will delete ALL ingested user data! ⚠️
-- Run this in Supabase SQL editor ONLY if you want to start completely fresh

-- Uncomment the BEGIN; line below only when you're ready to execute
-- BEGIN;

-- Step 1: Clear all user interaction data (preserves table structure)
DELETE FROM public.user_track_interactions;
DELETE FROM public.user_album_interactions;
DELETE FROM public.user_artist_interactions;
DELETE FROM public.user_similarities;
DELETE FROM public.collaborative_recommendations;

-- Step 2: Clear user data (but keep table structure)
DELETE FROM public.lastfm_users;

-- Step 3: Clear content tables (tracks, albums, artists created during ingestion)
DELETE FROM public.tracks WHERE id LIKE 'lastfm_%';  -- Only Last.fm created tracks
DELETE FROM public.albums WHERE id LIKE 'album_%';   -- Only ingestion-created albums  
DELETE FROM public.artists WHERE id LIKE 'artist_%'; -- Only ingestion-created artists

-- Alternative: If you want to completely wipe tracks/albums/artists tables
-- (Uncomment these lines if you want to remove ALL content, not just ingestion-created)
-- DELETE FROM public.tracks;
-- DELETE FROM public.albums; 
-- DELETE FROM public.artists;

-- Step 4: Reset any sequences/auto-increment counters (if they exist)
-- This ensures clean ID generation on fresh start
DO $$
DECLARE
    seq_record RECORD;
BEGIN
    FOR seq_record IN 
        SELECT sequence_name 
        FROM information_schema.sequences 
        WHERE sequence_schema = 'public'
    LOOP
        EXECUTE format('ALTER SEQUENCE %I RESTART WITH 1', seq_record.sequence_name);
    END LOOP;
END $$;

-- Uncomment the COMMIT; line below only when you're ready to execute
-- COMMIT;

-- Verification queries (always safe to run)
SELECT 'Data verification after nuke:' as status;

SELECT 
    'lastfm_users' as table_name, 
    COUNT(*) as row_count 
FROM public.lastfm_users
UNION ALL
SELECT 
    'user_track_interactions' as table_name, 
    COUNT(*) as row_count 
FROM public.user_track_interactions
UNION ALL
SELECT 
    'user_album_interactions' as table_name, 
    COUNT(*) as row_count 
FROM public.user_album_interactions
UNION ALL
SELECT 
    'user_artist_interactions' as table_name, 
    COUNT(*) as row_count 
FROM public.user_artist_interactions
UNION ALL
SELECT 
    'tracks' as table_name, 
    COUNT(*) as row_count 
FROM public.tracks
UNION ALL
SELECT 
    'albums' as table_name, 
    COUNT(*) as row_count 
FROM public.albums
UNION ALL
SELECT 
    'artists' as table_name, 
    COUNT(*) as row_count 
FROM public.artists;

SELECT 'If all counts are 0, database is successfully nuked and ready for fresh ingestion!' as result;