-- Fix for the collaborative filtering views with correct PostgreSQL syntax

-- Drop the existing view if it exists
DROP VIEW IF EXISTS public.user_listening_stats;

-- 12. Create a view for easy access to user listening statistics (FIXED)
CREATE OR REPLACE VIEW public.user_listening_stats AS
SELECT 
    lu.id as user_id,
    lu.lastfm_username,
    lu.display_name,
    COUNT(DISTINCT uti.track_id) as unique_tracks_listened,
    COUNT(DISTINCT CONCAT(uai.album_title, '|', uai.album_artist)) as unique_albums_listened,
    COUNT(DISTINCT uari.artist_name) as unique_artists_listened,
    COALESCE(SUM(uti.play_count), 0) as total_track_plays,
    COALESCE(SUM(uai.play_count), 0) as total_album_plays,
    COALESCE(SUM(uari.play_count), 0) as total_artist_plays,
    COUNT(uti.id) FILTER (WHERE uti.user_loved = true) as loved_tracks_count,
    COUNT(uai.id) FILTER (WHERE uai.user_loved = true) as loved_albums_count,
    COUNT(uari.id) FILTER (WHERE uari.user_loved = true) as loved_artists_count,
    lu.last_updated
FROM public.lastfm_users lu
LEFT JOIN public.user_track_interactions uti ON lu.id = uti.lastfm_user_id
LEFT JOIN public.user_album_interactions uai ON lu.id = uai.lastfm_user_id
LEFT JOIN public.user_artist_interactions uari ON lu.id = uari.lastfm_user_id
WHERE lu.is_active = true
GROUP BY lu.id, lu.lastfm_username, lu.display_name, lu.last_updated;