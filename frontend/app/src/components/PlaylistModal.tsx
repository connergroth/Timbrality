import { useState, useEffect } from 'react';
import { X, Plus, ListMusic, ExternalLink, Loader2, Check, AlertTriangle } from 'lucide-react';
import { hasRequiredSpotifyScopes, forceSpotifyReauth } from '@/lib/spotify-auth';

interface Playlist {
  id: string;
  name: string;
  description: string;
  total_tracks: number;
  cover_art?: string;
  spotify_url: string;
  owner: string;
}

interface PlaylistModalProps {
  isOpen: boolean;
  onClose: () => void;
  trackId: string;
  trackName: string;
  artistName: string;
  userId: string;
  user?: any; // Add user prop for scope checking
}

export function PlaylistModal({
  isOpen,
  onClose,
  trackId,
  trackName,
  artistName,
  userId,
  user
}: PlaylistModalProps) {
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [loading, setLoading] = useState(false);
  const [adding, setAdding] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [newPlaylistName, setNewPlaylistName] = useState('');
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [authError, setAuthError] = useState(false);

  // Fetch user playlists when modal opens
  useEffect(() => {
    if (isOpen && playlists.length === 0) {
      fetchPlaylists();
    }
  }, [isOpen]);

  const fetchPlaylists = async () => {
    setLoading(true);
    setAuthError(false);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_AGENT_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/agent/spotify-playlist`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'list_playlists',
          user_id: userId,
          limit: 50
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setPlaylists(data.results || []);
      } else if (response.status === 401 || response.status === 403) {
        // Authentication error - likely missing scopes
        setAuthError(true);
        setPlaylists([]);
      } else {
        console.error('Failed to fetch playlists');
        setPlaylists([]);
      }
    } catch (error) {
      console.error('Error fetching playlists:', error);
      setPlaylists([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSpotifyReauth = async () => {
    try {
      await forceSpotifyReauth();
    } catch (error) {
      console.error('Error re-authenticating Spotify:', error);
    }
  };

  const addToPlaylist = async (playlistId: string, playlistName: string) => {
    setAdding(playlistId);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_AGENT_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/agent/spotify-playlist`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'add_tracks_to_playlist',
          user_id: userId,
          playlist_id: playlistId,
          track_ids: [trackId]
        }),
      });

      if (response.ok) {
        setSuccessMessage(`Added "${trackName}" to "${playlistName}"`);
        setTimeout(() => {
          setSuccessMessage('');
          onClose();
        }, 2000);
      } else {
        console.error('Failed to add track to playlist');
      }
    } catch (error) {
      console.error('Error adding track to playlist:', error);
    } finally {
      setAdding(null);
    }
  };

  const createPlaylist = async () => {
    if (!newPlaylistName.trim()) return;
    
    setCreating(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_AGENT_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/agent/spotify-playlist`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'create_playlist',
          user_id: userId,
          playlist_name: newPlaylistName.trim(),
          description: `Created from Timbrality recommendations`,
          public: false,
          track_ids: [trackId]
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setSuccessMessage(`Created "${newPlaylistName}" and added "${trackName}"`);
        setNewPlaylistName('');
        setShowCreateForm(false);
        // Refresh playlists
        fetchPlaylists();
        setTimeout(() => {
          setSuccessMessage('');
          onClose();
        }, 2000);
      } else {
        console.error('Failed to create playlist');
      }
    } catch (error) {
      console.error('Error creating playlist:', error);
    } finally {
      setCreating(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-background border border-border rounded-lg p-6 w-full max-w-md mx-4 max-h-[80vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-inter font-semibold">Add to Playlist</h2>
            <p className="text-sm text-muted-foreground">
              "{trackName}" by {artistName}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-muted rounded transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Success Message */}
        {successMessage && (
          <div className="mb-4 p-3 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-400 rounded-lg flex items-center space-x-2">
            <Check className="w-4 h-4" />
            <span className="text-sm font-inter">{successMessage}</span>
          </div>
        )}

        {/* Create New Playlist */}
        <div className="mb-4">
          {!showCreateForm ? (
            <button
              onClick={() => setShowCreateForm(true)}
              className="w-full flex items-center space-x-2 p-3 border border-dashed border-border rounded-lg hover:bg-muted transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span className="font-inter text-sm">Create New Playlist</span>
            </button>
          ) : (
            <div className="border border-border rounded-lg p-3">
              <input
                type="text"
                placeholder="Playlist name"
                value={newPlaylistName}
                onChange={(e) => setNewPlaylistName(e.target.value)}
                className="w-full mb-3 p-2 border border-border rounded bg-background text-foreground placeholder:text-muted-foreground"
                onKeyPress={(e) => e.key === 'Enter' && createPlaylist()}
              />
              <div className="flex space-x-2">
                <button
                  onClick={createPlaylist}
                  disabled={!newPlaylistName.trim() || creating}
                  className="flex-1 bg-primary text-primary-foreground px-3 py-2 rounded text-sm font-inter disabled:opacity-50 flex items-center justify-center space-x-1"
                >
                  {creating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
                  <span>Create & Add</span>
                </button>
                <button
                  onClick={() => {
                    setShowCreateForm(false);
                    setNewPlaylistName('');
                  }}
                  className="px-3 py-2 border border-border rounded text-sm font-inter hover:bg-muted transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Existing Playlists */}
        <div>
          <h3 className="text-sm font-inter font-medium mb-3 text-muted-foreground">
            Your Playlists
          </h3>
          
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : playlists.length > 0 ? (
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {playlists.map((playlist) => (
                <div
                  key={playlist.id}
                  className="flex items-center space-x-3 p-3 border border-border rounded-lg hover:bg-muted transition-colors"
                >
                  {/* Playlist Cover */}
                  <div className="w-10 h-10 bg-muted rounded flex-shrink-0 overflow-hidden">
                    {playlist.cover_art ? (
                      <img 
                        src={playlist.cover_art} 
                        alt={playlist.name}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <ListMusic className="w-4 h-4 text-muted-foreground" />
                      </div>
                    )}
                  </div>

                  {/* Playlist Info */}
                  <div className="flex-1 min-w-0">
                    <h4 className="font-inter font-medium text-sm truncate">
                      {playlist.name}
                    </h4>
                    <p className="text-xs text-muted-foreground">
                      {playlist.total_tracks} tracks • {playlist.owner}
                    </p>
                  </div>

                  {/* Add Button */}
                  <button
                    onClick={() => addToPlaylist(playlist.id, playlist.name)}
                    disabled={adding === playlist.id}
                    className="px-3 py-1 bg-primary text-primary-foreground rounded text-xs font-inter hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center space-x-1"
                  >
                    {adding === playlist.id ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      <Plus className="w-3 h-3" />
                    )}
                    <span>Add</span>
                  </button>
                </div>
              ))}
            </div>
          ) : authError ? (
            <div className="text-center py-8">
              <AlertTriangle className="w-12 h-12 mx-auto mb-2 text-yellow-500" />
              <p className="text-sm font-inter text-foreground mb-2">Spotify Authentication Required</p>
              <p className="text-xs text-muted-foreground mb-4">
                Your Spotify account needs to be re-authenticated with playlist permissions.
              </p>
              <button
                onClick={handleSpotifyReauth}
                className="px-4 py-2 bg-primary text-primary-foreground rounded text-sm font-inter hover:bg-primary/90 transition-colors"
              >
                Re-authenticate Spotify
              </button>
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <ListMusic className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm font-inter">No playlists found</p>
              <p className="text-xs">Create your first playlist above</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}