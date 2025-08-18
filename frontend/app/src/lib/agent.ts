interface ChatRequest {
  message: string;
  user_id: string;
  session_id?: string;
  context?: {
    mood?: string;
  };
}

interface ChatResponse {
  response: string;
  tracks: Track[];
  explanations: string[];
  confidence: number;
  session_id: string;
  metadata: {
    tools_used: string[];
    context_updates: Record<string, any>;
    [key: string]: any;
  };
}

interface Track {
  id: string;
  name: string;
  artist: string;
  artists?: string[];
  album?: string;
  album_id?: string;
  artwork_url?: string;
  spotify_url?: string;
  preview_url?: string;
  source: string;
  similarity?: number;
  genres?: string[];
  audio_features?: Record<string, number>;
  
  // Spotify metadata
  duration_ms?: number;
  popularity?: number;
  release_date?: string;
  explicit?: boolean;
  spotify_id?: string;
  
  // Database ratings
  aoty_score?: number;
  rating?: number;
  
  // Artist data
  artist_id?: string;
  artist_image_url?: string;
}

interface FeedbackRequest {
  user_id: string;
  track_id: string;
  feedback_type: 'like' | 'dislike' | 'skip' | 'play_full';
  feedback_data?: Record<string, any>;
}

class AgentService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_AGENT_API_URL || 'http://localhost:8000';
  }

  /**
   * Send a chat message to the agent
   */
  async chat(request: ChatRequest): Promise<ChatResponse> {
    if (typeof fetch === 'undefined') {
      throw new Error('Fetch API not available');
    }

    const response = await fetch(`${this.baseUrl}/agent/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Agent chat failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Stream chat responses for real-time interaction
   */
  async *chatStream(request: ChatRequest): AsyncGenerator<any, void, unknown> {
    if (typeof fetch === 'undefined') {
      throw new Error('Fetch API not available');
    }

    const response = await fetch(`${this.baseUrl}/agent/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Agent chat stream failed: ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error('No response body for streaming');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              yield data;
            } catch (e) {
              // Skip malformed JSON
              console.warn('Malformed JSON in stream:', line);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Submit feedback for a track recommendation
   */
  async submitFeedback(request: FeedbackRequest): Promise<{ status: string; message: string }> {
    if (typeof fetch === 'undefined') {
      throw new Error('Fetch API not available');
    }

    const response = await fetch(`${this.baseUrl}/agent/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Feedback submission failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Analyze a Spotify playlist
   */
  async analyzePlaylist(playlistUrl: string, userId: string): Promise<any> {
    if (typeof fetch === 'undefined') {
      throw new Error('Fetch API not available');
    }

    const response = await fetch(`${this.baseUrl}/agent/analyze-playlist`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        playlist_url: playlistUrl,
        user_id: userId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Playlist analysis failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get user's profile and stats
   */
  async getUserProfile(userId: string): Promise<any> {
    if (typeof fetch === 'undefined') {
      throw new Error('Fetch API not available');
    }

    const response = await fetch(`${this.baseUrl}/agent/user/${userId}/profile`);

    if (!response.ok) {
      throw new Error(`Profile fetch failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get user's interaction history
   */
  async getUserHistory(userId: string, limit: number = 50): Promise<any> {
    if (typeof fetch === 'undefined') {
      throw new Error('Fetch API not available');
    }

    const response = await fetch(`${this.baseUrl}/agent/user/${userId}/history?limit=${limit}`);

    if (!response.ok) {
      throw new Error(`History fetch failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Add a listening event
   */
  async addListeningEvent(
    userId: string, 
    trackData: Record<string, any>, 
    eventType: string = 'play'
  ): Promise<{ status: string; message: string }> {
    if (typeof fetch === 'undefined') {
      throw new Error('Fetch API not available');
    }

    const response = await fetch(`${this.baseUrl}/agent/user/${userId}/listening-event`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ track_data: trackData, event_type: eventType }),
    });

    if (!response.ok) {
      throw new Error(`Listening event failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Generate a chat title based on the first message
   */
  async generateChatTitle(message: string, userId: string): Promise<{ title: string }> {
    if (typeof fetch === 'undefined') {
      throw new Error('Fetch API not available');
    }

    const response = await fetch(`${this.baseUrl}/agent/generate-title`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        conversation_text: message,
        user_id: userId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Title generation failed: ${response.statusText}`);
    }

    return response.json();
  }
}

export const agentService = new AgentService();
export type { ChatRequest, ChatResponse, Track, FeedbackRequest };