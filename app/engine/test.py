from engine import fetch_user_liked_songs, fetch_user_playlists, fetch_recently_played, fetch_user_songs, insert_data_to_db, create_combined_dataset


# Initialize Spotify API (assuming you have an `sp` object configured already)
# sp = SpotifyAPIInstance()  # Make sure to configure this with the actual Spotify credentials

def test_insert_data():
    """Test function to fetch data from Spotify and Last.fm, combine, and insert into the database."""
    
    # Fetch data from Spotify
    print("Fetching Spotify data...")
    spotify_liked_songs = fetch_user_liked_songs(sp)
    spotify_playlists = fetch_user_playlists(sp)
    spotify_recently_played = fetch_recently_played(sp)
    
    # Fetch data from Last.fm
    print("Fetching Last.fm data...")
    lastfm_songs = fetch_user_songs("lastfm_username")  # Replace with an actual username

    # Combine Spotify and Last.fm data into a single dataset
    combined_data = create_combined_dataset(spotify_liked_songs, lastfm_songs)

    # Insert combined data into 'listening_histories' table
    print("Inserting combined data into 'listening_histories' table...")
    insert_data_to_db(combined_data, 'listening_histories')

    # Insert playlists data into 'playlists' table (optional)
    print("Inserting playlists data into 'playlists' table...")
    insert_data_to_db(spotify_playlists, 'playlists')

    # Insert recently played data into 'listening_histories' table (optional)
    print("Inserting recently played data into 'listening_histories' table...")
    insert_data_to_db(spotify_recently_played, 'listening_histories')

    print("Data insertion completed successfully.")

# Run the test function
test_insert_data()
