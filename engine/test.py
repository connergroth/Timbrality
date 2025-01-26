from recommender import APIRecommender  # Replace with your recommender module
import pandas as pd
from scipy.sparse import coo_matrix

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Define required Spotify scopes
SCOPE = "user-library-read user-read-recently-played playlist-read-private"

def main():
    print("Starting the recommendation system...")

    # Initialize Spotify client
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SCOPE
    ))

    # Instantiate the APIRecommender
    print("Initializing the APIRecommender...")
    recommender = APIRecommender(api_key=None)  # No Last.fm API key needed for Spotify-only testing

    # Ensure `fetch_user_matrix` is called
    print("Calling fetch_user_matrix...")
    try:
        user_matrix, mapped_data = recommender.fetch_user_matrix(
            users=["connergroth"],  # Replace with the appropriate username(s)
            data_type="songs",  # Testing with songs data
            sp=sp  # Pass the Spotify client
        )
        print("fetch_user_matrix executed successfully.")
    except Exception as e:
        print(f"Error while calling fetch_user_matrix: {e}")
        return

    # Check if data was returned
    if user_matrix is None or mapped_data is None:
        print("No data returned from fetch_user_matrix. Exiting...")
        return

    # Debugging: Print the first few rows of the mapped data
    print("Mapped Data (Preview):")
    print(mapped_data.head())

    # Train the recommendation model
    print("Training the recommendation model...")
    try:
        recommender.train_model(user_matrix)
        print("Model training complete.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # Debugging: Output the interaction matrix
    print("Interaction Matrix:")
    print(user_matrix.toarray())

    # Generate recommendations
    print("Generating recommendations...")
    try:
        recommendations = recommender.recommend(0, user_matrix, mapped_data, n=5)
        print("Recommendations:")
        for rec in recommendations:
            print(f"  {rec['name']} - Score: {rec['score']:.2f}")
    except Exception as e:
        print(f"Error while generating recommendations: {e}")

if __name__ == "__main__":
    main()
